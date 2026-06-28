#!/usr/bin/env python3
"""Export trainable PEFT weights and changed mutable buffers from a full run."""

import argparse
import json
import sys
from pathlib import Path

import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from utils.adapter_checkpoint import (
    adapter_metadata,
    build_model,
    checkpoint_state_dict,
    load_adapter_state,
    load_yaml,
    resolve_checkpoint_path,
    resolve_run_dir,
)


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", required=True)
    parser.add_argument(
        "--checkpoint",
        required=True,
        help="Full checkpoint path or a filename under RUN_DIR/checkpoints.",
    )
    parser.add_argument("--out-path", required=True)
    parser.add_argument(
        "--base-checkpoint",
        default=None,
        help="Override model.pretrained from the run config.",
    )
    parser.add_argument(
        "--verify-load",
        action="store_true",
        help="Reload the exported file and compare every model state tensor.",
    )
    return parser.parse_args()


def export_adapter_checkpoint(
    run_dir,
    checkpoint_path,
    out_path,
    base_checkpoint=None,
    verify_load=False,
    extra_metadata=None,
):
    run_dir = resolve_run_dir(run_dir)
    config_path = run_dir / "config.yaml"
    config = load_yaml(config_path)
    checkpoint_path = resolve_checkpoint_path(run_dir, checkpoint_path)
    base_checkpoint = Path(
        base_checkpoint or config["model"]["pretrained"]
    ).expanduser()
    if not base_checkpoint.is_absolute():
        base_checkpoint = (REPO_ROOT / base_checkpoint).resolve()
    if not base_checkpoint.is_file():
        raise FileNotFoundError(f"Base checkpoint not found: {base_checkpoint}")

    print("Reconstructing base model and adapter architecture...")
    model, base_load_info, inserted = build_model(config, base_checkpoint)
    baseline_state = model.state_dict()
    trainable_names = {
        name for name, parameter in model.named_parameters() if parameter.requires_grad
    }
    buffer_names = {name for name, _ in model.named_buffers()}

    print(f"Loading full checkpoint: {checkpoint_path}")
    full_payload = torch.load(str(checkpoint_path), map_location="cpu")
    full_state = checkpoint_state_dict(full_payload)
    available_metrics = {
        "checkpoint": {
            "epoch": full_payload.get("epoch"),
            "best_epoch": full_payload.get("best_epoch"),
            "best_metric": full_payload.get("best_metric"),
        }
    }
    inference_results_path = run_dir / "inference_results.json"
    if inference_results_path.is_file():
        with inference_results_path.open() as handle:
            inference_results = json.load(handle)
        available_metrics["inference"] = inference_results.get("dataset")
    missing = sorted(set(baseline_state) - set(full_state))
    unexpected = sorted(set(full_state) - set(baseline_state))
    if missing or unexpected:
        raise RuntimeError(
            "Full checkpoint does not match the reconstructed architecture: "
            f"missing={missing[:10]} unexpected={unexpected[:10]}"
        )

    adapter_state = {
        name: full_state[name].detach().cpu().clone()
        for name in sorted(trainable_names)
    }
    changed_buffers = {
        name: full_state[name].detach().cpu().clone()
        for name in sorted(buffer_names)
        if name in full_state and not torch.equal(full_state[name], baseline_state[name])
    }

    # Exact deployment cannot silently omit a frozen parameter that changed.
    # Reproducing the training seed/init should make this list empty.
    frozen_parameter_changes = []
    for name, parameter in model.named_parameters():
        if name in trainable_names:
            continue
        if not torch.equal(full_state[name], baseline_state[name]):
            frozen_parameter_changes.append(name)
    if frozen_parameter_changes:
        raise RuntimeError(
            "Frozen original parameters differ from the reconstructed base model; "
            "refusing to include them in an adapter-only checkpoint. Examples: "
            f"{frozen_parameter_changes[:10]}"
        )

    metadata = adapter_metadata(
        config=config,
        run_dir=run_dir,
        checkpoint_path=checkpoint_path,
        base_checkpoint=base_checkpoint,
        model=model,
    )
    metadata.update(
        {
            "base_load_info": base_load_info,
            "inserted_decoder_conv_adapters": len(inserted),
            "adapter_state_tensor_count": len(adapter_state),
            "mutable_buffer_tensor_count": len(changed_buffers),
            "mutable_buffer_names": sorted(changed_buffers),
            "frozen_parameter_override_count": 0,
            "metrics": available_metrics,
        }
    )
    if extra_metadata:
        extra_metadata = dict(extra_metadata)
        extra_metrics = extra_metadata.pop("metrics", None)
        if extra_metrics:
            metadata["metrics"].update(extra_metrics)
        metadata.update(extra_metadata)
    output = {
        "format": "cellvit_adapter_checkpoint",
        "format_version": 1,
        "metadata": metadata,
        "adapter_state_dict": adapter_state,
        "mutable_buffer_state_dict": changed_buffers,
    }
    out_path = Path(out_path).expanduser()
    if not out_path.is_absolute():
        out_path = (REPO_ROOT / out_path).resolve()
    if out_path.suffix.lower() != ".pth":
        raise ValueError("Adapter checkpoint output must use the .pth extension")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(output, out_path)
    if verify_load:
        reloaded = torch.load(str(out_path), map_location="cpu")
        load_adapter_state(model, reloaded)
        mismatched = [
            key
            for key, value in model.state_dict().items()
            if not torch.equal(value, full_state[key])
        ]
        if mismatched:
            out_path.unlink(missing_ok=True)
            raise AssertionError(
                "Exported adapter does not reproduce the full checkpoint. "
                f"Mismatched tensors: {mismatched[:10]}"
            )
        metadata["load_verification"] = "exact_state_match"
        output["metadata"] = metadata
        torch.save(output, out_path)
        print("Adapter load verification: all state tensors match exactly")
    size_mib = out_path.stat().st_size / (1024 ** 2)
    print(f"Exported adapter checkpoint: {out_path}")
    print(f"Size: {size_mib:.2f} MiB")
    print(f"Trainable parameters: {metadata['trainable_parameter_count']:,}")
    print(f"Trainable ratio: {metadata['trainable_ratio_percent']:.4f}%")
    print(f"Adapter parameter tensors: {len(adapter_state)}")
    print(f"Changed mutable buffers: {len(changed_buffers)}")
    print("Frozen original parameter overrides: 0")
    return out_path


def main():
    args = parse_args()
    export_adapter_checkpoint(
        run_dir=args.run_dir,
        checkpoint_path=args.checkpoint,
        out_path=args.out_path,
        base_checkpoint=args.base_checkpoint,
        verify_load=args.verify_load,
    )


if __name__ == "__main__":
    main()
