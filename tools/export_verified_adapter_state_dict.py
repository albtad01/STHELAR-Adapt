#!/usr/bin/env python3
"""Export and verify a flat, deployment-only PEFT state_dict.

The artifact contains trainable PEFT parameters plus any mutable buffers that
changed while training.  It contains no optimizer, scheduler, scaler, epoch,
or canonical base-model tensors.  Canonical checkpoints are read only.
"""

import argparse
import hashlib
import json
import sys
from collections import Counter, OrderedDict
from pathlib import Path

import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from utils.adapter_checkpoint import (  # noqa: E402
    build_model,
    checkpoint_state_dict,
    load_yaml,
    resolve_checkpoint_path,
    resolve_run_dir,
)


def _component(name: str) -> str:
    if ".attn.qkv.adapter_" in name:
        return "LoRA"
    if ".mlp.adapter_" in name:
        return "AdaptFormer"
    if ".decoder0_header.2." in name:
        return "NP/HV/NT final heads"
    if name.startswith("classifier_head.") or name.startswith("encoder.head."):
        return "tissue classifier"
    return "other"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _compare_outputs(left, right, prefix="output"):
    rows = []
    if torch.is_tensor(left) and torch.is_tensor(right):
        if left.shape != right.shape:
            raise AssertionError(
                f"{prefix}: output shapes differ: {tuple(left.shape)} != {tuple(right.shape)}"
            )
        difference = (left - right).abs()
        rows.append(
            {
                "name": prefix,
                "shape": list(left.shape),
                "exact": bool(torch.equal(left, right)),
                "max_abs_difference": float(difference.max()) if difference.numel() else 0.0,
            }
        )
        return rows
    if isinstance(left, dict) and isinstance(right, dict):
        if set(left) != set(right):
            raise AssertionError(f"{prefix}: output dictionary keys differ")
        for key in sorted(left):
            rows.extend(_compare_outputs(left[key], right[key], f"{prefix}.{key}"))
        return rows
    if isinstance(left, (tuple, list)) and isinstance(right, type(left)):
        if len(left) != len(right):
            raise AssertionError(f"{prefix}: output sequence lengths differ")
        for index, (left_item, right_item) in enumerate(zip(left, right)):
            rows.extend(
                _compare_outputs(left_item, right_item, f"{prefix}[{index}]")
            )
        return rows
    if left != right:
        raise AssertionError(f"{prefix}: non-tensor outputs differ")
    return rows


def export(run_dir, checkpoint, output, verify_forward=False):
    run_dir = resolve_run_dir(run_dir)
    config = load_yaml(run_dir / "config.yaml")
    checkpoint_path = resolve_checkpoint_path(run_dir, checkpoint)
    base_checkpoint = Path(config["model"]["pretrained"])
    if not base_checkpoint.is_absolute():
        base_checkpoint = (REPO_ROOT / base_checkpoint).resolve()

    model, base_load_info, _ = build_model(config, base_checkpoint)
    base_state = model.state_dict()
    trainable_names = {
        name for name, parameter in model.named_parameters() if parameter.requires_grad
    }
    parameter_names = {name for name, _ in model.named_parameters()}
    buffer_names = {name for name, _ in model.named_buffers()}

    payload = torch.load(str(checkpoint_path), map_location="cpu")
    canonical_state = checkpoint_state_dict(payload)
    if set(canonical_state) != set(base_state):
        missing = sorted(set(base_state) - set(canonical_state))
        unexpected = sorted(set(canonical_state) - set(base_state))
        raise RuntimeError(
            f"Canonical architecture mismatch: missing={missing[:10]}, "
            f"unexpected={unexpected[:10]}"
        )

    frozen_parameter_changes = sorted(
        name
        for name in parameter_names - trainable_names
        if not torch.equal(canonical_state[name], base_state[name])
    )
    if frozen_parameter_changes:
        raise RuntimeError(
            "Frozen parameters changed; refusing to call the result adapter-only: "
            f"{frozen_parameter_changes[:10]}"
        )

    changed_buffers = sorted(
        name
        for name in buffer_names
        if not torch.equal(canonical_state[name], base_state[name])
    )
    artifact_state = OrderedDict(
        (name, canonical_state[name].detach().cpu().clone())
        for name in sorted(trainable_names | set(changed_buffers))
    )

    output = Path(output).expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(artifact_state, output)

    reloaded = torch.load(str(output), map_location="cpu", weights_only=True)
    incompatibility = model.load_state_dict(reloaded, strict=False)
    if incompatibility.unexpected_keys:
        raise AssertionError(
            f"Adapter has unexpected keys: {incompatibility.unexpected_keys[:10]}"
        )
    mismatched_state = sorted(
        name
        for name, value in model.state_dict().items()
        if not torch.equal(value, canonical_state[name])
    )
    if mismatched_state:
        raise AssertionError(
            "Base + adapter did not reconstruct the canonical state: "
            f"{mismatched_state[:10]}"
        )

    output_comparison = []
    if verify_forward:
        model.eval()
        input_size = int(config["data"].get("input_shape", 256))
        sample = torch.linspace(
            -1.0,
            1.0,
            steps=3 * input_size * input_size,
            dtype=torch.float32,
        ).reshape(1, 3, input_size, input_size)
        with torch.inference_mode():
            adapter_output = model(sample)
        model.load_state_dict(canonical_state, strict=True)
        model.eval()
        with torch.inference_mode():
            canonical_output = model(sample)
        output_comparison = _compare_outputs(adapter_output, canonical_output)
        if not all(row["exact"] for row in output_comparison):
            raise AssertionError("Adapter and canonical forward outputs are not exact")

    component_parameter_tensors = Counter(_component(name) for name in trainable_names)
    component_parameter_elements = Counter()
    for name in trainable_names:
        component_parameter_elements[_component(name)] += canonical_state[name].numel()

    metadata = {
        "format": "flat_adapter_only_state_dict",
        "format_version": 1,
        "backbone": config.get("efficiency", {}).get(
            "backbone", config["model"]["backbone"]
        ),
        "fold": config.get("efficiency", {}).get("fold"),
        "method": config.get("efficiency", {}).get("method"),
        "source_run_dir": str(run_dir),
        "source_checkpoint": str(checkpoint_path),
        "base_checkpoint": str(base_checkpoint),
        "base_load_info": base_load_info,
        "artifact_path": str(output),
        "artifact_size_bytes": output.stat().st_size,
        "artifact_sha256": _sha256(output),
        "parameter_tensor_count": len(trainable_names),
        "parameter_element_count": sum(
            canonical_state[name].numel() for name in trainable_names
        ),
        "component_parameter_tensor_counts": dict(component_parameter_tensors),
        "component_parameter_element_counts": dict(component_parameter_elements),
        "changed_mutable_buffer_tensor_count": len(changed_buffers),
        "changed_mutable_buffer_element_count": sum(
            canonical_state[name].numel() for name in changed_buffers
        ),
        "changed_mutable_buffer_names": changed_buffers,
        "frozen_parameter_override_count": 0,
        "state_reconstruction": "exact_all_tensors",
        "forward_verification": (
            "exact_all_output_tensors" if verify_forward else "not_requested"
        ),
        "forward_output_comparison": output_comparison,
    }
    metadata_path = output.with_suffix(".verification.json")
    metadata_path.write_text(json.dumps(metadata, indent=2, sort_keys=True) + "\n")
    print(json.dumps(metadata, indent=2, sort_keys=True))
    return metadata


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--checkpoint", default="checkpoint_10.pth")
    parser.add_argument("--output", required=True)
    parser.add_argument("--verify-forward", action="store_true")
    args = parser.parse_args()
    export(
        run_dir=args.run_dir,
        checkpoint=args.checkpoint,
        output=args.output,
        verify_forward=args.verify_forward,
    )


if __name__ == "__main__":
    main()
