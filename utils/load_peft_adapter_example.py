#!/usr/bin/env python3
"""Minimal CellViT PEFT adapter loading example."""

import json
import sys
from pathlib import Path

import torch


def _load_adapter_payload(path: Path) -> dict:
    if path.suffix == ".safetensors":
        from safetensors import safe_open

        adapter_state = {}
        mutable_buffers = {}
        metadata = {}
        with safe_open(str(path), framework="pt", device="cpu") as handle:
            raw_metadata = handle.metadata() or {}
            if raw_metadata.get("metadata_json"):
                metadata = json.loads(raw_metadata["metadata_json"])
            for key in handle.keys():
                tensor = handle.get_tensor(key)
                if key.startswith("adapter_state_dict."):
                    adapter_state[key.removeprefix("adapter_state_dict.")] = tensor
                elif key.startswith("mutable_buffer_state_dict."):
                    mutable_buffers[key.removeprefix("mutable_buffer_state_dict.")] = tensor
        return {
            "format": "cellvit_adapter_checkpoint",
            "metadata": metadata,
            "adapter_state_dict": adapter_state,
            "mutable_buffer_state_dict": mutable_buffers,
        }
    return torch.load(str(path), map_location="cpu")


def parse_args():
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-dir", required=True, help="HF adapter repo directory.")
    parser.add_argument(
        "--sthelar-repo",
        default=None,
        help="Path to a STHELAR-Adapt checkout. Defaults to current working directory.",
    )
    parser.add_argument(
        "--base-checkpoint",
        required=True,
        help="Path to models/pretrained/CellViT-SAM-H-x40.pth.",
    )
    parser.add_argument(
        "--adapter-weights",
        default=None,
        help="Override adapter weight path. Defaults to adapter_config.yaml entry.",
    )
    parser.add_argument(
        "--dummy-forward",
        action="store_true",
        help="Run a CPU dummy forward pass. This can be slow for SAM-H.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    sthelar_repo = Path(args.sthelar_repo or Path.cwd()).expanduser().resolve()
    if not (sthelar_repo / "utils" / "adapter_checkpoint.py").is_file():
        raise FileNotFoundError(
            "Could not find STHELAR-Adapt helper code. Pass --sthelar-repo /path/to/STHELAR-Adapt"
        )
    sys.path.insert(0, str(sthelar_repo))

    from utils.adapter_checkpoint import build_model, load_adapter_state, load_yaml

    repo_dir = Path(args.repo_dir).expanduser().resolve()
    adapter_config = load_yaml(repo_dir / "adapter_config.yaml")
    training_config = load_yaml(repo_dir / "training_config.yaml")
    training_config["model"]["pretrained"] = str(Path(args.base_checkpoint).expanduser().resolve())

    adapter_weights = Path(
        args.adapter_weights or repo_dir / adapter_config.get("adapter_weight_file", "adapter_model.pth")
    ).expanduser()
    if not adapter_weights.is_absolute():
        adapter_weights = (repo_dir / adapter_weights).resolve()

    model, base_load_info, inserted = build_model(
        training_config,
        Path(args.base_checkpoint).expanduser().resolve(),
    )
    payload = _load_adapter_payload(adapter_weights)
    loaded = load_adapter_state(model, payload)
    model.eval()

    print(f"Base tensors loaded: {base_load_info['loaded_tensors']}")
    print(f"Decoder conv adapters inserted: {len(inserted)}")
    print(f"Adapter tensors loaded: {len(loaded)}")
    print(f"Adapter weights: {adapter_weights}")

    if args.dummy_forward:
        input_shape = int(training_config["data"].get("input_shape", 256))
        sample = torch.zeros(1, 3, input_shape, input_shape)
        with torch.no_grad():
            output = model(sample)
        if isinstance(output, dict):
            print(f"Dummy forward output keys: {sorted(output)}")
        else:
            print(f"Dummy forward output type: {type(output).__name__}")


if __name__ == "__main__":
    main()
