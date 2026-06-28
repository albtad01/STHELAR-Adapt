#!/usr/bin/env python3
"""Validate base + adapter weights against the original full checkpoint."""

import argparse
import sys
from pathlib import Path

import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from utils.adapter_checkpoint import (
    build_model,
    checkpoint_state_dict,
    load_adapter_state,
    load_yaml,
)


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-checkpoint", required=True)
    parser.add_argument("--adapter-checkpoint", required=True)
    parser.add_argument("--config", required=True)
    parser.add_argument(
        "--full-checkpoint",
        default=None,
        help="Override metadata.source_checkpoint for comparison.",
    )
    parser.add_argument(
        "--device",
        default="auto",
        help="auto, cpu, cuda, or a concrete device such as cuda:0.",
    )
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument(
        "--input-size",
        type=int,
        default=None,
        help="Synthetic validation input size; defaults to data.input_shape.",
    )
    parser.add_argument("--atol", type=float, default=1e-6)
    parser.add_argument("--rtol", type=float, default=1e-5)
    parser.add_argument(
        "--state-only",
        action="store_true",
        help="Verify every state tensor without running a forward pass.",
    )
    return parser.parse_args()


def resolve_path(path):
    path = Path(path).expanduser()
    if not path.is_absolute():
        path = REPO_ROOT / path
    return path.resolve()


def compare_state_dicts(full_model, adapter_model):
    full_state = full_model.state_dict()
    adapter_state = adapter_model.state_dict()
    if set(full_state) != set(adapter_state):
        raise RuntimeError("Model state dictionaries have different keys")
    differences = []
    for key in full_state:
        left = full_state[key]
        right = adapter_state[key]
        if torch.equal(left, right):
            continue
        if left.is_floating_point():
            max_difference = float((left - right).abs().max())
        else:
            max_difference = float((left != right).to(torch.float32).max())
        differences.append((key, max_difference))
    return differences


def run_forward(model, inputs, device):
    model.eval()
    model.to(device)
    inputs = inputs.to(device)
    with torch.inference_mode():
        outputs = model(inputs)
    return {key: value.detach().cpu() for key, value in outputs.items()}


def main():
    args = parse_args()
    base_checkpoint = resolve_path(args.base_checkpoint)
    adapter_checkpoint = resolve_path(args.adapter_checkpoint)
    config_path = resolve_path(args.config)
    config = load_yaml(config_path)
    adapter_payload = torch.load(str(adapter_checkpoint), map_location="cpu")
    if adapter_payload.get("format") != "cellvit_adapter_checkpoint":
        raise ValueError("Not a CellViT adapter-only checkpoint")

    metadata = adapter_payload["metadata"]
    if metadata["num_nuclei_classes"] != config["data"]["num_nuclei_classes"]:
        raise ValueError("num_nuclei_classes differs between config and adapter")
    if metadata["num_tissue_classes"] != config["data"]["num_tissue_classes"]:
        raise ValueError("num_tissue_classes differs between config and adapter")

    full_checkpoint = resolve_path(
        args.full_checkpoint or metadata["source_checkpoint"]
    )
    print("Building full-checkpoint reference model...")
    full_model, _, _ = build_model(config, base_checkpoint)
    full_payload = torch.load(str(full_checkpoint), map_location="cpu")
    full_model.load_state_dict(checkpoint_state_dict(full_payload), strict=True)
    del full_payload

    print("Building base + adapter model...")
    adapter_model, _, inserted = build_model(config, base_checkpoint)
    loaded_keys = load_adapter_state(adapter_model, adapter_payload)
    print(f"Loaded adapter/buffer tensors: {len(loaded_keys)}")
    print(f"Inserted decoder convolutional adapters: {len(inserted)}")

    differences = compare_state_dicts(full_model, adapter_model)
    if differences:
        print("State mismatch examples:")
        for key, difference in differences[:20]:
            print(f"  {key}: max_abs_diff={difference}")
        raise AssertionError(
            f"Base + adapter state differs from full checkpoint in {len(differences)} tensors"
        )
    print("State comparison: all tensors match exactly")

    if args.state_only:
        print("Forward comparison skipped (--state-only).")
        return

    if args.device == "auto":
        device = "cuda:0" if torch.cuda.is_available() else "cpu"
    else:
        device = args.device
    input_size = int(args.input_size or config["data"].get("input_shape", 256))
    generator = torch.Generator(device="cpu")
    generator.manual_seed(12345)
    inputs = torch.randn(
        args.batch_size,
        int(config["model"].get("input_channels", 3)),
        input_size,
        input_size,
        generator=generator,
    )

    full_outputs = run_forward(full_model, inputs, device)
    full_model.to("cpu")
    if str(device).startswith("cuda"):
        torch.cuda.empty_cache()
    adapter_outputs = run_forward(adapter_model, inputs, device)
    print("Output comparison:")
    failed = False
    for branch in sorted(full_outputs):
        max_difference = float(
            (full_outputs[branch] - adapter_outputs[branch]).abs().max()
        )
        matches = torch.allclose(
            full_outputs[branch],
            adapter_outputs[branch],
            atol=args.atol,
            rtol=args.rtol,
        )
        failed = failed or not matches
        print(
            f"  {branch}: max_abs_diff={max_difference:.8g} "
            f"allclose={matches}"
        )
    if failed:
        raise AssertionError("At least one output branch exceeds tolerance")
    print("Validation passed: full and base + adapter outputs match.")


if __name__ == "__main__":
    main()
