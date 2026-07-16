#!/usr/bin/env python3
"""Materialize a full CellViT checkpoint from a base model plus STHELAR adapter.

This is a compatibility bridge for the existing
`inference_cellvit_experiment_pannuke.py` runner, which expects a checkpoint
under RUN_DIR/checkpoints rather than a base checkpoint plus adapter alias.
"""

import argparse
import sys
from pathlib import Path

import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

import hub
from utils.adapter_checkpoint import load_yaml
from utils.tools import flatten_dict


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--run-dir",
        required=True,
        help="Timestamped run directory containing config.yaml and checkpoints/.",
    )
    parser.add_argument(
        "--adapter",
        default="ovary_5",
        help="Adapter alias, local adapter path, HF-style adapter directory, or HF repo id.",
    )
    parser.add_argument(
        "--base-checkpoint",
        default="models/pretrained/CellViT-SAM-H-x40.pth",
        help="Path to the CellViT-SAM-H-x40 pretrained checkpoint.",
    )
    parser.add_argument(
        "--out-name",
        default=None,
        help="Output checkpoint filename under RUN_DIR/checkpoints.",
    )
    parser.add_argument(
        "--prefer-hf",
        action="store_true",
        help="Resolve registered aliases through Hugging Face instead of local files.",
    )
    parser.add_argument("--revision", default=None, help="Optional HF revision.")
    parser.add_argument("--cache-dir", default=None, help="Optional HF cache directory.")
    return parser.parse_args()


def main():
    args = parse_args()
    run_dir = Path(args.run_dir).expanduser().resolve()
    config_path = run_dir / "config.yaml"
    checkpoint_dir = run_dir / "checkpoints"
    if not config_path.is_file():
        raise FileNotFoundError(f"Missing run config: {config_path}")
    if not checkpoint_dir.is_dir():
        raise FileNotFoundError(f"Missing checkpoint dir: {checkpoint_dir}")

    adapter_label = (
        str(args.adapter)
        .replace("/", "_")
        .replace(":", "_")
        .replace("\\", "_")
        .replace(".", "_")
    )
    out_name = args.out_name or f"{adapter_label}_materialized_for_inference.pth"
    if not out_name.endswith(".pth"):
        raise ValueError("--out-name must end with .pth")
    out_path = checkpoint_dir / out_name

    print(f"Run dir:          {run_dir}")
    print(f"Run config:       {config_path}")
    print(f"Base checkpoint:  {args.base_checkpoint}")
    print(f"Adapter:          {args.adapter}")
    print(f"Output checkpoint:{out_path}")

    cellvit = hub.model(
        "cellvit-sam-h-x40",
        base_checkpoint=args.base_checkpoint,
        config_path=config_path,
        device="cpu",
        eval_mode=True,
    )
    hub.load_sthelar_adapter(
        cellvit,
        args.adapter,
        print_metadata=False,
        prefer_hf=args.prefer_hf,
        revision=args.revision,
        cache_dir=args.cache_dir,
    )

    config = load_yaml(config_path)
    state = {
        "arch": type(cellvit).__name__,
        "epoch": None,
        "model_state_dict": cellvit.state_dict(),
        "optimizer_state_dict": None,
        "scheduler_state_dict": None,
        "best_metric": None,
        "best_epoch": None,
        "config": flatten_dict(config),
        "wandb_id": None,
        "logdir": str(run_dir),
        "run_name": run_dir.name,
        "scaler_state_dict": None,
        "materialized_from_adapter": str(args.adapter),
        "base_checkpoint": str(Path(args.base_checkpoint).expanduser().resolve()),
    }
    temporary = out_path.with_name(f".{out_path.name}.tmp")
    try:
        torch.save(state, temporary)
        temporary.replace(out_path)
    finally:
        temporary.unlink(missing_ok=True)

    size_mib = out_path.stat().st_size / (1024 ** 2)
    print(f"Materialized checkpoint written: {out_path}")
    print(f"Size: {size_mib:.2f} MiB")
    print()
    print("Run inference with:")
    print(
        "python cell_segmentation/inference/inference_cellvit_experiment_pannuke.py "
        f"--run_dir {run_dir} --checkpoint_name {out_path.name} --gpu 0 --magnification 40"
    )


if __name__ == "__main__":
    main()
