#!/usr/bin/env python3
"""
Rerun CellViT inference from an existing training run without retraining.

The input can be either:
  - the timestamped log directory that contains config.yaml and checkpoints/; or
  - the parent run directory that contains a log/ subdirectory.
"""

import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))


def _latest_numeric_checkpoint(checkpoint_dir):
    checkpoints = list(checkpoint_dir.glob("checkpoint_*.pth"))
    if not checkpoints:
        return None

    def checkpoint_epoch(path):
        try:
            return int(path.stem.split("_")[1])
        except (IndexError, ValueError):
            return -1

    return max(checkpoints, key=checkpoint_epoch)


def resolve_run_dir(path):
    path = Path(path).expanduser().resolve()

    if (path / "config.yaml").is_file() and (path / "checkpoints").is_dir():
        return path

    log_dir = path / "log"
    if not log_dir.is_dir():
        raise FileNotFoundError(
            "Expected either a timestamped run directory with config.yaml/checkpoints "
            "or a parent run directory with a log/ subdirectory: {}".format(path)
        )

    candidates = [
        candidate
        for candidate in log_dir.iterdir()
        if candidate.is_dir()
        and (candidate / "config.yaml").is_file()
        and (candidate / "checkpoints").is_dir()
    ]
    if not candidates:
        raise FileNotFoundError(
            "No timestamped log directories with config.yaml/checkpoints found under {}".format(
                log_dir
            )
        )

    return sorted(candidates, key=lambda candidate: candidate.name)[-1]


def resolve_checkpoint_name(run_dir, checkpoint_name):
    checkpoint_dir = run_dir / "checkpoints"

    if checkpoint_name == "latest_checkpoint.pth":
        latest = _latest_numeric_checkpoint(checkpoint_dir)
        if latest is None:
            raise FileNotFoundError(
                "No checkpoint_*.pth files found under {}".format(checkpoint_dir)
            )
        return latest.name

    checkpoint_path = checkpoint_dir / checkpoint_name
    if not checkpoint_path.is_file():
        raise FileNotFoundError("Checkpoint not found: {}".format(checkpoint_path))

    return checkpoint_name


def parse_args():
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        description="Rerun CellViT inference from an existing run directory.",
    )
    parser.add_argument(
        "--run_dir",
        required=True,
        help="Timestamped log dir, or parent run dir containing log/<timestamped-run>.",
    )
    parser.add_argument(
        "--checkpoint_name",
        default="latest_checkpoint.pth",
        help="latest_checkpoint.pth, model_best.pth, checkpoint_10.pth, etc.",
    )
    parser.add_argument("--gpu", default="0", help="CUDA GPU id, or mps.")
    parser.add_argument(
        "--magnification",
        type=int,
        choices=[20, 40],
        default=40,
        help="Dataset magnification.",
    )
    parser.add_argument(
        "--cell_tokens",
        choices=["nucleus", "cell", "no"],
        default="no",
        help="Whether to save cell tokens.",
    )
    parser.add_argument(
        "--plots",
        action="store_true",
        help="Generate inference prediction plots in the run directory.",
    )
    parser.add_argument("--qc-metric", default=None)
    parser.add_argument("--qc-thresholds", nargs="+", default=None)
    parser.add_argument("--qc-bins", nargs="+", default=None)
    parser.add_argument(
        "--reuse-results",
        action="store_true",
        help="Build only QC sweep files from the existing inference_results.json.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    run_dir = resolve_run_dir(args.run_dir)
    checkpoint_name = resolve_checkpoint_name(run_dir, args.checkpoint_name)

    print("Resolved run_dir: {}".format(run_dir))
    print("Resolved checkpoint_name: {}".format(checkpoint_name))

    from cell_segmentation.inference.inference_cellvit_experiment_pannuke import (
        InferenceCellViT,
    )

    inference = InferenceCellViT(
        run_dir=run_dir,
        checkpoint_name=checkpoint_name,
        gpu=args.gpu,
        magnification=args.magnification,
        cell_tokens=args.cell_tokens,
        qc_metric=args.qc_metric,
        qc_thresholds=args.qc_thresholds,
        qc_bins=args.qc_bins,
    )
    if args.reuse_results:
        inference.run_qc_sweep_from_saved_results()
        return
    model, dataloader, dataset_config = inference.setup_patch_inference()
    inference.run_patch_inference(
        model,
        dataloader,
        dataset_config,
        generate_plots=args.plots,
    )


if __name__ == "__main__":
    main()
