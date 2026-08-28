#!/usr/bin/env python3
"""Evaluate an untouched pretrained CellViT with class-agnostic metrics only."""

import argparse
import hashlib
import json
import os
import sys
import tempfile
from pathlib import Path
from typing import Any, Dict

import torch
import yaml
from torch.utils.data import DataLoader, Subset

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from cell_segmentation.inference.inference_cellvit_experiment_pannuke import (
    InferenceCellViT,
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _atomic_yaml(path: Path, payload: Dict[str, Any]) -> None:
    temporary = path.with_name(".{}.tmp-{}".format(path.name, os.getpid()))
    try:
        with temporary.open("w") as handle:
            yaml.safe_dump(payload, handle, sort_keys=False)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(str(temporary), str(path))
    finally:
        if temporary.exists():
            temporary.unlink()


def _checkpoint_metadata(
    checkpoint: Dict[str, Any], expected_arch: str, expected_backbone: str
) -> Dict[str, Any]:
    flattened = checkpoint.get("config", {})
    required = {
        "arch": checkpoint.get("arch"),
        "dataset": flattened.get("data.dataset"),
        "num_nuclei_classes": flattened.get("data.num_nuclei_classes"),
        "num_tissue_classes": flattened.get("data.num_tissue_classes"),
        "backbone": flattened.get("model.backbone"),
        "normalize_mean": flattened.get(
            "transformations.normalize.mean", [0.5, 0.5, 0.5]
        ),
        "normalize_std": flattened.get(
            "transformations.normalize.std", [0.5, 0.5, 0.5]
        ),
    }
    if required["arch"] != expected_arch:
        raise ValueError(
            "Frozen checkpoint architecture mismatch: expected={} observed={}".format(
                expected_arch, required
            )
        )
    if str(required["backbone"]).lower() != str(expected_backbone).lower():
        raise ValueError(
            "Frozen checkpoint backbone mismatch: expected={} observed={}".format(
                expected_backbone, required
            )
        )
    if required["num_nuclei_classes"] is None or required["num_tissue_classes"] is None:
        raise ValueError("Checkpoint lacks original output taxonomy dimensions")
    return required


def _run_config(
    config: Dict[str, Any], checkpoint_meta: Dict[str, Any], output_dir: Path
) -> Dict[str, Any]:
    backbone_identity = config.get("backbone_identity")
    if backbone_identity is None:
        backbone_identity = (
            "CellViT-SAM-H x40"
            if str(checkpoint_meta["backbone"]).lower() == "sam-h"
            else "CellViT-256 x40"
        )
    return {
        "adapters": {"adapter_type": "frozen"},
        "logging": {
            "level": "debug",
            "log_comment": config["run_name"],
            "notes": (
                "Untouched pretrained {} x40 checkpoint. PanNuke semantic "
                "head retained; only class-agnostic metrics may be reported."
            ).format(checkpoint_meta["backbone"]),
            "log_dir": str(output_dir),
        },
        "random_seed": int(config.get("seed", 42)),
        "gpu": config.get("gpu", 0),
        "data": {
            "dataset": "STHELAR",
            "dataset_path": str(Path(config["dataset_path"]).resolve()),
            "split": "slide_independent",
            "train_folds": ["train"],
            "val_folds": ["valid"],
            "test_folds": ["test"],
            "num_nuclei_classes": int(checkpoint_meta["num_nuclei_classes"]),
            "num_tissue_classes": int(checkpoint_meta["num_tissue_classes"]),
            "input_shape": 256,
            "magnification": 40,
        },
        "model": {
            "backbone": checkpoint_meta["backbone"],
            "shared_decoders": False,
            "regression_loss": False,
        },
        "training": {
            "batch_size": int(config.get("batch_size", 16)),
            "mixed_precision": bool(config.get("mixed_precision", True)),
        },
        "transformations": {
            "normalize": {
                "mean": checkpoint_meta["normalize_mean"],
                "std": checkpoint_meta["normalize_std"],
            }
        },
        "inference": {
            "batch_size": int(config.get("batch_size", 16)),
            "num_workers": int(config.get("num_workers", 12)),
        },
        "efficiency": {
            "enabled": True,
            "fold": str(config["fold"]),
            "method": "Frozen",
            "backbone": backbone_identity,
        },
        "frozen_evaluation": {
            "untouched_pretrained_checkpoint": True,
            "optimizer_created": False,
            "semantic_head_reinitialized": False,
            "original_taxonomy": checkpoint_meta["dataset"],
            "target_taxonomy": "STHELAR",
            "taxonomy_mapping": None,
            "metric_scope": "class_agnostic_only",
            "checkpoint_sha256": config["expected_checkpoint_sha256"],
        },
    }


def _prepare_run_dir(
    output_dir: Path, checkpoint_path: Path, run_config: Dict[str, Any]
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_dir = output_dir / "checkpoints"
    checkpoint_dir.mkdir(exist_ok=True)
    link = checkpoint_dir / "frozen_pretrained.pth"
    if link.exists() or link.is_symlink():
        if link.resolve() != checkpoint_path.resolve():
            raise FileExistsError("Frozen checkpoint link points elsewhere: {}".format(link))
    else:
        os.symlink(str(checkpoint_path.resolve()), str(link))
    config_path = output_dir / "config.yaml"
    if config_path.exists():
        existing = yaml.safe_load(config_path.read_text())
        if existing != run_config:
            raise FileExistsError("Existing Frozen config differs: {}".format(config_path))
    else:
        _atomic_yaml(config_path, run_config)


def evaluate(config_path: Path, smoke_test: bool = False, gpu_override: str = None) -> None:
    config = yaml.safe_load(config_path.read_text())
    checkpoint_path = Path(config["checkpoint_path"]).resolve()
    dataset_path = Path(config["dataset_path"]).resolve()
    if not checkpoint_path.is_file():
        raise FileNotFoundError(checkpoint_path)
    if not dataset_path.is_dir():
        raise FileNotFoundError(dataset_path)
    observed_sha256 = _sha256(checkpoint_path)
    if observed_sha256 != config["expected_checkpoint_sha256"]:
        raise ValueError(
            "Frozen checkpoint SHA256 mismatch: expected={} observed={}".format(
                config["expected_checkpoint_sha256"], observed_sha256
            )
        )
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    checkpoint_meta = _checkpoint_metadata(
        checkpoint,
        expected_arch=config.get("expected_arch", "CellViTSAM"),
        expected_backbone=config.get("expected_backbone", "SAM-H"),
    )

    if smoke_test:
        context = tempfile.TemporaryDirectory(prefix="sthelar-frozen-smoke-")
        output_dir = Path(context.name)
        config = dict(config)
        config["batch_size"] = 2
        config["num_workers"] = 0
    else:
        context = None
        output_dir = Path(config["output_dir"]).resolve()
        if (output_dir / "frozen_class_agnostic_results.json").exists():
            raise FileExistsError("Frozen evaluation already complete: {}".format(output_dir))

    try:
        run_config = _run_config(config, checkpoint_meta, output_dir)
        _prepare_run_dir(output_dir, checkpoint_path, run_config)
        gpu = gpu_override if gpu_override is not None else str(config.get("gpu", 0))
        inference = InferenceCellViT(
            run_dir=output_dir,
            checkpoint_name="frozen_pretrained.pth",
            gpu=gpu,
            magnification=40,
            class_agnostic_only=True,
        )
        model, dataloader, dataset_config = inference.setup_patch_inference()
        model.requires_grad_(False)
        if any(parameter.requires_grad for parameter in model.parameters()):
            raise AssertionError("Frozen model has trainable parameters")

        if smoke_test:
            smoke_loader = DataLoader(
                # The historical evaluator squeezes singleton batch dimensions;
                # use two patches so this smoke follows the production code path.
                Subset(dataloader.dataset, [0, 1]),
                batch_size=2,
                num_workers=0,
                pin_memory=False,
                shuffle=False,
            )
            inference.run_patch_inference(
                model, smoke_loader, dataset_config, generate_plots=False
            )
            results_path = output_dir / "frozen_class_agnostic_results.json"
            results = json.loads(results_path.read_text())
            invalid_dataset_keys = {
                "Tissue-Multiclass-Accuracy",
                "mPQ",
                "mDQ",
                "mSQ",
                "f1_type",
            }
            if invalid_dataset_keys & set(results["dataset"]):
                raise AssertionError("Frozen smoke emitted class-aware metrics")
            if "nuclei_metrics_pq" in results or "nuclei_metrics_d" in results:
                raise AssertionError("Frozen smoke emitted per-class nuclei metrics")
            print(
                json.dumps(
                    {
                        "frozen_smoke_test": "passed",
                        "checkpoint_sha256": observed_sha256,
                        "optimizer_created": False,
                        "semantic_head_reinitialized": False,
                        "original_taxonomy_retained": True,
                        "trainable_parameters": 0,
                        "processed_patches": 2,
                        "reported_dataset_metrics": sorted(results["dataset"]),
                        "invalid_class_aware_metrics_reported": False,
                    },
                    indent=2,
                    sort_keys=True,
                )
            )
            return

        inference.run_patch_inference(
            model, dataloader, dataset_config, generate_plots=False
        )
    finally:
        if context is not None:
            context.cleanup()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--smoke-test", action="store_true")
    parser.add_argument("--gpu", default=None)
    args = parser.parse_args()
    evaluate(args.config, smoke_test=args.smoke_test, gpu_override=args.gpu)


if __name__ == "__main__":
    main()
