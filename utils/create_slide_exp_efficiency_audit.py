#!/usr/bin/env python3
"""Build the matched KLT slide-independent efficiency audit tables.

This script reads completed run artifacts only.  It never starts training and
never writes into a run directory or canonical checkpoint.
"""

import csv
import json
from pathlib import Path

from utils.collect_slide_exp_efficiency import collect


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "reports" / "efficiency_audit"
FOLDS = ("A", "B")
BACKBONES = ("SAM-H", "CellViT-256")
METHODS = ("Frozen", "LP", "Selected PEFT", "FullFT")
BACKBONE_IN_RECORD = {
    "SAM-H": "CellViT-SAM-H x40",
    "CellViT-256": "CellViT-256 x40",
}


def mean(values):
    values = [float(value) for value in values if value not in (None, "")]
    return sum(values) / len(values) if values else None


def write_csv(path, rows, fieldnames=None):
    path.parent.mkdir(parents=True, exist_ok=True)
    if fieldnames is None:
        fieldnames = list(rows[0]) if rows else []
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def inference_result_path(row):
    metrics_path = ROOT / row["inference_metrics_path"]
    if row["method"] == "Frozen":
        return metrics_path.with_name("frozen_class_agnostic_results.json")
    return metrics_path.with_name("inference_results.json")


def checkpoint_path(row):
    if row["method"] == "Frozen":
        return None
    metrics_path = ROOT / row["training_metrics_path"]
    return metrics_path.parent / "checkpoints" / "checkpoint_10.pth"


def matched_rows():
    all_rows = collect(ROOT / "run")
    selected = {}
    for row in all_rows:
        if str(row.get("tissue")) not in {"None", ""}:
            continue
        if str(row.get("seed")) != "42" or row.get("fold") not in FOLDS:
            continue
        if row.get("method") not in METHODS:
            continue
        backbone = next(
            (
                name
                for name, record_name in BACKBONE_IN_RECORD.items()
                if row.get("backbone") == record_name
            ),
            None,
        )
        if backbone is None:
            continue
        key = (backbone, row["method"], row["fold"])
        selected[key] = row

    expected = {
        (backbone, method, fold)
        for backbone in BACKBONES
        for method in METHODS
        for fold in FOLDS
    }
    if set(selected) != expected:
        raise RuntimeError(
            f"Matched run inventory differs from expectation: "
            f"missing={sorted(expected - set(selected))}, "
            f"unexpected={sorted(set(selected) - expected)}"
        )
    return selected


def load_fold_rows():
    selected = matched_rows()
    fold_rows = []
    for backbone in BACKBONES:
        for method in METHODS:
            for fold in FOLDS:
                raw = selected[(backbone, method, fold)]
                result = json.loads(inference_result_path(raw).read_text())["dataset"]
                checkpoint = checkpoint_path(raw)
                fold_rows.append(
                    {
                        "backbone": backbone,
                        "method": method,
                        "fold": fold,
                        "seed": 42,
                        "total_parameters": int(raw["total_parameters"]),
                        "trainable_parameters": int(raw["trainable_parameters"]),
                        "trainable_percent": float(raw["trainable_percentage"]),
                        "training_wall_seconds": (
                            float(raw["training_time_seconds"])
                            if raw["training_time_seconds"] not in (None, "")
                            else None
                        ),
                        "mean_seconds_per_epoch": (
                            float(raw["mean_seconds_per_epoch"])
                            if raw["mean_seconds_per_epoch"] not in (None, "")
                            else None
                        ),
                        "training_peak_allocated_gib": (
                            float(raw["training_peak_cuda_memory_allocated_gib"])
                            if raw["training_peak_cuda_memory_allocated_gib"]
                            not in (None, "")
                            else None
                        ),
                        "training_peak_reserved_gib": (
                            float(raw["training_peak_cuda_memory_reserved_gib"])
                            if raw["training_peak_cuda_memory_reserved_gib"]
                            not in (None, "")
                            else None
                        ),
                        "current_training_checkpoint_bytes": (
                            checkpoint.stat().st_size if checkpoint else None
                        ),
                        "inference_peak_allocated_gib": float(
                            raw["inference_peak_cuda_memory_allocated_gib"]
                        ),
                        "inference_peak_reserved_gib": float(
                            raw["inference_peak_cuda_memory_reserved_gib"]
                        ),
                        "inference_patches_per_second": float(raw["patches_per_second"]),
                        "test_patches": int(raw["test_patch_count"]),
                        "dice": float(result["Binary-Cell-Dice-Mean"]),
                        "jaccard": float(result["Binary-Cell-Jacard-Mean"]),
                        "bPQ": float(result["bPQ"]),
                        "mPQ": float(result["mPQ"]) if "mPQ" in result else None,
                        "f1_detection": float(result["f1_detection"]),
                        "gpu_model": raw["gpu_model"],
                        "partition": raw.get("partition"),
                        "training_batch_size": (
                            int(raw["batch_size_training"])
                            if raw["batch_size_training"] not in (None, "")
                            else None
                        ),
                        "inference_batch_size": int(raw["batch_size_inference"]),
                        "amp_policy": (
                            "CUDA autocast FP16 inference; training N/A"
                            if raw["method"] == "Frozen"
                            else "CUDA autocast FP16 train/inference; GradScaler train"
                        ),
                        "pytorch_version": raw["pytorch_version"],
                        "cuda_version": raw["cuda_version"],
                        "training_metrics_path": raw["training_metrics_path"],
                        "inference_metrics_path": raw["inference_metrics_path"],
                    }
                )
    return fold_rows


def summarize(fold_rows):
    weight_sizes = json.loads(
        (OUT / "inference_weight_sizes.json").read_text()
    )
    adapter_sizes = {}
    for backbone, slug in (("SAM-H", "sam_h"), ("CellViT-256", "cellvit256")):
        sizes = []
        for fold in FOLDS:
            metadata = json.loads(
                (
                    OUT
                    / "adapters"
                    / slug
                    / f"fold{fold}"
                    / "adapter_weights.verification.json"
                ).read_text()
            )
            if metadata["state_reconstruction"] != "exact_all_tensors":
                raise RuntimeError(f"Unverified adapter: {backbone} fold {fold}")
            if metadata["forward_verification"] != "exact_all_output_tensors":
                raise RuntimeError(f"Forward-unverified adapter: {backbone} fold {fold}")
            sizes.append(int(metadata["artifact_size_bytes"]))
        adapter_sizes[backbone] = int(mean(sizes))

    summary = []
    for backbone in BACKBONES:
        for method in METHODS:
            group = [
                row
                for row in fold_rows
                if row["backbone"] == backbone and row["method"] == method
            ]
            first = group[0]
            size_key = f"{backbone}|{method}"
            summary.append(
                {
                    "backbone": backbone,
                    "method": method,
                    "seed": 42,
                    "matched_folds": 2,
                    "total_parameters": first["total_parameters"],
                    "trainable_parameters": first["trainable_parameters"],
                    "trainable_percent": first["trainable_percent"],
                    "training_wall_seconds_mean": mean(
                        row["training_wall_seconds"] for row in group
                    ),
                    "training_wall_hours_mean": (
                        mean(row["training_wall_seconds"] for row in group) / 3600
                        if method != "Frozen"
                        else None
                    ),
                    "mean_seconds_per_epoch": mean(
                        row["mean_seconds_per_epoch"] for row in group
                    ),
                    "mean_minutes_per_epoch": (
                        mean(row["mean_seconds_per_epoch"] for row in group) / 60
                        if method != "Frozen"
                        else None
                    ),
                    "training_peak_allocated_gib_mean": mean(
                        row["training_peak_allocated_gib"] for row in group
                    ),
                    "training_peak_reserved_gib_mean": mean(
                        row["training_peak_reserved_gib"] for row in group
                    ),
                    "current_training_checkpoint_bytes_mean": mean(
                        row["current_training_checkpoint_bytes"] for row in group
                    ),
                    "current_training_checkpoint_gib_mean": (
                        mean(row["current_training_checkpoint_bytes"] for row in group)
                        / (1024**3)
                        if method != "Frozen"
                        else None
                    ),
                    "inference_model_weight_bytes": weight_sizes[size_key][
                        "inference_weight_size_bytes"
                    ],
                    "inference_model_weight_gib": weight_sizes[size_key][
                        "inference_weight_size_bytes"
                    ]
                    / (1024**3),
                    "adapter_only_weight_bytes": (
                        adapter_sizes[backbone] if method == "Selected PEFT" else None
                    ),
                    "adapter_only_weight_mib": (
                        adapter_sizes[backbone] / (1024**2)
                        if method == "Selected PEFT"
                        else None
                    ),
                    "inference_peak_allocated_gib_mean": mean(
                        row["inference_peak_allocated_gib"] for row in group
                    ),
                    "inference_peak_reserved_gib_mean": mean(
                        row["inference_peak_reserved_gib"] for row in group
                    ),
                    "inference_patches_per_second_mean": mean(
                        row["inference_patches_per_second"] for row in group
                    ),
                    "dice_mean": mean(row["dice"] for row in group),
                    "jaccard_mean": mean(row["jaccard"] for row in group),
                    "bPQ_mean": mean(row["bPQ"] for row in group),
                    "mPQ_mean": mean(row["mPQ"] for row in group),
                    "f1_detection_mean": mean(
                        row["f1_detection"] for row in group
                    ),
                    "gpu_model": first["gpu_model"],
                    "partition": first["partition"],
                    "training_batch_size": first["training_batch_size"],
                    "inference_batch_size": first["inference_batch_size"],
                    "amp_policy": first["amp_policy"],
                    "pytorch_version": first["pytorch_version"],
                    "cuda_version": first["cuda_version"],
                }
            )
    return summary


def deployment_rows(summary):
    by_key = {(row["backbone"], row["method"]): row for row in summary}
    base_sizes = {
        backbone: by_key[(backbone, "Frozen")]["inference_model_weight_bytes"]
        for backbone in BACKBONES
    }
    strategies = (
        ("A", "SAM-H", "FullFT", "N complete SAM-H inference models"),
        ("B", "SAM-H", "Selected PEFT", "one SAM-H base + N adapter packages"),
        ("C", "CellViT-256", "FullFT", "N complete CellViT-256 inference models"),
        (
            "D",
            "CellViT-256",
            "Selected PEFT",
            "one CellViT-256 base + N adapter packages",
        ),
    )
    rows = []
    for strategy, backbone, method, description in strategies:
        metric = by_key[(backbone, method)]
        for n_domains in (1, 3, 6, 9):
            if method == "FullFT":
                total = n_domains * metric["inference_model_weight_bytes"]
                base = 0
                per_domain = metric["inference_model_weight_bytes"]
            else:
                base = base_sizes[backbone]
                per_domain = metric["adapter_only_weight_bytes"]
                total = base + n_domains * per_domain
            rows.append(
                {
                    "strategy": strategy,
                    "description": description,
                    "backbone": backbone,
                    "method": method,
                    "N_domains": n_domains,
                    "shared_base_bytes": base,
                    "per_domain_artifact_bytes": per_domain,
                    "total_deployment_bytes": total,
                    "total_deployment_gib": total / (1024**3),
                    "mPQ_mean": metric["mPQ_mean"],
                    "bPQ_mean": metric["bPQ_mean"],
                    "f1_detection_mean": metric["f1_detection_mean"],
                    "inference_peak_allocated_gib_mean": metric[
                        "inference_peak_allocated_gib_mean"
                    ],
                    "inference_patches_per_second_mean": metric[
                        "inference_patches_per_second_mean"
                    ],
                }
            )
    return rows


def delta_rows(summary, deployment):
    by_key = {(row["backbone"], row["method"]): row for row in summary}
    comparisons = (
        (
            "SAM-H PEFT vs SAM-H FullFT",
            ("SAM-H", "Selected PEFT"),
            ("SAM-H", "FullFT"),
        ),
        (
            "CellViT-256 PEFT vs CellViT-256 FullFT",
            ("CellViT-256", "Selected PEFT"),
            ("CellViT-256", "FullFT"),
        ),
        (
            "SAM-H PEFT vs CellViT-256 PEFT",
            ("SAM-H", "Selected PEFT"),
            ("CellViT-256", "Selected PEFT"),
        ),
        (
            "SAM-H PEFT vs CellViT-256 FullFT",
            ("SAM-H", "Selected PEFT"),
            ("CellViT-256", "FullFT"),
        ),
    )
    metrics = (
        ("mPQ", "mPQ_mean", "higher_better"),
        ("bPQ", "bPQ_mean", "higher_better"),
        ("F1 detection", "f1_detection_mean", "higher_better"),
        ("Dice", "dice_mean", "higher_better"),
        ("trainable parameters", "trainable_parameters", "lower_better"),
        ("training wall hours", "training_wall_hours_mean", "lower_better"),
        ("mean minutes per epoch", "mean_minutes_per_epoch", "lower_better"),
        (
            "training peak allocated GiB",
            "training_peak_allocated_gib_mean",
            "lower_better",
        ),
        (
            "training peak reserved GiB",
            "training_peak_reserved_gib_mean",
            "lower_better",
        ),
        (
            "current training checkpoint bytes",
            "current_training_checkpoint_bytes_mean",
            "lower_better",
        ),
        (
            "inference model weight bytes",
            "inference_model_weight_bytes",
            "lower_better",
        ),
        (
            "inference peak allocated GiB",
            "inference_peak_allocated_gib_mean",
            "lower_better",
        ),
        (
            "inference patches per second",
            "inference_patches_per_second_mean",
            "higher_better",
        ),
    )
    rows = []
    for label, lhs_key, rhs_key in comparisons:
        lhs = by_key[lhs_key]
        rhs = by_key[rhs_key]
        for metric, field, direction in metrics:
            left = lhs[field]
            right = rhs[field]
            rows.append(
                {
                    "comparison": label,
                    "metric": metric,
                    "directionality": direction,
                    "lhs_value": left,
                    "rhs_value": right,
                    "absolute_delta_lhs_minus_rhs": left - right,
                    "relative_delta_percent": 100 * (left / right - 1),
                }
            )

    deployment_by = {
        (row["backbone"], row["method"], row["N_domains"]): row
        for row in deployment
    }
    for label, lhs_key, rhs_key in comparisons:
        for n_domains in (1, 3, 6, 9):
            left = deployment_by[(*lhs_key, n_domains)]["total_deployment_bytes"]
            right = deployment_by[(*rhs_key, n_domains)]["total_deployment_bytes"]
            rows.append(
                {
                    "comparison": label,
                    "metric": f"deployment bytes N={n_domains}",
                    "directionality": "lower_better",
                    "lhs_value": left,
                    "rhs_value": right,
                    "absolute_delta_lhs_minus_rhs": left - right,
                    "relative_delta_percent": 100 * (left / right - 1),
                }
            )
    return rows


def main():
    fold_rows = load_fold_rows()
    summary = summarize(fold_rows)
    deployment = deployment_rows(summary)
    deltas = delta_rows(summary, deployment)

    write_csv(OUT / "matched_fold_measurements.csv", fold_rows)
    write_csv(OUT / "paper_ready_efficiency.csv", summary)
    write_csv(
        OUT / "performance_vs_vram.csv",
        [
            {
                key: row[key]
                for key in (
                    "backbone",
                    "method",
                    "mPQ_mean",
                    "bPQ_mean",
                    "f1_detection_mean",
                    "training_peak_allocated_gib_mean",
                    "training_peak_reserved_gib_mean",
                    "inference_peak_allocated_gib_mean",
                    "inference_peak_reserved_gib_mean",
                    "inference_patches_per_second_mean",
                )
            }
            for row in summary
        ],
    )
    write_csv(OUT / "performance_vs_deployment_storage.csv", deployment)
    write_csv(OUT / "performance_cost_deltas.csv", deltas)
    print(f"Wrote audit tables to {OUT}")


if __name__ == "__main__":
    main()
