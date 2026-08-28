#!/usr/bin/env python3
"""Aggregate matched slide-independent efficiency JSON files without mutation."""

import argparse
import csv
import json
import os
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, Tuple


FIELDS = [
    "fold",
    "tissue",
    "backbone",
    "method",
    "seed",
    "training_completed",
    "inference_completed",
    "gpu_model",
    "partition",
    "pytorch_version",
    "cuda_version",
    "amp_mixed_precision",
    "batch_size_training",
    "batch_size_inference",
    "train_patch_count",
    "validation_patch_count",
    "test_patch_count",
    "trainable_parameters",
    "total_parameters",
    "trainable_percentage",
    "training_time_seconds",
    "training_time_hours",
    "epoch_time_seconds",
    "mean_seconds_per_epoch",
    "training_peak_cuda_memory_allocated_bytes",
    "training_peak_cuda_memory_reserved_bytes",
    "training_peak_cuda_memory_allocated_gib",
    "training_peak_cuda_memory_reserved_gib",
    "checkpoint_size_bytes",
    "checkpoint_size_gib",
    "inference_time_seconds",
    "inference_patch_count",
    "patches_per_second",
    "inference_peak_cuda_memory_allocated_bytes",
    "inference_peak_cuda_memory_reserved_bytes",
    "inference_peak_cuda_memory_allocated_gib",
    "inference_peak_cuda_memory_reserved_gib",
    "training_metrics_path",
    "inference_metrics_path",
]


def _backbone(data: Dict[str, Any], path: Path) -> str:
    if data.get("backbone"):
        return str(data["backbone"])
    # Schema-v1 slide-independent records created before backbone identity was
    # added are from the original SAM-H campaign. Keep this compatibility
    # branch path-scoped so CellViT-256 records can never be silently merged.
    if "cellvit256" not in str(path).lower():
        return "CellViT-SAM-H x40"
    return "CellViT-256 x40"


def _key(data: Dict[str, Any], path: Path) -> Tuple[str, str, str, str, str]:
    return (
        str(data.get("fold")),
        str(data.get("tissue")),
        _backbone(data, path),
        str(data.get("method")),
        str(data.get("seed")),
    )


def collect(run_root: Path) -> list:
    groups: Dict[Tuple[str, str, str, str, str], Dict[str, list]] = defaultdict(
        lambda: {"training": [], "inference": []}
    )
    for path in sorted(run_root.glob("*slideind*/**/*efficiency_metrics.json")):
        data = json.loads(path.read_text())
        mode = data.get("mode")
        if mode not in {"training", "inference"}:
            continue
        groups[_key(data, path)][mode].append((path, data))

    rows = []
    for key, modes in sorted(groups.items()):
        for mode, records in modes.items():
            completed = [record for record in records if record[1].get("completed")]
            if len(completed) > 1:
                raise RuntimeError(
                    "Multiple completed {} records for {}: {}".format(
                        mode, key, [str(record[0]) for record in completed]
                    )
                )
            if completed:
                modes[mode] = completed
            elif records:
                modes[mode] = [max(records, key=lambda record: record[0].stat().st_mtime)]

        training_record = modes["training"][0] if modes["training"] else (None, {})
        inference_record = modes["inference"][0] if modes["inference"] else (None, {})
        training_path, training = training_record
        inference_path, inference = inference_record
        common = training or inference
        row = {
            "fold": key[0],
            "tissue": key[1],
            "backbone": key[2],
            "method": key[3],
            "seed": key[4],
            "training_completed": training.get("completed") if training else None,
            "inference_completed": inference.get("completed") if inference else None,
            "gpu_model": common.get("gpu_model"),
            "partition": common.get("partition"),
            "pytorch_version": common.get("pytorch_version"),
            "cuda_version": common.get("cuda_version"),
            "amp_mixed_precision": common.get("amp_mixed_precision"),
            "batch_size_training": training.get("batch_size"),
            "batch_size_inference": inference.get("batch_size"),
            "train_patch_count": common.get("train_patch_count"),
            "validation_patch_count": common.get("validation_patch_count"),
            "test_patch_count": common.get("test_patch_count"),
            "trainable_parameters": common.get("trainable_parameters"),
            "total_parameters": common.get("total_parameters"),
            "trainable_percentage": common.get("trainable_percentage"),
            "training_time_seconds": training.get("training_time_seconds"),
            "training_time_hours": training.get("training_time_hours"),
            "epoch_time_seconds": json.dumps(training.get("epoch_time_seconds", [])),
            "mean_seconds_per_epoch": training.get("mean_seconds_per_epoch"),
            "training_peak_cuda_memory_allocated_bytes": training.get(
                "peak_cuda_memory_allocated_bytes"
            ),
            "training_peak_cuda_memory_reserved_bytes": training.get(
                "peak_cuda_memory_reserved_bytes"
            ),
            "training_peak_cuda_memory_allocated_gib": training.get(
                "peak_cuda_memory_allocated_gib"
            ),
            "training_peak_cuda_memory_reserved_gib": training.get(
                "peak_cuda_memory_reserved_gib"
            ),
            "checkpoint_size_bytes": training.get("checkpoint_size_bytes"),
            "checkpoint_size_gib": training.get("checkpoint_size_gib"),
            "inference_time_seconds": inference.get("inference_time_seconds"),
            "inference_patch_count": inference.get("inference_patch_count"),
            "patches_per_second": inference.get("patches_per_second"),
            "inference_peak_cuda_memory_allocated_bytes": inference.get(
                "peak_cuda_memory_allocated_bytes"
            ),
            "inference_peak_cuda_memory_reserved_bytes": inference.get(
                "peak_cuda_memory_reserved_bytes"
            ),
            "inference_peak_cuda_memory_allocated_gib": inference.get(
                "peak_cuda_memory_allocated_gib"
            ),
            "inference_peak_cuda_memory_reserved_gib": inference.get(
                "peak_cuda_memory_reserved_gib"
            ),
            "training_metrics_path": str(training_path) if training_path else None,
            "inference_metrics_path": str(inference_path) if inference_path else None,
        }
        rows.append(row)
    return rows


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-root", type=Path, default=Path("run"))
    parser.add_argument(
        "--output", type=Path, default=Path("reports/slide_exp_efficiency.csv")
    )
    args = parser.parse_args()
    rows = collect(args.run_root)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_name(".{}.tmp-{}".format(args.output.name, os.getpid()))
    try:
        with temporary.open("w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=FIELDS)
            writer.writeheader()
            writer.writerows(rows)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(str(temporary), str(args.output))
    finally:
        if temporary.exists():
            temporary.unlink()
    print("Wrote {} rows to {}".format(len(rows), args.output))


if __name__ == "__main__":
    main()
