#!/usr/bin/env python3
"""Matched-nuclei type-assignment analysis for STHELAR CellViT inference runs.

The default path uses the per-image ``detection_stats`` saved in
``inference_results.json``. Those stats are sufficient for type assignment on
CellViT detection matches because they store the paired true/predicted type
confusion and unmatched true/predicted counts per patch.
"""

import argparse
import csv
import json
import math
import os
import re
from pathlib import Path
from typing import Any

import numpy as np


os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")

CANONICAL_CLASSES = ("Epithelial", "Immune", "Stromal", "Melanocyte", "Other")
DEFAULT_SUMMARY = Path("reports/runs_summary.csv")
DEFAULT_OUT_DIR = Path("reports/type_assignment")


class RunSpec:
    def __init__(self, label, cohort, run_dir, source, required=True):
        self.label = label
        self.cohort = cohort
        self.run_dir = run_dir
        self.source = source
        self.required = required


class RunRequest:
    def __init__(self, label, cohort, includes, excludes=(), required=True):
        self.label = label
        self.cohort = cohort
        self.includes = includes
        self.excludes = excludes
        self.required = required


def requested_runs():
    klt = "sthelar40x_kidney_liver_tonsil_5class_spatial_margin128"
    requests = [
        RunRequest("KLT Frozen CellViT seed42", "KLT", (klt, "freeze_e10_seed42")),
        RunRequest("KLT FullFT seed42", "KLT", (klt, "fullft_lr1e-5_e10_seed42")),
        RunRequest("KLT FullFT seed43", "KLT", (klt, "fullft_lr1e-5_e10_seed43")),
        RunRequest(
            "KLT LoRA+AF heads_only seed42",
            "KLT",
            (klt, "lora_adaptformer", "decoder_heads_only", "seed42"),
        ),
        RunRequest(
            "KLT LoRA+AF heads_only seed43",
            "KLT",
            (klt, "lora_adaptformer", "decoder_heads_only", "seed43"),
        ),
        RunRequest(
            "KLT LoRA+AF last_stage seed42",
            "KLT",
            (klt, "lora_adaptformer", "decoder_last_stage", "seed42"),
        ),
        RunRequest(
            "KLT LoRA+AF last_stage seed43",
            "KLT",
            (klt, "lora_adaptformer", "decoder_last_stage", "seed43"),
        ),
        RunRequest(
            "KLT LoRA+AF conv_adapters seed42",
            "KLT",
            (klt, "lora_adaptformer", "decoder_conv_adapters", "seed42"),
        ),
        RunRequest(
            "KLT LoRA+AF conv_adapters seed43",
            "KLT",
            (klt, "lora_adaptformer", "decoder_conv_adapters", "seed43"),
        ),
        RunRequest(
            "KLT VeRA+AF conv_adapters seed42",
            "KLT",
            (klt, "vera_adaptformer", "decoder_conv_adapters", "seed42"),
            required=False,
        ),
        RunRequest(
            "KLT VeRA+AF conv_adapters seed43",
            "KLT",
            (klt, "vera_adaptformer", "decoder_conv_adapters", "seed43"),
            required=False,
        ),
        RunRequest(
            "KLT VeRA+AF heads_only seed42",
            "KLT",
            (klt, "vera_adaptformer", "decoder_heads_only", "seed42"),
            required=False,
        ),
        RunRequest(
            "KLT VeRA+AF heads_only seed43",
            "KLT",
            (klt, "vera_adaptformer", "decoder_heads_only", "seed43"),
            required=False,
        ),
    ]
    for tissue in ("liver", "kidney", "ovary"):
        stem = f"sthelar40x_{tissue}_5class_spatial_margin128"
        requests.extend(
            [
                RunRequest(
                    f"{tissue.title()} FullFT seed42",
                    tissue.title(),
                    (stem, "fullft_lr1e-5_e10_seed42"),
                ),
                RunRequest(
                    f"{tissue.title()} LoRA+AF heads_only seed42",
                    tissue.title(),
                    (stem, "lora_adaptformer", "decoder_heads_only", "seed42"),
                ),
            ]
        )
    return requests


def safe_slug(text: str) -> str:
    slug = re.sub(r"[^A-Za-z0-9]+", "_", text).strip("_").lower()
    return slug or "run"


def resolve_inference_json(path):
    if path.is_file() and path.name.endswith(".json"):
        return path
    candidate = path / "inference_results.json"
    if candidate.is_file():
        return candidate
    return None


def read_csv(path):
    with path.open(newline="") as handle:
        return list(csv.DictReader(handle))


def has_inference(row):
    if row.get("has_inference", "").lower() != "true":
        return False
    run_dir = row.get("run_dir", "")
    return bool(run_dir and resolve_inference_json(Path(run_dir)))


def select_default_runs(summary_path):
    rows = read_csv(summary_path)
    selected = []
    notes = []
    for request in requested_runs():
        matches = []
        for row in rows:
            run_name = row.get("run_name", "")
            if not all(token in run_name for token in request.includes):
                continue
            if any(token in run_name for token in request.excludes):
                continue
            if not has_inference(row):
                continue
            matches.append(row)
        if not matches:
            level = "missing required" if request.required else "optional unavailable"
            notes.append(f"{level}: {request.label}")
            continue
        matches.sort(key=lambda item: item.get("run_dir", ""))
        row = matches[-1]
        selected.append(
            RunSpec(
                label=request.label,
                cohort=request.cohort,
                run_dir=Path(row["run_dir"]),
                source="runs_summary",
                required=request.required,
            )
        )
        if len(matches) > 1:
            notes.append(
                f"multiple completed matches for {request.label}; selected {row['run_dir']}"
            )
    return selected, notes


def load_json(path):
    with path.open() as handle:
        return json.load(handle)


def load_yaml(path):
    try:
        import yaml
    except ImportError as exc:
        raise RuntimeError("PyYAML is required to read CellViT config files") from exc
    with path.open() as handle:
        return yaml.safe_load(handle)


def load_nuclei_types(run_dir, inference):
    config_path = run_dir / "config.yaml"
    if config_path.is_file():
        config = load_yaml(config_path)
        dataset_path = Path(config.get("data", {}).get("dataset_path", ""))
        dataset_config_path = dataset_path / "dataset_config.yaml"
        if dataset_config_path.is_file():
            dataset_config = load_yaml(dataset_config_path)
            nuclei_types = dataset_config.get("nuclei_types")
            if isinstance(nuclei_types, dict):
                return {str(k): int(v) for k, v in nuclei_types.items()}

    # Fallback for already-known STHELAR 5-class outputs.
    matrix_size = None
    for record in (inference.get("image_metrics") or {}).values():
        stats = record.get("detection_stats") if isinstance(record, dict) else None
        if stats and "paired_confusion" in stats:
            matrix_size = len(stats["paired_confusion"])
            break
    if matrix_size == 6:
        return {
            "Background": 0,
            "Immune": 1,
            "Stromal": 2,
            "Epithelial": 3,
            "Melanocyte": 4,
            "Other": 5,
        }
    raise ValueError(f"Could not determine nuclei type mapping for {run_dir}")


def canonical_ids(nuclei_types):
    lower_to_id = {name.lower(): int(class_id) for name, class_id in nuclei_types.items()}
    ids = {}
    for class_name in CANONICAL_CLASSES:
        key = class_name.lower()
        if key not in lower_to_id:
            raise ValueError(f"Missing class {class_name!r} in nuclei_types={nuclei_types}")
        ids[class_name] = lower_to_id[key]
    return ids


def pad_vector(values: Any, size: int) -> np.ndarray:
    arr = np.asarray(values, dtype=np.int64).reshape(-1)
    if arr.size >= size:
        return arr[:size]
    out = np.zeros(size, dtype=np.int64)
    out[: arr.size] = arr
    return out


def pad_matrix(values: Any, size: int) -> np.ndarray:
    arr = np.asarray(values, dtype=np.int64)
    out = np.zeros((size, size), dtype=np.int64)
    rows = min(size, arr.shape[0])
    cols = min(size, arr.shape[1])
    out[:rows, :cols] = arr[:rows, :cols]
    return out


def aggregate_detection_stats(
    records, n_classes
):
    confusion = np.zeros((n_classes, n_classes), dtype=np.int64)
    unpaired_true = np.zeros(n_classes, dtype=np.int64)
    unpaired_pred = np.zeros(n_classes, dtype=np.int64)
    found = 0
    for record in records:
        stats = record.get("detection_stats") if isinstance(record, dict) else None
        if not stats:
            continue
        if "paired_confusion" in stats:
            confusion += pad_matrix(stats["paired_confusion"], n_classes)
            unpaired_true += pad_vector(stats.get("unpaired_true_counts", []), n_classes)
            unpaired_pred += pad_vector(stats.get("unpaired_pred_counts", []), n_classes)
            found += 1
        elif {"paired_true", "paired_pred"} <= set(stats):
            paired_true = np.asarray(stats.get("paired_true", []), dtype=int)
            paired_pred = np.asarray(stats.get("paired_pred", []), dtype=int)
            for true_id, pred_id in zip(paired_true, paired_pred):
                if 0 <= true_id < n_classes and 0 <= pred_id < n_classes:
                    confusion[true_id, pred_id] += 1
            unpaired_true += np.bincount(
                np.asarray(stats.get("unpaired_true", []), dtype=int),
                minlength=n_classes,
            )[:n_classes]
            unpaired_pred += np.bincount(
                np.asarray(stats.get("unpaired_pred", []), dtype=int),
                minlength=n_classes,
            )[:n_classes]
            found += 1
    return confusion, unpaired_true, unpaired_pred, found


def finite_mean(values):
    parsed = []
    for value in values:
        try:
            parsed.append(float(value))
        except (TypeError, ValueError):
            continue
    arr = np.asarray(parsed, dtype=float)
    if arr.size == 0 or np.all(np.isnan(arr)):
        return None
    return float(np.nanmean(arr))


def safe_ratio(num, den):
    return float(num / den) if den else None


def type_metrics(
    confusion_all: np.ndarray,
    unpaired_true_all: np.ndarray,
    unpaired_pred_all: np.ndarray,
    class_ids,
):
    ids = [class_ids[name] for name in CANONICAL_CLASSES]
    confusion = confusion_all[np.ix_(ids, ids)]
    unpaired_true = unpaired_true_all[ids]
    unpaired_pred = unpaired_pred_all[ids]

    rows = []
    support_total = int(confusion.sum(axis=1).sum())
    correct_total = int(np.trace(confusion))
    f1_values = []
    weighted_terms = []
    for idx, class_name in enumerate(CANONICAL_CLASSES):
        tp = int(confusion[idx, idx])
        pred_support = int(confusion[:, idx].sum())
        true_support = int(confusion[idx, :].sum())
        precision = safe_ratio(tp, pred_support) or 0.0
        recall = safe_ratio(tp, true_support) or 0.0
        f1 = safe_ratio(2 * precision * recall, precision + recall) or 0.0
        f1_values.append(f1)
        weighted_terms.append(f1 * true_support)
        rows.append(
            {
                "class": class_name,
                "precision": precision,
                "recall": recall,
                "f1": f1,
                "matched_true_support": true_support,
                "matched_pred_support": pred_support,
                "unmatched_true": int(unpaired_true[idx]),
                "unmatched_pred": int(unpaired_pred[idx]),
            }
        )

    matched = int(confusion.sum())
    unmatched_true = int(unpaired_true.sum())
    unmatched_pred = int(unpaired_pred.sum())
    summary = {
        "matched_nuclei": matched,
        "unmatched_true_nuclei": unmatched_true,
        "unmatched_pred_nuclei": unmatched_pred,
        "type_accuracy": safe_ratio(correct_total, matched),
        "macro_f1": float(np.mean(f1_values)) if f1_values else None,
        "weighted_f1": (
            float(sum(weighted_terms) / support_total) if support_total else None
        ),
        "detection_f1_from_stats": safe_ratio(
            2 * matched, 2 * matched + unmatched_pred + unmatched_true
        ),
        "detection_precision_from_stats": safe_ratio(matched, matched + unmatched_pred),
        "detection_recall_from_stats": safe_ratio(matched, matched + unmatched_true),
    }
    return summary, rows, confusion, unpaired_true, unpaired_pred


def extract_slide_id(image_name: str) -> str:
    stem = Path(image_name).name
    if "__" in stem:
        return stem.split("__", 1)[0]
    match = re.match(r"(.+?)_\d+(?:\.[A-Za-z0-9]+)?$", stem)
    if match:
        return match.group(1)
    return Path(stem).stem


def slide_summaries(
    image_metrics,
    n_classes,
    class_ids,
):
    grouped = {}
    for name, record in image_metrics.items():
        grouped.setdefault(extract_slide_id(name), []).append((name, record))

    rows = []
    for slide_id, items in sorted(grouped.items()):
        records = [record for _, record in items]
        confusion_all, unpaired_true_all, unpaired_pred_all, detection_records = (
            aggregate_detection_stats(records, n_classes)
        )
        summary, _, _, _, _ = type_metrics(
            confusion_all, unpaired_true_all, unpaired_pred_all, class_ids
        )
        rows.append(
            {
                "slide_id": slide_id,
                "num_patches": len(items),
                "detection_records": detection_records,
                "bPQ": finite_mean([record.get("bPQ") for record in records]),
                "mPQ": finite_mean([record.get("mPQ") for record in records]),
                "detection_f1": summary["detection_f1_from_stats"],
                "type_macro_f1": summary["macro_f1"],
                "type_weighted_f1": summary["weighted_f1"],
                "type_accuracy": summary["type_accuracy"],
                "matched_nuclei": summary["matched_nuclei"],
                "unmatched_true_nuclei": summary["unmatched_true_nuclei"],
                "unmatched_pred_nuclei": summary["unmatched_pred_nuclei"],
            }
        )
    total_patches = sum(row["num_patches"] for row in rows)
    total_matched = sum(row["matched_nuclei"] for row in rows)
    for row in rows:
        row["patch_fraction"] = safe_ratio(row["num_patches"], total_patches)
        row["matched_fraction"] = safe_ratio(row["matched_nuclei"], total_matched)
    return rows


def write_dict_csv(path, rows, fieldnames):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key) for key in fieldnames})


def write_confusion_csv(path, matrix):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["true_label"] + list(CANONICAL_CLASSES))
        for class_name, values in zip(CANONICAL_CLASSES, matrix):
            writer.writerow([class_name] + [int(value) for value in values])


def plot_confusion_png(path, matrix, title):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    row_sums = matrix.sum(axis=1, keepdims=True)
    normalized = np.divide(
        matrix,
        row_sums,
        out=np.zeros_like(matrix, dtype=float),
        where=row_sums != 0,
    )
    fig, ax = plt.subplots(figsize=(7.2, 5.8))
    image = ax.imshow(normalized, cmap="Blues", vmin=0.0, vmax=1.0)
    ax.set_title(title)
    ax.set_xlabel("Predicted type")
    ax.set_ylabel("True type")
    ax.set_xticks(range(len(CANONICAL_CLASSES)), CANONICAL_CLASSES, rotation=35, ha="right")
    ax.set_yticks(range(len(CANONICAL_CLASSES)), CANONICAL_CLASSES)
    for row in range(matrix.shape[0]):
        for col in range(matrix.shape[1]):
            value = normalized[row, col]
            count = int(matrix[row, col])
            color = "white" if value > 0.55 else "black"
            ax.text(col, row, f"{value:.2f}\n{count}", ha="center", va="center", color=color, fontsize=8)
    fig.colorbar(image, ax=ax, fraction=0.046, pad=0.04, label="Row-normalized fraction")
    fig.tight_layout()
    fig.savefig(path, dpi=220)
    plt.close(fig)


def plot_combined_klt(path, matrices):
    if not matrices:
        return
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, len(matrices), figsize=(6.2 * len(matrices), 5.5))
    if len(matrices) == 1:
        axes = [axes]
    for ax, (title, matrix) in zip(axes, matrices.items()):
        row_sums = matrix.sum(axis=1, keepdims=True)
        normalized = np.divide(
            matrix,
            row_sums,
            out=np.zeros_like(matrix, dtype=float),
            where=row_sums != 0,
        )
        image = ax.imshow(normalized, cmap="Blues", vmin=0.0, vmax=1.0)
        ax.set_title(title)
        ax.set_xlabel("Predicted")
        ax.set_xticks(range(len(CANONICAL_CLASSES)), CANONICAL_CLASSES, rotation=35, ha="right")
        ax.set_yticks(range(len(CANONICAL_CLASSES)), CANONICAL_CLASSES)
        if ax is axes[0]:
            ax.set_ylabel("True")
        for row in range(matrix.shape[0]):
            for col in range(matrix.shape[1]):
                value = normalized[row, col]
                color = "white" if value > 0.55 else "black"
                ax.text(col, row, f"{value:.2f}", ha="center", va="center", color=color, fontsize=8)
    fig.colorbar(image, ax=axes, fraction=0.025, pad=0.02, label="Row-normalized fraction")
    fig.savefig(path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def inspect_existing_outputs(summary_path):
    rows = read_csv(summary_path)
    completed = [
        row
        for row in rows
        if row.get("has_inference", "").lower() == "true" or row.get("test_mPQ")
    ]
    missing_f1 = [
        row.get("run_name", "")
        for row in completed
        if not row.get("test_F1")
    ]
    return {
        "runs_summary_rows": len(rows),
        "completed_or_inferred_rows": len(completed),
        "missing_f1_count": len(missing_f1),
        "missing_f1_examples": missing_f1[:10],
    }


def analyze_run(spec, out_dir, matching):
    inference_json = resolve_inference_json(spec.run_dir)
    if inference_json is None:
        raise FileNotFoundError(f"No inference_results.json found for {spec.run_dir}")
    inference = load_json(inference_json)
    image_metrics = inference.get("image_metrics")
    if not isinstance(image_metrics, dict):
        raise ValueError(f"{inference_json} has no image_metrics dictionary")
    if matching == "iou":
        raise ValueError(
            "IoU matching cannot be computed from the saved inference_results.json "
            "because per-instance masks/contours are not exported."
        )

    nuclei_types = load_nuclei_types(spec.run_dir, inference)
    class_ids = canonical_ids(nuclei_types)
    n_classes = max(nuclei_types.values()) + 1
    records = list(image_metrics.values())
    confusion_all, unpaired_true_all, unpaired_pred_all, detection_records = (
        aggregate_detection_stats(records, n_classes)
    )
    if detection_records == 0:
        raise ValueError(f"{inference_json} has no per-image detection_stats")
    summary, per_class_rows, confusion, _, _ = type_metrics(
        confusion_all, unpaired_true_all, unpaired_pred_all, class_ids
    )
    dataset = inference.get("dataset", {})
    slug = safe_slug(spec.label)
    run_out = out_dir / slug
    run_out.mkdir(parents=True, exist_ok=True)

    confusion_csv = run_out / "confusion_matrix.csv"
    confusion_png = run_out / "confusion_matrix.png"
    per_class_csv = run_out / "per_class_type_metrics.csv"
    slide_csv = run_out / "slide_summary.csv"
    write_confusion_csv(confusion_csv, confusion)
    plot_confusion_png(confusion_png, confusion, spec.label)
    write_dict_csv(
        per_class_csv,
        per_class_rows,
        [
            "class",
            "precision",
            "recall",
            "f1",
            "matched_true_support",
            "matched_pred_support",
            "unmatched_true",
            "unmatched_pred",
        ],
    )
    slide_rows = slide_summaries(image_metrics, n_classes, class_ids)
    write_dict_csv(
        slide_csv,
        slide_rows,
        [
            "slide_id",
            "num_patches",
            "patch_fraction",
            "detection_records",
            "bPQ",
            "mPQ",
            "detection_f1",
            "type_macro_f1",
            "type_weighted_f1",
            "type_accuracy",
            "matched_nuclei",
            "matched_fraction",
            "unmatched_true_nuclei",
            "unmatched_pred_nuclei",
        ],
    )
    dominance = max(
        slide_rows,
        key=lambda row: (row.get("matched_fraction") or 0.0, row.get("patch_fraction") or 0.0),
    )
    row = {
        "label": spec.label,
        "cohort": spec.cohort,
        "matching": matching,
        "run_dir": str(spec.run_dir),
        "inference_results": str(inference_json),
        "num_images": len(image_metrics),
        "detection_records": detection_records,
        "saved_f1_detection": dataset.get("f1_detection"),
        "saved_precision_detection": dataset.get("precision_detection"),
        "saved_recall_detection": dataset.get("recall_detection"),
        "saved_bPQ": dataset.get("bPQ"),
        "saved_mPQ": dataset.get("mPQ"),
        "type_accuracy": summary["type_accuracy"],
        "macro_f1": summary["macro_f1"],
        "weighted_f1": summary["weighted_f1"],
        "matched_nuclei": summary["matched_nuclei"],
        "unmatched_true_nuclei": summary["unmatched_true_nuclei"],
        "unmatched_pred_nuclei": summary["unmatched_pred_nuclei"],
        "detection_f1_from_stats": summary["detection_f1_from_stats"],
        "detection_precision_from_stats": summary["detection_precision_from_stats"],
        "detection_recall_from_stats": summary["detection_recall_from_stats"],
        "num_slides_or_regions": len(slide_rows),
        "dominant_slide_or_region": dominance["slide_id"],
        "dominant_patch_fraction": dominance.get("patch_fraction"),
        "dominant_matched_fraction": dominance.get("matched_fraction"),
        "confusion_csv": str(confusion_csv),
        "confusion_png": str(confusion_png),
        "slide_summary_csv": str(slide_csv),
    }
    for class_row in per_class_rows:
        prefix = safe_slug(class_row["class"])
        row[f"{prefix}_precision"] = class_row["precision"]
        row[f"{prefix}_recall"] = class_row["recall"]
        row[f"{prefix}_f1"] = class_row["f1"]
    detail = {
        "matrix": confusion,
        "slide_rows": slide_rows,
        "per_class_rows": per_class_rows,
    }
    return row, detail


def write_report(
    path,
    rows,
    selection_notes,
    output_info,
):
    best = sorted(
        rows,
        key=lambda row: row.get("macro_f1") if row.get("macro_f1") is not None else -1,
        reverse=True,
    )[:5]
    dominant = [
        row
        for row in rows
        if (row.get("dominant_matched_fraction") or 0.0) >= 0.5
        or (row.get("dominant_patch_fraction") or 0.0) >= 0.5
    ]
    lines = [
        "# Type Assignment Analysis",
        "",
        "## What Existing Outputs Contain",
        "",
        "- Final CellViT inference outputs are `inference_results.json` files under each run log directory.",
        "- `dataset` stores aggregate Dice/Jaccard, bPQ/mPQ, detection F1/precision/recall, DQ/SQ, and tissue accuracy.",
        "- `image_metrics` stores per-patch scalar bPQ/mPQ/DQ/SQ, per-class DQ/SQ/PQ, and `detection_stats`.",
        "- `detection_stats` contains a paired true-vs-predicted type confusion matrix plus unmatched true/predicted type counts.",
        "- The saved files do not contain per-instance masks, contours, centroids, or full true/predicted instance records.",
        "- The small `reports/qualitative/**/{baseline_predictions,predictions}/*.npz` files contain selected prediction `instance_map` and `type_map` arrays only; they are not full-run final outputs and do not include matched GT instances.",
        "",
        "## Metric Definitions In This Repo",
        "",
        "- Detection F1, precision, and recall are computed from `pair_coordinates` on true/predicted centroids, using radius 12 at 40x and 6 at 20x, then `cell_detection_scores`.",
        "- bPQ is the mean patch-level binary panoptic quality from `get_fast_pq` with IoU threshold 0.5 on remapped binary instance maps.",
        "- mPQ is the mean over patch-level per-class PQ values, each computed with `get_fast_pq` at IoU threshold 0.5 on class-specific instance maps.",
        "- Per-class PQ/DQ/SQ are saved from those class-specific PQ computations.",
        "- Per-class F1 in the inference log is `cell_type_detection_scores`, which uses matched centroid pairs plus unmatched true/predicted nuclei.",
        "",
        "## Recomputability",
        "",
        f"- `reports/runs_summary.csv` rows: {output_info['runs_summary_rows']}.",
        f"- Completed or inferred rows checked for F1: {output_info['completed_or_inferred_rows']}.",
        f"- Completed rows missing `test_F1`: {output_info['missing_f1_count']}.",
        "- Matched-nuclei type assignment can be computed from existing `detection_stats` without rerunning inference.",
        "- IoU-based matched type assignment cannot be recomputed from current outputs. Minimal export needed: per patch, save true/pred instance IDs with type labels and either masks/label maps/contours or sufficient geometry to compute pairwise IoU.",
        "",
        "## Summary",
        "",
    ]
    if best:
        lines.append("Top runs by matched-nuclei macro-F1:")
        for row in best:
            lines.append(
                f"- {row['label']}: macro-F1={fmt(row.get('macro_f1'))}, "
                f"weighted-F1={fmt(row.get('weighted_f1'))}, "
                f"accuracy={fmt(row.get('type_accuracy'))}, matched={row.get('matched_nuclei')}"
            )
    else:
        lines.append("No runs were analyzed.")
    lines.extend(["", "## Slide Or Region Dominance", ""])
    if dominant:
        lines.append("Runs where one slide/region contributes at least half of patches or matched nuclei:")
        for row in dominant:
            lines.append(
                f"- {row['label']}: {row['dominant_slide_or_region']} "
                f"patch_fraction={fmt(row.get('dominant_patch_fraction'))}, "
                f"matched_fraction={fmt(row.get('dominant_matched_fraction'))}"
            )
    else:
        lines.append("No analyzed run has a single slide/region contributing at least half of patches or matched nuclei.")
    if selection_notes:
        lines.extend(["", "## Selection Notes", ""])
        lines.extend(f"- {note}" for note in selection_notes)
    lines.extend(
        [
            "",
            "## Outputs",
            "",
            "- `type_f1_summary.csv`: one row per analyzed run.",
            "- Per-run subdirectories contain `confusion_matrix.csv`, `confusion_matrix.png`, `per_class_type_metrics.csv`, and `slide_summary.csv`.",
            "- `klt_frozen_fullft_lora_heads_confusions.png` aggregates KLT Frozen seed42, FullFT seeds 42/43, and LoRA+AF heads_only seeds 42/43 when available.",
            "",
        ]
    )
    path.write_text("\n".join(lines))


def fmt(value):
    if value is None:
        return "NA"
    try:
        value = float(value)
    except (TypeError, ValueError):
        return str(value)
    if math.isnan(value):
        return "NA"
    return f"{value:.4f}"


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--runs-summary",
        type=Path,
        default=DEFAULT_SUMMARY,
        help="CSV summary used for default workshop-run selection.",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=DEFAULT_OUT_DIR,
        help="Output directory for CSVs, PNGs, and the markdown report.",
    )
    parser.add_argument(
        "--run",
        type=Path,
        action="append",
        default=[],
        help="Explicit run log directory or inference_results.json. May be repeated.",
    )
    parser.add_argument(
        "--label",
        action="append",
        default=[],
        help="Label for each explicit --run. Defaults to the directory name.",
    )
    parser.add_argument(
        "--cohort",
        action="append",
        default=[],
        help="Cohort for each explicit --run. Defaults to Explicit.",
    )
    parser.add_argument(
        "--matching",
        choices=("cellvit-detection", "iou"),
        default="cellvit-detection",
        help="Matching basis. Current saved outputs support cellvit-detection.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    if args.run:
        specs = []
        notes = []
        for idx, run_path in enumerate(args.run):
            label = args.label[idx] if idx < len(args.label) else run_path.parent.name
            cohort = args.cohort[idx] if idx < len(args.cohort) else "Explicit"
            if run_path.is_file():
                run_dir = run_path.parent
            else:
                run_dir = run_path
            specs.append(RunSpec(label=label, cohort=cohort, run_dir=run_dir, source="cli"))
    else:
        specs, notes = select_default_runs(args.runs_summary)

    rows = []
    details = {}
    for spec in specs:
        try:
            row, detail = analyze_run(spec, args.out_dir, args.matching)
        except Exception as exc:
            notes.append(f"analysis failed for {spec.label}: {exc}")
            continue
        rows.append(row)
        details[spec.label] = detail

    summary_fields = [
        "label",
        "cohort",
        "matching",
        "num_images",
        "detection_records",
        "saved_f1_detection",
        "saved_precision_detection",
        "saved_recall_detection",
        "saved_bPQ",
        "saved_mPQ",
        "detection_f1_from_stats",
        "detection_precision_from_stats",
        "detection_recall_from_stats",
        "type_accuracy",
        "macro_f1",
        "weighted_f1",
        "matched_nuclei",
        "unmatched_true_nuclei",
        "unmatched_pred_nuclei",
        "num_slides_or_regions",
        "dominant_slide_or_region",
        "dominant_patch_fraction",
        "dominant_matched_fraction",
    ]
    for class_name in CANONICAL_CLASSES:
        prefix = safe_slug(class_name)
        summary_fields.extend([f"{prefix}_precision", f"{prefix}_recall", f"{prefix}_f1"])
    summary_fields.extend(["run_dir", "inference_results", "confusion_csv", "confusion_png", "slide_summary_csv"])
    write_dict_csv(args.out_dir / "type_f1_summary.csv", rows, summary_fields)

    klt_combined = {}
    groups = {
        "Frozen": ["KLT Frozen CellViT seed42"],
        "FullFT": ["KLT FullFT seed42", "KLT FullFT seed43"],
        "LoRA+AF heads": ["KLT LoRA+AF heads_only seed42", "KLT LoRA+AF heads_only seed43"],
    }
    for group_name, labels in groups.items():
        matrices = [details[label]["matrix"] for label in labels if label in details]
        if matrices:
            klt_combined[group_name] = np.sum(matrices, axis=0)
    plot_combined_klt(args.out_dir / "klt_frozen_fullft_lora_heads_confusions.png", klt_combined)

    output_info = inspect_existing_outputs(args.runs_summary)
    write_report(
        args.out_dir / "type_assignment_report.md",
        rows,
        notes,
        output_info,
    )
    print(f"Wrote {len(rows)} run summaries to {args.out_dir / 'type_f1_summary.csv'}")
    if notes:
        print("Notes:")
        for note in notes:
            print(f"- {note}")


if __name__ == "__main__":
    main()
