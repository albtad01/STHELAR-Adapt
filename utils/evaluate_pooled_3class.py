#!/usr/bin/env python3
"""Post-hoc pooled 3-class evaluation for saved STHELAR 5-class inference results.

This utility intentionally does not retrain or rerun inference. It consumes the
saved ``inference_results.json`` produced by the CellViT STHELAR inference code
and remaps saved matched-instance type statistics from 5 classes to pooled
classes.

Important limitation:
    Current saved STHELAR inference files contain per-image paired type
    confusion and unmatched type counts, which are sufficient for pooled
    matched-cell classification and detection metrics. They do not contain the
    per-instance masks or matched-pair IoUs needed to recompute class-aware
    pooled mPQ/mDQ/mSQ exactly after label pooling. The script reports this
    explicitly and preserves class-agnostic bPQ/bDQ/bSQ, which are unaffected by
    label pooling.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
from typing import Any

import numpy as np
import yaml


DEFAULT_SOURCE_CLASSES = {
    "Background": 0,
    "Immune": 1,
    "Stromal": 2,
    "Epithelial": 3,
    "Melanocyte": 4,
    "Other": 5,
}


def parse_bool(value: str | bool) -> bool:
    if isinstance(value, bool):
        return value
    normalized = value.strip().lower()
    if normalized in {"1", "true", "yes", "y", "on"}:
        return True
    if normalized in {"0", "false", "no", "n", "off"}:
        return False
    raise argparse.ArgumentTypeError(f"Invalid boolean value: {value!r}")


def safe_ratio(numerator: float, denominator: float) -> float | None:
    if denominator == 0:
        return None
    return float(numerator / denominator)


def finite_or_none(value: Any) -> Any:
    if isinstance(value, (np.floating, float)):
        value = float(value)
        if math.isnan(value) or math.isinf(value):
            return None
        return value
    if isinstance(value, (np.integer, int)):
        return int(value)
    if isinstance(value, dict):
        return {k: finite_or_none(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [finite_or_none(v) for v in value]
    return value


def find_inference_json(inference_dir: Path) -> Path:
    if inference_dir.is_file():
        return inference_dir
    candidate = inference_dir / "inference_results.json"
    if candidate.is_file():
        return candidate
    raise FileNotFoundError(
        f"Could not find inference_results.json in {inference_dir}"
    )


def load_mapping(mapping_yaml: Path) -> tuple[dict[str, int], dict[str, int], dict[int, int]]:
    with mapping_yaml.open() as handle:
        config = yaml.safe_load(handle)
    if not isinstance(config, dict):
        raise ValueError(f"Mapping YAML is empty or invalid: {mapping_yaml}")

    source_classes = config.get("source_classes") or DEFAULT_SOURCE_CLASSES
    pooled_classes = config.get("pooled_classes")
    mapping = config.get("mapping_5_to_3")
    if not isinstance(source_classes, dict):
        raise ValueError("mapping YAML must define source_classes as a mapping")
    if not isinstance(pooled_classes, dict):
        raise ValueError("mapping YAML must define pooled_classes as a mapping")
    if not isinstance(mapping, dict):
        raise ValueError("mapping YAML must define mapping_5_to_3 as a mapping")

    source_classes = {str(k): int(v) for k, v in source_classes.items()}
    pooled_classes = {str(k): int(v) for k, v in pooled_classes.items()}

    old_to_new: dict[int, int] = {}
    for source_name, pooled_name in mapping.items():
        if source_name not in source_classes:
            raise ValueError(f"Unknown source class in mapping_5_to_3: {source_name}")
        if pooled_name not in pooled_classes:
            raise ValueError(f"Unknown pooled class in mapping_5_to_3: {pooled_name}")
        old_to_new[source_classes[source_name]] = pooled_classes[pooled_name]

    missing = sorted(set(source_classes.values()) - set(old_to_new))
    if missing:
        raise ValueError(f"Missing mapping for source class ids: {missing}")
    return source_classes, pooled_classes, old_to_new


def sorted_names_by_id(classes: dict[str, int]) -> list[str]:
    return [name for name, _ in sorted(classes.items(), key=lambda item: item[1])]


def pad_or_crop_vector(values: Any, length: int) -> np.ndarray:
    arr = np.asarray(values, dtype=np.int64).reshape(-1)
    out = np.zeros(length, dtype=np.int64)
    n = min(length, arr.shape[0])
    out[:n] = arr[:n]
    return out


def pad_or_crop_matrix(values: Any, length: int) -> np.ndarray:
    arr = np.asarray(values, dtype=np.int64)
    out = np.zeros((length, length), dtype=np.int64)
    if arr.ndim != 2:
        return out
    rows = min(length, arr.shape[0])
    cols = min(length, arr.shape[1])
    out[:rows, :cols] = arr[:rows, :cols]
    return out


def aggregate_detection_stats(
    image_metrics: dict[str, Any], n_source_classes: int
) -> tuple[np.ndarray, np.ndarray, np.ndarray, int]:
    paired_confusion = np.zeros((n_source_classes, n_source_classes), dtype=np.int64)
    unpaired_true = np.zeros(n_source_classes, dtype=np.int64)
    unpaired_pred = np.zeros(n_source_classes, dtype=np.int64)
    n_records = 0

    for record in image_metrics.values():
        stats = record.get("detection_stats") if isinstance(record, dict) else None
        if not stats:
            continue
        n_records += 1
        if "paired_confusion" in stats:
            paired_confusion += pad_or_crop_matrix(
                stats["paired_confusion"], n_source_classes
            )
            unpaired_true += pad_or_crop_vector(
                stats.get("unpaired_true_counts", []), n_source_classes
            )
            unpaired_pred += pad_or_crop_vector(
                stats.get("unpaired_pred_counts", []), n_source_classes
            )
            continue

        # Backward-compatible support for early detailed result files.
        paired_true = np.asarray(stats.get("paired_true", []), dtype=int)
        paired_pred = np.asarray(stats.get("paired_pred", []), dtype=int)
        for true_id, pred_id in zip(paired_true, paired_pred):
            if 0 <= true_id < n_source_classes and 0 <= pred_id < n_source_classes:
                paired_confusion[true_id, pred_id] += 1
        unpaired_true += np.bincount(
            np.asarray(stats.get("unpaired_true", []), dtype=int),
            minlength=n_source_classes,
        )[:n_source_classes]
        unpaired_pred += np.bincount(
            np.asarray(stats.get("unpaired_pred", []), dtype=int),
            minlength=n_source_classes,
        )[:n_source_classes]

    return paired_confusion, unpaired_true, unpaired_pred, n_records


def remap_detection_stats(
    paired_confusion: np.ndarray,
    unpaired_true: np.ndarray,
    unpaired_pred: np.ndarray,
    old_to_new: dict[int, int],
    n_pooled_classes: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    pooled_confusion = np.zeros((n_pooled_classes, n_pooled_classes), dtype=np.int64)
    pooled_unpaired_true = np.zeros(n_pooled_classes, dtype=np.int64)
    pooled_unpaired_pred = np.zeros(n_pooled_classes, dtype=np.int64)

    for old_true in range(paired_confusion.shape[0]):
        new_true = old_to_new.get(old_true)
        if new_true is None:
            continue
        for old_pred in range(paired_confusion.shape[1]):
            new_pred = old_to_new.get(old_pred)
            if new_pred is None:
                continue
            pooled_confusion[new_true, new_pred] += int(
                paired_confusion[old_true, old_pred]
            )

    for old_id, new_id in old_to_new.items():
        if old_id < len(unpaired_true):
            pooled_unpaired_true[new_id] += int(unpaired_true[old_id])
        if old_id < len(unpaired_pred):
            pooled_unpaired_pred[new_id] += int(unpaired_pred[old_id])

    return pooled_confusion, pooled_unpaired_true, pooled_unpaired_pred


def class_metrics_from_confusion(
    confusion: np.ndarray, class_ids: list[int], class_names_by_id: list[str]
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    f1_values: list[float] = []
    total = int(confusion[np.ix_(class_ids, class_ids)].sum())
    correct = int(sum(confusion[class_id, class_id] for class_id in class_ids))

    for class_id in class_ids:
        tp = int(confusion[class_id, class_id])
        support_true = int(confusion[class_id, class_ids].sum())
        support_pred = int(confusion[class_ids, class_id].sum())
        precision = safe_ratio(tp, support_pred)
        recall = safe_ratio(tp, support_true)
        if precision is None and recall is None:
            f1 = None
        elif precision is None or recall is None or (precision + recall) == 0:
            f1 = 0.0
        else:
            f1 = float(2 * precision * recall / (precision + recall))
        if f1 is not None:
            f1_values.append(f1)
        rows.append(
            {
                "class_id": class_id,
                "class_name": class_names_by_id[class_id],
                "support_true": support_true,
                "support_pred": support_pred,
                "tp_classification": tp,
                "classification_precision": precision,
                "classification_recall": recall,
                "classification_f1": f1,
            }
        )

    summary = {
        "accuracy": safe_ratio(correct, total),
        "macro_f1": float(np.mean(f1_values)) if f1_values else None,
        "evaluated_matched_instances": total,
        "correct_matched_instances": correct,
    }
    return rows, summary


def detection_summary(
    confusion: np.ndarray,
    unpaired_true: np.ndarray,
    unpaired_pred: np.ndarray,
    class_ids: list[int],
) -> dict[str, Any]:
    paired = int(confusion[np.ix_(class_ids, class_ids)].sum())
    fp = int(unpaired_pred[class_ids].sum())
    fn = int(unpaired_true[class_ids].sum())
    return {
        "f1_detection": safe_ratio(2 * paired, 2 * paired + fp + fn),
        "precision_detection": safe_ratio(paired, paired + fp),
        "recall_detection": safe_ratio(paired, paired + fn),
        "paired_instances": paired,
        "unpaired_true_instances": fn,
        "unpaired_pred_instances": fp,
    }


def per_class_detection_metrics(
    confusion: np.ndarray,
    unpaired_true: np.ndarray,
    unpaired_pred: np.ndarray,
    class_ids: list[int],
    class_names_by_id: list[str],
) -> dict[int, dict[str, Any]]:
    output = {}
    for class_id in class_ids:
        tp = int(confusion[class_id, class_id])
        fp_type = int(confusion[class_ids, class_id].sum() - tp)
        fn_type = int(confusion[class_id, class_ids].sum() - tp)
        fp_det = int(unpaired_pred[class_id])
        fn_det = int(unpaired_true[class_id])
        precision_den = tp + 2 * fp_type + fp_det
        recall_den = tp + 2 * fn_type + fn_det
        f1_den = 2 * tp + 2 * fp_type + 2 * fn_type + fp_det + fn_det
        output[class_id] = {
            "class_name": class_names_by_id[class_id],
            "detection_precision": safe_ratio(tp, precision_den),
            "detection_recall": safe_ratio(tp, recall_den),
            "detection_f1": safe_ratio(2 * tp, f1_den),
            "detection_tp_type_correct": tp,
            "detection_fp_type": fp_type,
            "detection_fn_type": fn_type,
            "detection_fp_unpaired": fp_det,
            "detection_fn_unpaired": fn_det,
        }
    return output


def inspect_pq_recomputability(image_metrics: dict[str, Any]) -> dict[str, Any]:
    records = [record for record in image_metrics.values() if isinstance(record, dict)]
    if not records:
        return {
            "possible": False,
            "missing": ["image_metrics records"],
            "note": "No per-image records were found.",
        }

    has_instance_maps = all(
        {
            "true_instance_map",
            "pred_instance_map",
            "true_instance_types",
            "pred_instance_types",
        }.issubset(record)
        for record in records
    )
    has_pair_iou = all(
        {
            "paired_true_types",
            "paired_pred_types",
            "paired_ious",
            "unpaired_true_types",
            "unpaired_pred_types",
        }.issubset(record)
        for record in records
    )
    if has_instance_maps or has_pair_iou:
        return {
            "possible": True,
            "missing": [],
            "note": (
                "The file appears to contain detailed instance/PQ data, but this "
                "script currently implements the saved STHELAR detection_stats "
                "post-hoc path only."
            ),
        }
    return {
        "possible": False,
        "missing": [
            "per-instance ground-truth masks or matched true instance ids",
            "per-instance predicted masks or matched predicted instance ids",
            "matched-pair IoU/intersection-union values",
            "matched/unmatched instance type lists with enough geometry to recompute class-aware PQ",
        ],
        "note": (
            "Saved inference_results.json contains only already-aggregated "
            "per-class PQ/DQ/SQ and detection_stats. Pooled mPQ/mDQ/mSQ must be "
            "recomputed from instance-level predictions/ground truth before "
            "aggregation; transforming the old 5-class PQ values would be wrong."
        ),
    }


def write_key_value_csv(path: Path, rows: list[tuple[str, Any]]) -> None:
    with path.open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["metric", "value"])
        for key, value in rows:
            writer.writerow([key, "" if value is None else value])


def write_dict_rows_csv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str]) -> None:
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow({key: "" if row.get(key) is None else row.get(key) for key in fieldnames})


def write_confusion_csv(
    path: Path, confusion: np.ndarray, class_ids: list[int], class_names_by_id: list[str]
) -> None:
    labels = [class_names_by_id[class_id] for class_id in class_ids]
    with path.open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["true\\pred", *labels])
        for true_id in class_ids:
            writer.writerow(
                [
                    class_names_by_id[true_id],
                    *[int(confusion[true_id, pred_id]) for pred_id in class_ids],
                ]
            )


def build_outputs(
    inference_json: Path,
    mapping_yaml: Path,
    output_prefix: Path,
    exclude_background: bool,
) -> dict[str, Any]:
    with inference_json.open() as handle:
        inference_results = json.load(handle)
    image_metrics = inference_results.get("image_metrics")
    if not isinstance(image_metrics, dict):
        raise ValueError("inference_results.json does not contain image_metrics dict")

    source_classes, pooled_classes, old_to_new = load_mapping(mapping_yaml)
    source_names_by_id = sorted_names_by_id(source_classes)
    pooled_names_by_id = sorted_names_by_id(pooled_classes)
    n_source = max(source_classes.values()) + 1
    n_pooled = max(pooled_classes.values()) + 1
    if len(source_names_by_id) != n_source or len(pooled_names_by_id) != n_pooled:
        raise ValueError("Class ids must be contiguous starting at 0")

    paired_5, unpaired_true_5, unpaired_pred_5, detection_records = (
        aggregate_detection_stats(image_metrics, n_source)
    )
    if detection_records == 0:
        raise ValueError(
            "No per-image detection_stats found. Cannot compute post-hoc pooled "
            "classification/detection metrics from this inference output."
        )
    paired_3, unpaired_true_3, unpaired_pred_3 = remap_detection_stats(
        paired_5, unpaired_true_5, unpaired_pred_5, old_to_new, n_pooled
    )

    source_class_ids = list(range(n_source))
    pooled_class_ids = list(range(n_pooled))
    if exclude_background:
        source_class_ids = [
            class_id
            for class_id, name in enumerate(source_names_by_id)
            if name.lower() != "background"
        ]
        pooled_class_ids = [
            class_id
            for class_id, name in enumerate(pooled_names_by_id)
            if name.lower() != "background"
        ]

    original_rows, original_summary = class_metrics_from_confusion(
        paired_5, source_class_ids, source_names_by_id
    )
    pooled_rows, pooled_classification = class_metrics_from_confusion(
        paired_3, pooled_class_ids, pooled_names_by_id
    )
    pooled_detection = detection_summary(
        paired_3, unpaired_true_3, unpaired_pred_3, pooled_class_ids
    )
    pooled_per_class_detection = per_class_detection_metrics(
        paired_3, unpaired_true_3, unpaired_pred_3, pooled_class_ids, pooled_names_by_id
    )
    pq_recomputability = inspect_pq_recomputability(image_metrics)

    dataset_metrics = inference_results.get("dataset", {})
    pq_summary = {
        "pooled_bPQ": dataset_metrics.get("bPQ"),
        "pooled_bDQ": dataset_metrics.get("bDQ"),
        "pooled_bSQ": dataset_metrics.get("bSQ"),
        "pooled_mPQ": None,
        "pooled_mDQ": None,
        "pooled_mSQ": None,
        "pq_recompute_possible": bool(pq_recomputability["possible"]),
        "pq_recompute_note": pq_recomputability["note"],
        "missing_for_pooled_mPQ": pq_recomputability["missing"],
    }

    per_class_rows = []
    for row in pooled_rows:
        class_id = row["class_id"]
        merged = dict(row)
        merged.update(pooled_per_class_detection.get(class_id, {}))
        merged.update(
            {
                "PQ": None,
                "DQ": None,
                "SQ": None,
                "pq_status": (
                    "not_recomputed_missing_instance_level_outputs"
                    if not pq_recomputability["possible"]
                    else "not_implemented_for_detailed_outputs"
                ),
            }
        )
        per_class_rows.append(merged)

    output_prefix.parent.mkdir(parents=True, exist_ok=True)
    metrics_json_path = output_prefix.with_name(output_prefix.name + "_pooled3_metrics.json")
    classification_summary_path = output_prefix.with_name(
        output_prefix.name + "_pooled3_classification_summary.csv"
    )
    confusion_path = output_prefix.with_name(
        output_prefix.name + "_pooled3_classification_confusion.csv"
    )
    pq_summary_path = output_prefix.with_name(output_prefix.name + "_pooled3_pq_summary.csv")
    per_class_path = output_prefix.with_name(
        output_prefix.name + "_pooled3_per_class_metrics.csv"
    )

    summary_rows = [
        ("inference_results", str(inference_json)),
        ("mapping_yaml", str(mapping_yaml)),
        ("num_images", len(image_metrics)),
        ("detection_records", detection_records),
        ("exclude_background", exclude_background),
        ("original_5class_macro_f1_matched", original_summary["macro_f1"]),
        ("original_5class_evaluated_matched_instances", original_summary["evaluated_matched_instances"]),
        ("pooled3_macro_f1_matched", pooled_classification["macro_f1"]),
        ("pooled3_accuracy_matched", pooled_classification["accuracy"]),
        ("pooled3_evaluated_matched_instances", pooled_classification["evaluated_matched_instances"]),
        ("pooled3_f1_detection", pooled_detection["f1_detection"]),
        ("pooled3_precision_detection", pooled_detection["precision_detection"]),
        ("pooled3_recall_detection", pooled_detection["recall_detection"]),
        ("pooled3_paired_instances", pooled_detection["paired_instances"]),
        ("pooled3_unpaired_true_instances", pooled_detection["unpaired_true_instances"]),
        ("pooled3_unpaired_pred_instances", pooled_detection["unpaired_pred_instances"]),
        ("pooled_mPQ_recomputed", pq_recomputability["possible"]),
        ("pooled_mPQ_note", pq_recomputability["note"]),
    ]
    write_key_value_csv(classification_summary_path, summary_rows)
    write_confusion_csv(confusion_path, paired_3, pooled_class_ids, pooled_names_by_id)

    pq_rows = [
        {
            "metric": "pooled_bPQ",
            "value": pq_summary["pooled_bPQ"],
            "status": "class_agnostic_original_metric_reused",
            "note": "bPQ is class-agnostic and unaffected by label pooling.",
        },
        {
            "metric": "pooled_bDQ",
            "value": pq_summary["pooled_bDQ"],
            "status": "class_agnostic_original_metric_reused",
            "note": "bDQ is class-agnostic and unaffected by label pooling.",
        },
        {
            "metric": "pooled_bSQ",
            "value": pq_summary["pooled_bSQ"],
            "status": "class_agnostic_original_metric_reused",
            "note": "bSQ is class-agnostic and unaffected by label pooling.",
        },
        {
            "metric": "pooled_mPQ",
            "value": None,
            "status": "not_recomputed_missing_instance_level_outputs",
            "note": pq_recomputability["note"],
        },
        {
            "metric": "pooled_mDQ",
            "value": None,
            "status": "not_recomputed_missing_instance_level_outputs",
            "note": pq_recomputability["note"],
        },
        {
            "metric": "pooled_mSQ",
            "value": None,
            "status": "not_recomputed_missing_instance_level_outputs",
            "note": pq_recomputability["note"],
        },
    ]
    write_dict_rows_csv(
        pq_summary_path,
        pq_rows,
        fieldnames=["metric", "value", "status", "note"],
    )
    write_dict_rows_csv(
        per_class_path,
        per_class_rows,
        fieldnames=[
            "class_id",
            "class_name",
            "support_true",
            "support_pred",
            "tp_classification",
            "classification_precision",
            "classification_recall",
            "classification_f1",
            "detection_precision",
            "detection_recall",
            "detection_f1",
            "detection_tp_type_correct",
            "detection_fp_type",
            "detection_fn_type",
            "detection_fp_unpaired",
            "detection_fn_unpaired",
            "PQ",
            "DQ",
            "SQ",
            "pq_status",
        ],
    )

    metrics_json = {
        "inputs": {
            "inference_results": str(inference_json),
            "mapping_yaml": str(mapping_yaml),
            "output_prefix": str(output_prefix),
            "exclude_background": exclude_background,
        },
        "source_classes": source_classes,
        "pooled_classes": pooled_classes,
        "old_to_new_class_id": {str(k): v for k, v in old_to_new.items()},
        "counts": {
            "num_images": len(image_metrics),
            "detection_records": detection_records,
        },
        "original_5class_matched_classification": {
            "summary": original_summary,
            "per_class": original_rows,
        },
        "pooled3_matched_classification": {
            "summary": pooled_classification,
            "per_class": pooled_rows,
            "confusion_matrix": {
                "class_names": [pooled_names_by_id[i] for i in pooled_class_ids],
                "raw": paired_3[np.ix_(pooled_class_ids, pooled_class_ids)].tolist(),
            },
        },
        "pooled3_detection": {
            "summary": pooled_detection,
            "per_class": {
                pooled_names_by_id[class_id]: metrics
                for class_id, metrics in pooled_per_class_detection.items()
            },
        },
        "pooled3_panoptic": pq_summary,
        "original_dataset_metrics": dataset_metrics,
        "output_files": {
            "metrics_json": str(metrics_json_path),
            "classification_summary_csv": str(classification_summary_path),
            "classification_confusion_csv": str(confusion_path),
            "pq_summary_csv": str(pq_summary_path),
            "per_class_metrics_csv": str(per_class_path),
        },
    }
    with metrics_json_path.open("w") as handle:
        json.dump(finite_or_none(metrics_json), handle, indent=2, allow_nan=False)

    return metrics_json


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate pooled 3-class metrics from saved STHELAR 5-class inference outputs."
    )
    parser.add_argument(
        "--inference-dir",
        required=True,
        type=Path,
        help="Run log directory containing inference_results.json, or the JSON file itself.",
    )
    parser.add_argument("--mapping-yaml", required=True, type=Path)
    parser.add_argument("--output-prefix", required=True, type=Path)
    parser.add_argument(
        "--exclude-background",
        type=parse_bool,
        default=True,
        help="Whether to exclude Background from classification/detection summaries.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    inference_json = find_inference_json(args.inference_dir)
    outputs = build_outputs(
        inference_json=inference_json,
        mapping_yaml=args.mapping_yaml,
        output_prefix=args.output_prefix,
        exclude_background=args.exclude_background,
    )
    summary = outputs["pooled3_matched_classification"]["summary"]
    detection = outputs["pooled3_detection"]["summary"]
    pq = outputs["pooled3_panoptic"]
    print(f"Read: {inference_json}")
    print(f"Wrote prefix: {args.output_prefix}")
    print(
        "Pooled3 matched classification: "
        f"macro_f1={summary['macro_f1']} "
        f"accuracy={summary['accuracy']} "
        f"n={summary['evaluated_matched_instances']}"
    )
    print(
        "Pooled3 detection: "
        f"f1={detection['f1_detection']} "
        f"precision={detection['precision_detection']} "
        f"recall={detection['recall_detection']}"
    )
    print(
        "Pooled3 panoptic: "
        f"bPQ={pq['pooled_bPQ']} mPQ={pq['pooled_mPQ']} "
        f"recompute_possible={pq['pq_recompute_possible']}"
    )
    if not pq["pq_recompute_possible"]:
        print("Pooled mPQ/mDQ/mSQ not recomputed:", pq["pq_recompute_note"])
        print("Missing:")
        for item in pq["missing_for_pooled_mPQ"]:
            print(f"  - {item}")


if __name__ == "__main__":
    main()
