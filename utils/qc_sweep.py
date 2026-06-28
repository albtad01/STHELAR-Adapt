#!/usr/bin/env python3
"""Aggregate saved per-patch inference metrics over QC thresholds or bins."""

import argparse
import csv
import json
import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import yaml


SUMMARY_METRICS = (
    "Dice",
    "Jaccard",
    "bPQ",
    "bDQ",
    "bSQ",
    "mPQ",
    "mDQ",
    "mSQ",
    "f1_detection",
    "precision_detection",
    "recall_detection",
)


def normalize_thresholds(values):
    if values is None:
        return []
    normalized = []
    for value in values:
        if isinstance(value, str) and value.lower() == "all":
            value = 0.0
        value = float(value)
        if value not in normalized:
            normalized.append(value)
    return normalized


def _finite_mean(values):
    values = np.asarray(values, dtype=float)
    if values.size == 0 or np.all(np.isnan(values)):
        return None
    return float(np.nanmean(values))


def _safe_ratio(numerator, denominator):
    return float(numerator / denominator) if denominator else None


def _detection_metrics(records, n_classes):
    paired_confusion = np.zeros((n_classes, n_classes), dtype=np.int64)
    unpaired_true_counts = np.zeros(n_classes, dtype=np.int64)
    unpaired_pred_counts = np.zeros(n_classes, dtype=np.int64)
    found = False
    for record in records:
        stats = record.get("detection_stats")
        if not stats:
            continue
        found = True
        if "paired_confusion" in stats:
            paired_confusion += np.asarray(stats["paired_confusion"], dtype=np.int64)
            unpaired_true_counts += np.asarray(
                stats["unpaired_true_counts"], dtype=np.int64
            )
            unpaired_pred_counts += np.asarray(
                stats["unpaired_pred_counts"], dtype=np.int64
            )
            continue

        # Backward-compatible support for early detailed result files.
        paired_true = np.asarray(stats.get("paired_true", []), dtype=int)
        paired_pred = np.asarray(stats.get("paired_pred", []), dtype=int)
        for true_id, pred_id in zip(paired_true, paired_pred):
            if 0 <= true_id < n_classes and 0 <= pred_id < n_classes:
                paired_confusion[true_id, pred_id] += 1
        unpaired_true_counts += np.bincount(
            np.asarray(stats.get("unpaired_true", []), dtype=int), minlength=n_classes
        )[:n_classes]
        unpaired_pred_counts += np.bincount(
            np.asarray(stats.get("unpaired_pred", []), dtype=int), minlength=n_classes
        )[:n_classes]

    if not found:
        return None
    tp = int(paired_confusion.sum())
    fp = int(unpaired_pred_counts.sum())
    fn = int(unpaired_true_counts.sum())
    f1_detection = _safe_ratio(2 * tp, 2 * tp + fp + fn)
    return {
        "f1_detection": f1_detection,
        "F1_detection": f1_detection,
        "precision_detection": _safe_ratio(tp, tp + fp),
        "recall_detection": _safe_ratio(tp, tp + fn),
        "paired_confusion": paired_confusion,
        "unpaired_true_counts": unpaired_true_counts,
        "unpaired_pred_counts": unpaired_pred_counts,
    }


def _per_class_detection(detection, nuclei_types):
    if detection is None:
        return None
    output = {}
    paired_confusion = detection["paired_confusion"]
    unpaired_true_counts = detection["unpaired_true_counts"]
    unpaired_pred_counts = detection["unpaired_pred_counts"]
    for class_name, class_id in nuclei_types.items():
        if class_name.lower() == "background":
            continue
        tp = int(paired_confusion[class_id, class_id])
        fp_type = int(paired_confusion[:, class_id].sum() - tp)
        fn_type = int(paired_confusion[class_id, :].sum() - tp)
        fp_det = int(unpaired_pred_counts[class_id])
        fn_det = int(unpaired_true_counts[class_id])
        correct = tp
        precision_den = correct + 2 * fp_type + fp_det
        recall_den = correct + 2 * fn_type + fn_det
        f1_den = 2 * correct + 2 * fp_type + 2 * fn_type + fp_det + fn_det
        f1_detection = _safe_ratio(2 * correct, f1_den)
        output[class_name] = {
            "precision": _safe_ratio(correct, precision_den),
            "recall": _safe_ratio(correct, recall_den),
            "f1_detection": f1_detection,
            "F1": f1_detection,
        }
    return output


def _confusion_matrix(detection, nuclei_types):
    if detection is None:
        return None
    class_names = [
        name for name, _ in sorted(nuclei_types.items(), key=lambda item: item[1])
    ]
    n_classes = len(class_names)
    matrix = detection["paired_confusion"]
    row_sums = matrix.sum(axis=1, keepdims=True)
    normalized = np.divide(
        matrix,
        row_sums,
        out=np.zeros_like(matrix, dtype=float),
        where=row_sums != 0,
    )
    return {
        "class_names": class_names,
        "raw": matrix.tolist(),
        "normalized": normalized.tolist(),
    }


def aggregate_selection(image_metrics, selected_names, nuclei_types):
    records = [image_metrics[name] for name in selected_names]
    metrics = {}
    for metric in ("Dice", "Jaccard", "bPQ", "bDQ", "bSQ", "mPQ", "mDQ", "mSQ"):
        metrics[metric] = _finite_mean(
            [record.get(metric, float("nan")) for record in records]
        )

    detection = _detection_metrics(records, len(nuclei_types))
    for key in ("f1_detection", "precision_detection", "recall_detection"):
        metrics[key] = detection.get(key) if detection is not None else None
    # Keep the original key as a JSON compatibility alias.
    metrics["F1_detection"] = metrics["f1_detection"]

    per_class = {}
    has_per_class = any(record.get("per_class") for record in records)
    if has_per_class:
        for class_name, _ in sorted(nuclei_types.items(), key=lambda item: item[1]):
            if class_name.lower() == "background":
                continue
            per_class[class_name] = {
                metric: _finite_mean(
                    [
                        record.get("per_class", {}).get(class_name, {}).get(
                            metric, float("nan")
                        )
                        for record in records
                    ]
                )
                for metric in ("DQ", "SQ", "PQ")
            }

    return {
        **metrics,
        "per_class_DQ_SQ_PQ": per_class or None,
        "per_class_detection": _per_class_detection(detection, nuclei_types),
        "confusion_matrix": _confusion_matrix(detection, nuclei_types),
        "has_detailed_detection_statistics": detection is not None,
        "has_per_class_pq_statistics": has_per_class,
    }


def format_qc_record_lines(record):
    scalar_keys = (
        "Dice",
        "Jaccard",
        "bPQ",
        "bDQ",
        "bSQ",
        "mPQ",
        "mDQ",
        "mSQ",
        "f1_detection",
        "precision_detection",
        "recall_detection",
    )
    scalar_text = " ".join(f"{key}={record.get(key)}" for key in scalar_keys)
    lines = [
        f"QC sweep {record['selection']}: retained={record['n_retained']}/"
        f"{record['n_total_test_patches']} {scalar_text}"
    ]
    pq_by_class = record.get("per_class_DQ_SQ_PQ") or {}
    detection_by_class = record.get("per_class_detection") or {}
    for class_name in sorted(set(pq_by_class) | set(detection_by_class)):
        class_pq = pq_by_class.get(class_name, {})
        class_detection = detection_by_class.get(class_name, {})
        lines.append(
            f"QC sweep {record['selection']} class={class_name}: "
            f"PQ={class_pq.get('PQ')} DQ={class_pq.get('DQ')} "
            f"SQ={class_pq.get('SQ')} "
            f"f1_detection={class_detection.get('f1_detection', class_detection.get('F1'))} "
            f"precision_detection={class_detection.get('precision')} "
            f"recall_detection={class_detection.get('recall')}"
        )
    return lines


def compute_qc_sweep(
    image_metrics,
    patch_info,
    nuclei_types,
    qc_metric,
    thresholds,
    bins=None,
    low_count_warning=50,
    low_fraction_warning=0.05,
):
    if "packed_file_name" not in patch_info.columns:
        raise ValueError("patch_info_with_split.csv has no packed_file_name column")
    if qc_metric not in patch_info.columns:
        raise ValueError(f"QC metric column not found: {qc_metric}")

    patch_info = patch_info.copy()
    patch_info["packed_file_name"] = patch_info["packed_file_name"].astype(str)
    patch_info = patch_info.drop_duplicates("packed_file_name").set_index(
        "packed_file_name", drop=False
    )
    image_names = sorted(str(name) for name in image_metrics)
    missing = sorted(set(image_names) - set(patch_info.index))
    if missing:
        raise ValueError(
            f"{len(missing)} inferred patch IDs are missing from patch_info_with_split.csv; "
            f"examples: {missing[:5]}"
        )

    inferred_info = patch_info.loc[image_names]
    qc_values = pd.to_numeric(inferred_info[qc_metric], errors="coerce")
    total = len(image_names)
    warnings = []

    def build_result(label, selected, threshold=None, low=None, high=None):
        selected = sorted(selected)
        discarded = total - len(selected)
        retained_percent = 100.0 * len(selected) / total if total else 0.0
        slide_counts = {}
        if "slide_id" in inferred_info.columns:
            slide_counts = (
                inferred_info.loc[selected, "slide_id"].astype(str).value_counts().to_dict()
                if selected
                else {}
            )
        warning = None
        if len(selected) < low_count_warning or (
            total and len(selected) / total < low_fraction_warning
        ):
            warning = (
                f"QC selection {label} retains only {len(selected)}/{total} patches "
                f"({retained_percent:.2f}%). Interpret metrics cautiously."
            )
            warnings.append(warning)
        result = {
            "selection": label,
            "qc_metric": qc_metric,
            "qc_threshold": threshold,
            "qc_bin_low": low,
            "qc_bin_high": high,
            "n_total_test_patches": total,
            "n_retained": len(selected),
            "n_discarded": discarded,
            "retained_percent": retained_percent,
            "retained_per_slide": slide_counts,
            "warning": warning,
        }
        result.update(aggregate_selection(image_metrics, selected, nuclei_types))
        return result

    cumulative = []
    for threshold in normalize_thresholds(thresholds):
        if threshold <= 0:
            selected = image_names
        else:
            selected = qc_values.index[qc_values >= threshold].tolist()
        cumulative.append(
            build_result(f"{qc_metric}_ge_{threshold:g}", selected, threshold=threshold)
        )

    bin_results = []
    normalized_bins = normalize_thresholds(bins)
    if normalized_bins:
        if len(normalized_bins) < 2:
            raise ValueError("qc_bins needs at least two ordered boundaries")
        if normalized_bins != sorted(normalized_bins):
            raise ValueError("qc_bins must be sorted in ascending order")
        for low, high in zip(normalized_bins[:-1], normalized_bins[1:]):
            selected = qc_values.index[(qc_values >= low) & (qc_values < high)].tolist()
            bin_results.append(
                build_result(
                    f"{qc_metric}_{low:g}_to_{high:g}",
                    selected,
                    low=low,
                    high=high,
                )
            )

    return {
        "qc_metric": qc_metric,
        "n_total_test_patches": total,
        "thresholds": normalize_thresholds(thresholds),
        "bins": normalized_bins,
        "cumulative": cumulative,
        "bin_results": bin_results,
        "warnings": warnings,
    }


def _sanitize_json(value):
    if isinstance(value, dict):
        return {key: _sanitize_json(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_sanitize_json(item) for item in value]
    if isinstance(value, float) and (math.isnan(value) or math.isinf(value)):
        return None
    if isinstance(value, np.generic):
        return _sanitize_json(value.item())
    return value


def save_qc_sweep(result, out_dir):
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    safe_metric = "".join(
        character if character.isalnum() or character in "-_" else "_"
        for character in result["qc_metric"]
    )
    json_path = out_dir / f"inference_results_qc_sweep_{safe_metric}.json"
    csv_path = out_dir / f"inference_results_qc_sweep_{safe_metric}_summary.csv"
    json_path.write_text(json.dumps(_sanitize_json(result), indent=2) + "\n")

    rows = []
    for selection_type, records in (
        ("cumulative", result["cumulative"]),
        ("bin", result["bin_results"]),
    ):
        for record in records:
            row = {
                "selection_type": selection_type,
                "selection": record["selection"],
                "qc_metric": record["qc_metric"],
                "qc_threshold": record["qc_threshold"],
                "qc_bin_low": record["qc_bin_low"],
                "qc_bin_high": record["qc_bin_high"],
                "n_total_test_patches": record["n_total_test_patches"],
                "n_retained": record["n_retained"],
                "n_discarded": record["n_discarded"],
                "retained_percent": record["retained_percent"],
                "retained_per_slide": json.dumps(record["retained_per_slide"], sort_keys=True),
                "warning": record["warning"],
            }
            row.update({metric: record.get(metric) for metric in SUMMARY_METRICS})
            for class_name, class_metrics in (
                record.get("per_class_DQ_SQ_PQ") or {}
            ).items():
                safe_class = "".join(
                    character if character.isalnum() else "_"
                    for character in class_name
                )
                for metric in ("PQ", "DQ", "SQ"):
                    row[f"class_{safe_class}_{metric}"] = class_metrics.get(metric)
            for class_name, class_metrics in (
                record.get("per_class_detection") or {}
            ).items():
                safe_class = "".join(
                    character if character.isalnum() else "_"
                    for character in class_name
                )
                row[f"class_{safe_class}_f1_detection"] = class_metrics.get(
                    "f1_detection", class_metrics.get("F1")
                )
                row[f"class_{safe_class}_precision_detection"] = class_metrics.get(
                    "precision"
                )
                row[f"class_{safe_class}_recall_detection"] = class_metrics.get(
                    "recall"
                )
            rows.append(row)

    fieldnames = []
    for row in rows:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(key)
    if not fieldnames:
        fieldnames = ["selection_type", "selection"]
    with csv_path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    return json_path, csv_path


def sweep_saved_results(run_dir, qc_metric, thresholds, bins=None):
    run_dir = Path(run_dir).expanduser().resolve()
    with (run_dir / "config.yaml").open() as handle:
        config = yaml.safe_load(handle)
    with (run_dir / "inference_results.json").open() as handle:
        inference_results = json.load(handle)
    dataset_path = Path(config["data"]["dataset_path"])
    patch_info = pd.read_csv(dataset_path / "patch_info_with_split.csv")
    with (dataset_path / "dataset_config.yaml").open() as handle:
        dataset_config = yaml.safe_load(handle)
    result = compute_qc_sweep(
        image_metrics=inference_results["image_metrics"],
        patch_info=patch_info,
        nuclei_types=dataset_config["nuclei_types"],
        qc_metric=qc_metric,
        thresholds=thresholds,
        bins=bins,
    )
    return result, save_qc_sweep(result, run_dir)


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--qc-metric", default="Jaccard")
    parser.add_argument("--qc-thresholds", nargs="+", default=[0.0, 0.45, 0.6, 0.7, 0.8])
    parser.add_argument("--qc-bins", nargs="*", default=None)
    return parser.parse_args()


def main():
    args = parse_args()
    result, paths = sweep_saved_results(
        args.run_dir,
        args.qc_metric,
        normalize_thresholds(args.qc_thresholds),
        normalize_thresholds(args.qc_bins),
    )
    print(f"QC sweep JSON: {paths[0]}")
    print(f"QC sweep CSV:  {paths[1]}")
    for record in result["cumulative"] + result["bin_results"]:
        for line in format_qc_record_lines(record):
            print(line)
    for warning in result["warnings"]:
        print(f"WARNING: {warning}", file=sys.stderr)


if __name__ == "__main__":
    main()
