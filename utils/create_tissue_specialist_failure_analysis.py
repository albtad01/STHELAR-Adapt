#!/usr/bin/env python3
"""Build the canonical seed-42 KLT generalist/specialist failure analysis.

This is a read-only analysis of prepared split metadata, training logs, and
checkpoint-10 TEST inference JSONs.  It never selects or rewrites a checkpoint.
"""

import csv
import json
import math
import re
from collections import OrderedDict
from pathlib import Path


REPO = Path(__file__).resolve().parents[1]
DATA_ROOT = REPO.parent / "Datasets" / "cellvit_ready"
MASTER_CSV = REPO / "reports" / "workshop_master_results.csv"
OUT_MD = REPO / "reports" / "tissue_specialist_failure_analysis.md"
OUT_CSV = REPO / "reports" / "tissue_specialist_failure_analysis.csv"

CLASSES = ["Immune", "Stromal", "Epithelial", "Melanocyte", "Other"]
ALL_CLASSES = ["Background"] + CLASSES
TISSUES = ["Kidney", "Liver", "Tonsil"]
FOLDS = ["A", "B"]
MODELS = ["Generalist Selected PEFT", "Tissue-specific Selected PEFT"]


def finite_mean(values):
    values = [float(v) for v in values if isinstance(v, (int, float)) and math.isfinite(v)]
    return sum(values) / len(values) if values else float("nan")


def fmt(value, digits=4):
    if value is None or not isinstance(value, (int, float)) or not math.isfinite(value):
        return "--"
    return ("{:,." + str(digits) + "f}").format(value)


def pct(value):
    return fmt(100.0 * value, 2) + "%" if math.isfinite(value) else "--"


def data_dir(tissue, fold, generalist):
    if generalist:
        name = "sthelar40x_klt_5class_slideind_fold{}_margin128".format(fold)
    else:
        suffix = "_cap50000" if tissue == "Tonsil" else ""
        name = "sthelar40x_{}_5class_slideind_fold{}_margin128{}".format(
            tissue.lower(), fold, suffix
        )
    path = DATA_ROOT / name
    if not path.is_dir():
        raise FileNotFoundError(path)
    return path


def load_counts(path):
    rows = OrderedDict()
    with path.open(newline="") as handle:
        for row in csv.DictReader(handle):
            rows[row["Image"]] = {name: int(row[name]) for name in CLASSES}
    return rows


def sum_counts(rows, names=None):
    if names is None:
        selected = rows.values()
    else:
        missing = sorted(set(names) - set(rows))
        if missing:
            raise RuntimeError("Missing {} matched patch IDs in {}".format(len(missing), rows))
        selected = (rows[name] for name in names)
    result = {name: 0 for name in CLASSES}
    for row in selected:
        for name in CLASSES:
            result[name] += row[name]
    return result


def frequencies(counts):
    total = sum(counts.values())
    return {name: counts[name] / total if total else float("nan") for name in CLASSES}


def js_divergence(p, q):
    m = {name: 0.5 * (p[name] + q[name]) for name in CLASSES}

    def kl(a, b):
        return sum(a[name] * math.log(a[name] / b[name], 2) for name in CLASSES if a[name] > 0)

    return 0.5 * kl(p, m) + 0.5 * kl(q, m)


def total_variation(p, q):
    return 0.5 * sum(abs(p[name] - q[name]) for name in CLASSES)


def master_sources():
    sources = {}
    with MASTER_CSV.open(newline="") as handle:
        for row in csv.DictReader(handle):
            if (
                row["protocol"] == "composition_matched_generalist_vs_specialist"
                and row["backbone"] == "CellViT-SAM-H"
                and row["seed"] == "42"
            ):
                key = (row["tissue"], row["fold"], row["method"])
                sources[key] = REPO / row["source_inference_json"]
    expected = {(t, f, m) for t in TISSUES for f in FOLDS for m in MODELS}
    if set(sources) != expected:
        raise RuntimeError("Unexpected master source matrix: {}".format(sorted(set(sources) ^ expected)))
    return sources


def aggregate_inference(path, matched_names):
    payload = json.loads(path.read_text())
    metrics = payload["image_metrics"]
    missing = sorted(set(matched_names) - set(metrics))
    if missing:
        raise RuntimeError("{} lacks {} matched images".format(path, len(missing)))
    images = [metrics[name] for name in sorted(matched_names)]
    confusion = [[0 for _ in ALL_CLASSES] for _ in ALL_CLASSES]
    unpaired_true = [0 for _ in ALL_CLASSES]
    unpaired_pred = [0 for _ in ALL_CLASSES]
    for image in images:
        stats = image["detection_stats"]
        for i, row in enumerate(stats["paired_confusion"]):
            for j, value in enumerate(row):
                confusion[i][j] += int(value)
        for i, value in enumerate(stats["unpaired_true_counts"]):
            unpaired_true[i] += int(value)
        for i, value in enumerate(stats["unpaired_pred_counts"]):
            unpaired_pred[i] += int(value)

    paired = sum(sum(row) for row in confusion)
    fn = sum(unpaired_true)
    fp = sum(unpaired_pred)
    f1det = 2.0 * paired / (2.0 * paired + fp + fn) if paired or fp or fn else float("nan")
    per_class = {}
    macro = []
    for idx, name in enumerate(CLASSES, start=1):
        matched_support = sum(confusion[idx][1:])
        predicted_support = sum(confusion[row][idx] for row in range(1, len(ALL_CLASSES)))
        true_positive = confusion[idx][idx]
        denominator = matched_support + predicted_support
        f1 = 2.0 * true_positive / denominator if denominator else float("nan")
        if matched_support:
            macro.append(f1)
        per_class[name] = {
            "matched_true_support": matched_support,
            "total_true_support": matched_support + unpaired_true[idx],
            "f1": f1,
        }
    predicted = {}
    foreground_pred_total = 0
    for idx, name in enumerate(CLASSES, start=1):
        count = sum(confusion[row][idx] for row in range(1, len(ALL_CLASSES))) + unpaired_pred[idx]
        predicted[name] = count
        foreground_pred_total += count
    predicted_frequency = {
        name: predicted[name] / foreground_pred_total if foreground_pred_total else float("nan")
        for name in CLASSES
    }
    return {
        "n_images": len(images),
        "Dice": finite_mean([row.get("Dice") for row in images]),
        "Jaccard": finite_mean([row.get("Jaccard") for row in images]),
        "bPQ": finite_mean([row.get("bPQ") for row in images]),
        "mPQ": finite_mean([row.get("mPQ") for row in images]),
        "F1det": f1det,
        "F1type": finite_mean(macro),
        "confusion": confusion,
        "unpaired_true": unpaired_true,
        "unpaired_pred": unpaired_pred,
        "per_class": per_class,
        "predicted": predicted,
        "predicted_frequency": predicted_frequency,
        "background_or_untyped_predictions": sum(confusion[row][0] for row in range(len(ALL_CLASSES)))
        + unpaired_pred[0],
    }


FLOAT = r"[-+]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][-+]?\d+)?"


def extract_value(line, label):
    match = re.search(re.escape(label) + r":\s*(" + FLOAT + r")", line)
    return float(match.group(1)) if match else float("nan")


def training_curves(log_path):
    curves = OrderedDict()
    current = None
    for line in log_path.read_text(errors="replace").splitlines():
        epoch_match = re.search(r"Epoch:\s*(\d+)/10", line)
        if epoch_match:
            current = int(epoch_match.group(1))
            curves.setdefault(current, {"epoch": current})
        if current is None:
            continue
        if "Training epoch stats:" in line:
            curves[current].update(
                train_loss=extract_value(line, "Loss"),
                train_Dice=extract_value(line, "Binary-Cell-Dice"),
                train_Jaccard=extract_value(line, "Binary-Cell-Jacard"),
            )
        elif "Validation epoch stats:" in line:
            curves[current].update(
                val_loss=extract_value(line, "Loss"),
                val_Dice=extract_value(line, "Binary-Cell-Dice"),
                val_Jaccard=extract_value(line, "Binary-Cell-Jacard"),
                val_bPQ=extract_value(line, "bPQ-Score"),
                val_mPQ=extract_value(line, "mPQ-Score"),
            )
    if sorted(curves) != list(range(1, 11)):
        raise RuntimeError("Incomplete epoch curve in {}: {}".format(log_path, sorted(curves)))
    return list(curves.values())


def patch_count(path):
    with path.open(newline="") as handle:
        return sum(1 for _ in handle) - 1


def build_records():
    sources = master_sources()
    records = {}
    for tissue in TISSUES:
        for fold in FOLDS:
            general_path = sources[(tissue, fold, MODELS[0])]
            specialist_path = sources[(tissue, fold, MODELS[1])]
            general_json = json.loads(general_path.read_text())
            specialist_json = json.loads(specialist_path.read_text())
            matched_names = sorted(set(general_json["image_metrics"]) & set(specialist_json["image_metrics"]))
            if not matched_names:
                raise RuntimeError("Empty intersection for {} Fold {}".format(tissue, fold))

            for model, source in [(MODELS[0], general_path), (MODELS[1], specialist_path)]:
                is_generalist = model == MODELS[0]
                dataset = data_dir(tissue, fold, is_generalist)
                train_rows = load_counts(dataset / "cell_count_train.csv")
                val_rows = load_counts(dataset / "cell_count_valid.csv")
                test_rows = load_counts(dataset / "cell_count_test.csv")
                train_counts = sum_counts(train_rows)
                val_counts = sum_counts(val_rows)
                test_counts = sum_counts(test_rows, matched_names)
                train_freq = frequencies(train_counts)
                test_freq = frequencies(test_counts)
                inference = aggregate_inference(source, matched_names)
                records[(tissue, fold, model)] = {
                    "source": source,
                    "log": source.parent / "logs.log",
                    "dataset": dataset,
                    "train_patches": len(train_rows),
                    "val_patches": len(val_rows),
                    "test_patches_raw": len(test_rows),
                    "matched_patches": len(matched_names),
                    "train_counts": train_counts,
                    "val_counts": val_counts,
                    "test_counts": test_counts,
                    "train_freq": train_freq,
                    "test_freq": test_freq,
                    "tvd": total_variation(train_freq, test_freq),
                    "jsd": js_divergence(train_freq, test_freq),
                    "inference": inference,
                    "curves": training_curves(source.parent / "logs.log"),
                }

            left = records[(tissue, fold, MODELS[0])]["test_counts"]
            right = records[(tissue, fold, MODELS[1])]["test_counts"]
            if left != right:
                raise RuntimeError("Matched label counts differ for {} Fold {}".format(tissue, fold))
    return records


CSV_FIELDS = [
    "record_type",
    "tissue",
    "fold",
    "model",
    "seed",
    "split",
    "epoch",
    "true_class",
    "predicted_class",
    "count",
    "frequency",
    "metric",
    "value",
    "train_patches",
    "val_patches",
    "test_patches_raw",
    "matched_test_patches",
    "source_path",
    "note",
]


def base_csv_row(tissue, fold, model, record):
    return {
        "record_type": "",
        "tissue": tissue,
        "fold": fold,
        "model": model,
        "seed": 42,
        "split": "",
        "epoch": "",
        "true_class": "",
        "predicted_class": "",
        "count": "",
        "frequency": "",
        "metric": "",
        "value": "",
        "train_patches": record["train_patches"],
        "val_patches": record["val_patches"],
        "test_patches_raw": record["test_patches_raw"],
        "matched_test_patches": record["matched_patches"],
        "source_path": str(record["source"].relative_to(REPO)),
        "note": "",
    }


def write_csv(records):
    rows = []
    for tissue in TISSUES:
        for fold in FOLDS:
            for model in MODELS:
                record = records[(tissue, fold, model)]
                base = base_csv_row(tissue, fold, model, record)
                summary = dict(base)
                summary.update(
                    record_type="dataset_summary",
                    note="Test metrics and test class counts use exact generalist/specialist patch-ID intersection",
                )
                rows.append(summary)
                for split, counts, freqs in [
                    ("train", record["train_counts"], record["train_freq"]),
                    ("validation", record["val_counts"], frequencies(record["val_counts"])),
                    ("matched_test", record["test_counts"], record["test_freq"]),
                ]:
                    for name in CLASSES:
                        row = dict(base)
                        row.update(
                            record_type="class_distribution",
                            split=split,
                            true_class=name,
                            count=counts[name],
                            frequency="{:.12g}".format(freqs[name]),
                            note="Patch-level nucleus annotation occurrences; overlapping patches may repeat a biological cell",
                        )
                        rows.append(row)
                for metric in ["Dice", "Jaccard", "bPQ", "mPQ", "F1det", "F1type"]:
                    row = dict(base)
                    row.update(
                        record_type="matched_test_metric",
                        split="matched_test",
                        metric=metric,
                        value="{:.12g}".format(record["inference"][metric]),
                    )
                    rows.append(row)
                for name in CLASSES:
                    class_metric = record["inference"]["per_class"][name]
                    for metric, value in [
                        ("matched_true_support", class_metric["matched_true_support"]),
                        ("total_true_support", class_metric["total_true_support"]),
                        ("paired_type_F1", class_metric["f1"]),
                    ]:
                        row = dict(base)
                        row.update(
                            record_type="matched_class_metric",
                            split="matched_test",
                            true_class=name,
                            metric=metric,
                            value="{:.12g}".format(value) if isinstance(value, float) else value,
                            note="Type F1 is calculated from paired foreground confusion only",
                        )
                        rows.append(row)
                    row = dict(base)
                    row.update(
                        record_type="predicted_class_frequency",
                        split="matched_test",
                        predicted_class=name,
                        count=record["inference"]["predicted"][name],
                        frequency="{:.12g}".format(record["inference"]["predicted_frequency"][name]),
                        note="Foreground predicted instances; unmatched predictions included",
                    )
                    rows.append(row)
                for i, true_name in enumerate(ALL_CLASSES):
                    for j, pred_name in enumerate(ALL_CLASSES):
                        row = dict(base)
                        row.update(
                            record_type="paired_confusion",
                            split="matched_test",
                            true_class=true_name,
                            predicted_class=pred_name,
                            count=record["inference"]["confusion"][i][j],
                            note="Rows=true, columns=predicted; exact matched patch intersection",
                        )
                        rows.append(row)
                for metric, value in [("total_variation", record["tvd"]), ("jensen_shannon_bits", record["jsd"])]:
                    row = dict(base)
                    row.update(
                        record_type="distribution_shift_summary",
                        split="train_to_matched_test",
                        metric=metric,
                        value="{:.12g}".format(value),
                    )
                    rows.append(row)
                for name in CLASSES:
                    train_value = record["train_freq"][name]
                    test_value = record["test_freq"][name]
                    ratio = train_value / test_value if test_value else float("nan")
                    for metric, value in [
                        ("frequency_delta_test_minus_train", test_value - train_value),
                        ("train_to_test_frequency_ratio", ratio),
                    ]:
                        row = dict(base)
                        row.update(
                            record_type="distribution_shift_class",
                            split="train_to_matched_test",
                            true_class=name,
                            metric=metric,
                            value="{:.12g}".format(value),
                        )
                        rows.append(row)
                for curve in record["curves"]:
                    for metric in [
                        "train_loss",
                        "train_Dice",
                        "train_Jaccard",
                        "val_loss",
                        "val_Dice",
                        "val_Jaccard",
                        "val_bPQ",
                        "val_mPQ",
                    ]:
                        row = dict(base)
                        row.update(
                            record_type="training_curve",
                            split="train" if metric.startswith("train_") else "validation",
                            epoch=curve["epoch"],
                            metric=metric,
                            value="{:.12g}".format(curve.get(metric, float("nan"))),
                            source_path=str(record["log"].relative_to(REPO)),
                            note="Validation curve is descriptive only and is never used as canonical TEST evidence",
                        )
                        rows.append(row)
    with OUT_CSV.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=CSV_FIELDS)
        writer.writeheader()
        writer.writerows(rows)


def confusion_markdown(matrix):
    lines = ["| True \\ Pred | " + " | ".join(ALL_CLASSES) + " |", "|---|" + "---:|" * len(ALL_CLASSES)]
    for name, row in zip(ALL_CLASSES, matrix):
        lines.append("| {} | {} |".format(name, " | ".join("{:,}".format(v) for v in row)))
    return lines


def underrepresentation(record):
    findings = []
    for name in CLASSES:
        count = record["train_counts"][name]
        train_f = record["train_freq"][name]
        test_f = record["test_freq"][name]
        if count == 0:
            findings.append("{} absent".format(name))
        elif train_f < 0.01:
            findings.append("{} rare ({})".format(name, pct(train_f)))
        if test_f >= 0.01 and train_f < 0.25 * test_f:
            findings.append(
                "{} severely underrepresented (train {}, test {})".format(name, pct(train_f), pct(test_f))
            )
    return findings


def write_markdown(records):
    lines = [
        "# Tissue-specialist failure analysis — canonical seed42 evidence",
        "",
        "> Canonical policy: checkpoint-10 TEST inference (or exact verified adapter reconstruction) only. Validation curves are shown diagnostically and are never substituted for TEST evidence.",
        "",
        "This report reconstructs the exact patch-ID intersections used in `reports/workshop_master_results.csv` for CellViT-SAM-H Selected PEFT seed42. Cell counts below are **patch-level nucleus annotation occurrences**, not unique biological cells: STHELAR patches overlap, so one biological cell can occur in more than one patch. Fold A and Fold B remain reciprocal held-out-slide directions and are not pooled.",
        "",
        "### Biological-independence boundary",
        "",
        "The local STHELAR README states that the resource contains 27 FFPE slides from 20 cancer patients, but neither that README, `cell_metadata/index.csv`, the per-slide cell metadata, nor the prepared fold manifests provide a slide-to-patient/case/donor/specimen mapping. The `*_s0` and `*_s1` identifiers therefore establish different slides only; they do **not** establish different patients. These experiments support ‘slide-independent’, ‘held-out slide’, or ‘complete-slide holdout’ wording, not patient-independent or specimen-independent wording.",
        "",
        "| Protocol | Fold | Train/validation source slide(s) | Held-out TEST slide(s) |",
        "|---|:---:|---|---|",
        "| KLT generalist | A | `kidney_s0`, `liver_s0`, `tonsil_s0` | `kidney_s1`, `liver_s1`, `tonsil_s1` |",
        "| KLT generalist | B | `kidney_s1`, `liver_s1`, `tonsil_s1` | `kidney_s0`, `liver_s0`, `tonsil_s0` |",
        "| Kidney specialist | A / B | `kidney_s0` / `kidney_s1` | `kidney_s1` / `kidney_s0` |",
        "| Liver specialist | A / B | `liver_s0` / `liver_s1` | `liver_s1` / `liver_s0` |",
        "| Tonsil specialist | A / B | `tonsil_s0` / `tonsil_s1` | `tonsil_s1` / `tonsil_s0` |",
        "",
        "Source: prepared `split_manifest.yaml` files under `../Datasets/cellvit_ready/sthelar40x_{klt,kidney,liver,tonsil}_5class_slideind_fold*`. Train and validation are spatial partitions of the listed training slide(s); TEST is the complete reciprocal slide payload.",
        "",
        "## Summary",
        "",
        "| Tissue | Fold | Model | Train / val / raw test patches | Matched test | Train nuclei | TVD train→test | JSD (bits) | bPQ | mPQ | F1det | F1type |",
        "|---|:---:|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for tissue in TISSUES:
        for fold in FOLDS:
            for model in MODELS:
                r = records[(tissue, fold, model)]
                i = r["inference"]
                lines.append(
                    "| {} | {} | {} | {:,} / {:,} / {:,} | {:,} | {:,} | {} | {} | {} | {} | {} | {} |".format(
                        tissue,
                        fold,
                        "Generalist" if model == MODELS[0] else "Specialist",
                        r["train_patches"],
                        r["val_patches"],
                        r["test_patches_raw"],
                        r["matched_patches"],
                        sum(r["train_counts"].values()),
                        fmt(r["tvd"], 3),
                        fmt(r["jsd"], 3),
                        fmt(i["bPQ"]),
                        fmt(i["mPQ"]),
                        fmt(i["F1det"]),
                        fmt(i["F1type"]),
                    )
                )

    lines.extend(
        [
            "",
            "TVD is total-variation distance on the five foreground-class frequency vectors (0 means identical, 1 means disjoint). JSD is Jensen–Shannon divergence in bits. Test distributions always use the exact matched patch intersection.",
            "",
        ]
    )

    for tissue in TISSUES:
        for fold in FOLDS:
            lines.extend(["## {} — Fold {}".format(tissue, fold), ""])
            general = records[(tissue, fold, MODELS[0])]
            specialist = records[(tissue, fold, MODELS[1])]
            lines.extend(
                [
                    "Exact matched population: **{:,} patches**. Generalist raw test coverage: {:,}; specialist raw test coverage: {:,}.".format(
                        general["matched_patches"], general["test_patches_raw"], specialist["test_patches_raw"]
                    ),
                    "",
                    "### Data volume and class distribution",
                    "",
                    "| Model | Split | Patches | " + " | ".join(CLASSES) + " |",
                    "|---|---|---:|" + "---:|" * len(CLASSES),
                ]
            )
            for model, r in [("Generalist", general), ("Specialist", specialist)]:
                for split, patches, counts, freqs in [
                    ("Train", r["train_patches"], r["train_counts"], r["train_freq"]),
                    ("Validation", r["val_patches"], r["val_counts"], frequencies(r["val_counts"])),
                    ("Matched TEST", r["matched_patches"], r["test_counts"], r["test_freq"]),
                ]:
                    cells = ["{:,} ({})".format(counts[name], pct(freqs[name])) for name in CLASSES]
                    lines.append("| {} | {} | {:,} | {} |".format(model, split, patches, " | ".join(cells)))
            lines.extend(
                [
                    "",
                    "| Model | TVD | JSD bits | Specialist/generalist train-patch ratio | Absent, rare, or severely underrepresented training classes |",
                    "|---|---:|---:|---:|---|",
                ]
            )
            ratio = specialist["train_patches"] / general["train_patches"]
            for model, r in [("Generalist", general), ("Specialist", specialist)]:
                findings = underrepresentation(r)
                lines.append(
                    "| {} | {} | {} | {} | {} |".format(
                        model,
                        fmt(r["tvd"], 3),
                        fmt(r["jsd"], 3),
                        fmt(ratio, 3) if model == "Specialist" else "1.000",
                        "; ".join(findings) if findings else "none by declared thresholds",
                    )
                )
            lines.extend(
                [
                    "",
                    "Declared thresholds: rare = <1% of training annotation occurrences; severe underrepresentation = training frequency <25% of matched-test frequency when the test frequency is at least 1%.",
                    "",
                    "### Canonical matched TEST metrics and paired type performance",
                    "",
                    "| Model | Dice | bPQ | mPQ | F1det | F1type |",
                    "|---|---:|---:|---:|---:|---:|",
                ]
            )
            for model, r in [("Generalist", general), ("Specialist", specialist)]:
                i = r["inference"]
                lines.append(
                    "| {} | {} | {} | {} | {} | {} |".format(
                        model,
                        fmt(i["Dice"]),
                        fmt(i["bPQ"]),
                        fmt(i["mPQ"]),
                        fmt(i["F1det"]),
                        fmt(i["F1type"]),
                    )
                )
            lines.extend(
                [
                    "",
                    "| Model | Class | Matched true support | Total true support | Paired type F1 | Predicted count | Predicted frequency |",
                    "|---|---|---:|---:|---:|---:|---:|",
                ]
            )
            for model, r in [("Generalist", general), ("Specialist", specialist)]:
                i = r["inference"]
                for name in CLASSES:
                    c = i["per_class"][name]
                    lines.append(
                        "| {} | {} | {:,} | {:,} | {} | {:,} | {} |".format(
                            model,
                            name,
                            c["matched_true_support"],
                            c["total_true_support"],
                            fmt(c["f1"]),
                            i["predicted"][name],
                            pct(i["predicted_frequency"][name]),
                        )
                    )
            lines.append("")
            for model, r in [("Generalist", general), ("Specialist", specialist)]:
                lines.extend(
                    [
                        "#### {} full paired confusion matrix".format(model),
                        "",
                        "Rows are true labels and columns predicted labels. Background is retained because the saved canonical matrix is 6×6; foreground type F1 excludes its row and column.",
                        "",
                    ]
                )
                lines.extend(confusion_markdown(r["inference"]["confusion"]))
                lines.extend(
                    [
                        "",
                        "Unpaired true counts: `{}`. Unpaired predicted counts: `{}`. Background/untyped predicted instances: `{:,}`.".format(
                            dict(zip(ALL_CLASSES, r["inference"]["unpaired_true"])),
                            dict(zip(ALL_CLASSES, r["inference"]["unpaired_pred"])),
                            r["inference"]["background_or_untyped_predictions"],
                        ),
                        "",
                    ]
                )
            lines.extend(
                [
                    "### Training and validation curves",
                    "",
                    "Validation values below are diagnostic curves only. Canonical comparison values above come exclusively from checkpoint-10 TEST inference.",
                    "",
                    "| Epoch | Gen train loss | Gen val loss | Gen val bPQ | Gen val mPQ | Spec train loss | Spec val loss | Spec val bPQ | Spec val mPQ |",
                    "|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
                ]
            )
            for epoch in range(1, 11):
                g = general["curves"][epoch - 1]
                s = specialist["curves"][epoch - 1]
                lines.append(
                    "| {} | {} | {} | {} | {} | {} | {} | {} | {} |".format(
                        epoch,
                        fmt(g.get("train_loss")),
                        fmt(g.get("val_loss")),
                        fmt(g.get("val_bPQ")),
                        fmt(g.get("val_mPQ")),
                        fmt(s.get("train_loss")),
                        fmt(s.get("val_loss")),
                        fmt(s.get("val_bPQ")),
                        fmt(s.get("val_mPQ")),
                    )
                )
            lines.extend(
                [
                    "",
                    "Sources: `{}`; `{}`; split metadata under `{}` and `{}`.".format(
                        general["source"].relative_to(REPO),
                        specialist["source"].relative_to(REPO),
                        general["dataset"].relative_to(DATA_ROOT.parent),
                        specialist["dataset"].relative_to(DATA_ROOT.parent),
                    ),
                    "",
                ]
            )

    lines.extend(
        [
            "## Evidence-weighted interpretation",
            "",
            "### What the seed42 evidence supports",
            "",
            "- **Training-set size is a plausible but non-isolated factor.** Every specialist has fewer total training patches than the KLT generalist, but Liver/Tonsil specialists can have more same-tissue patches than the KLT contribution from that tissue. Size is therefore confounded with tissue diversity and split-specific coverage.",
            "- **Reduced cross-tissue diversity is structurally present but not independently randomized.** The generalist sees Kidney, Liver and Tonsil; each specialist sees one slide from one tissue. These runs cannot isolate diversity from sample count, class balance, or slide identity.",
            "- **Class-distribution shift is directly measurable.** The TVD/JSD tables and class-frequency deltas identify missing/rare/underrepresented classes. Melanocyte is absent throughout these KLT tissues and is excluded from present-class F1type.",
            "- **Detection is generally more stable than typing.** Across the six directions, specialist F1det differences are smaller than several F1type/mPQ changes. This argues against describing the Kidney loss as a pure detection collapse; paired type confusion and missing/shifted class support are more directly implicated descriptively.",
            "- **Segmentation quality can still contribute.** bPQ changes are reported beside F1det and type metrics. Liver shows direction-dependent mixtures of segmentation/detection and typing behavior, so no single failure mechanism explains every fold.",
            "- **Stochastic optimization cannot be judged from seed42.** The planned seed43/44 replications are required to determine whether Kidney negativity, Liver directionality, Tonsil near-neutrality, and F1det stability persist.",
            "",
            "### What must not be claimed",
            "",
            "These descriptive data do not identify a causal mechanism and do not support statistical significance, superiority, equivalence, or non-inferiority. Patches and cell occurrences are not independent biological replicates. Fold A/B are reciprocal slide directions, not an n=2 patient cohort estimate and not seed replicates.",
            "",
            "## Machine-readable provenance",
            "",
            "All distributions, matrices, per-class metrics, predicted frequencies, shift statistics, and epoch curves are in `reports/tissue_specialist_failure_analysis.csv`. Every row includes its canonical inference or training-log source.",
            "",
        ]
    )
    OUT_MD.write_text("\n".join(lines))


def main():
    records = build_records()
    write_csv(records)
    write_markdown(records)
    print(OUT_MD.relative_to(REPO))
    print(OUT_CSV.relative_to(REPO))


if __name__ == "__main__":
    main()
