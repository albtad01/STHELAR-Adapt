#!/usr/bin/env python3
"""Generate read-only workshop slide/tissue diagnostics and seed audit.

Only the four report files are written. Experimental runs, manifests,
checkpoints, configurations, adapters, and scheduler state are read-only.

By default Part B includes only the requested seed-42 Selected PEFT rows.
The FullFT discovery specifications use the same aggregation path and schema;
pass --include-fullft after the matched campaign is ready to add those rows
without changing this analysis.
"""

import argparse
import csv
import hashlib
import json
import math
import re
import statistics
from collections import defaultdict
from pathlib import Path

import yaml


REPO = Path(__file__).resolve().parents[2]
DATA_ROOT = REPO.parent / "Datasets" / "cellvit_ready"
MANIFEST_ROOT = REPO / "manifests" / "slide_independent"
REPORT_ROOT = REPO / "reports"

CLASSES = ["Immune", "Stromal", "Epithelial", "Melanocyte", "Other"]
ALL_CLASSES = ["Background"] + CLASSES
TISSUES = ["Breast", "Colon", "Kidney", "Liver", "Lung", "Ovary", "Pancreatic", "Skin", "Tonsil"]
KLT_TISSUES = ["Kidney", "Liver", "Tonsil"]
FOLDS = ["A", "B"]

DIAG_MD = REPORT_ROOT / "slide_tissue_diagnostics.md"
DIAG_CSV = REPORT_ROOT / "slide_tissue_diagnostics.csv"
AUDIT_MD = REPORT_ROOT / "workshop_ablation_seed_audit.md"
AUDIT_CSV = REPORT_ROOT / "workshop_ablation_seed_audit.csv"


def rel(path):
    try:
        return str(Path(path).resolve().relative_to(REPO.resolve()))
    except ValueError:
        return str(path)


def fmt(value, digits=4):
    if value is None or not isinstance(value, (int, float)) or not math.isfinite(value):
        return "--"
    return ("{:,." + str(digits) + "f}").format(value)


def pct(value, digits=2):
    return fmt(100.0 * value, digits) + "%" if value is not None and math.isfinite(value) else "--"


def mean(values):
    values = [float(v) for v in values if v is not None and math.isfinite(float(v))]
    return sum(values) / len(values) if values else float("nan")


def sample_sd(values):
    values = [float(v) for v in values if v is not None and math.isfinite(float(v))]
    return statistics.stdev(values) if len(values) >= 2 else float("nan")


def frequencies(counts):
    total = sum(counts.values())
    return {name: counts[name] / total if total else float("nan") for name in CLASSES}


def entropy_bits(counts):
    return -sum(p * math.log(p, 2) for p in frequencies(counts).values() if p > 0)


def total_variation(p, q):
    return 0.5 * sum(abs(p[name] - q[name]) for name in CLASSES)


def js_divergence_bits(p, q):
    mid = {name: 0.5 * (p[name] + q[name]) for name in CLASSES}

    def kl(left, right):
        return sum(left[name] * math.log(left[name] / right[name], 2) for name in CLASSES if left[name] > 0)

    return 0.5 * kl(p, mid) + 0.5 * kl(q, mid)


def rare_train_present_test(train_counts, test_counts):
    train_f = frequencies(train_counts)
    findings = []
    for name in CLASSES:
        if test_counts[name] <= 0:
            continue
        if train_counts[name] == 0:
            findings.append("{} absent in train; present in test".format(name))
        elif train_f[name] < 0.01:
            findings.append("{} rare in train ({})".format(name, pct(train_f[name])))
    return findings


def tissue_manifest_path(tissue, fold):
    matches = sorted(MANIFEST_ROOT.glob(
        "sthelar40x_{}_5class_slideind_fold{}_margin128*/split_manifest.yaml".format(tissue.lower(), fold)
    ))
    if len(matches) != 1:
        raise RuntimeError("Expected one manifest for {} {}, found {}".format(tissue, fold, matches))
    return matches[0]


def klt_manifest_path(fold):
    return MANIFEST_ROOT / "sthelar40x_klt_5class_slideind_fold{}_margin128".format(fold) / "split_manifest.yaml"


def load_manifest(path):
    with Path(path).open() as handle:
        return yaml.safe_load(handle)


def data_dir_from_manifest(manifest_path):
    name = Path(manifest_path).parent.name
    path = DATA_ROOT / name
    if not path.is_dir():
        raise FileNotFoundError(path)
    return path


def load_cell_counts(path):
    rows = {}
    with Path(path).open(newline="") as handle:
        for row in csv.DictReader(handle):
            rows[row["Image"]] = {name: int(row[name]) for name in CLASSES}
    return rows


def slide_id_from_patch(name):
    return name.split("__", 1)[0]


def group_counts_by_slide(rows):
    grouped = defaultdict(lambda: {name: 0 for name in CLASSES})
    patches = defaultdict(int)
    for patch, counts in rows.items():
        slide = slide_id_from_patch(patch)
        patches[slide] += 1
        for name in CLASSES:
            grouped[slide][name] += counts[name]
    return dict(grouped), dict(patches)


def composition_record(scope, tissue, fold, split, slide_id, patch_count, counts, source):
    freqs = frequencies(counts)
    record = {
        "record_type": "slide_composition",
        "scope": scope,
        "tissue": tissue,
        "fold": fold,
        "split": split,
        "slide_id": slide_id,
        "patch_count": patch_count,
        "annotated_cell_count": sum(counts.values()),
        "nuclei_per_patch": sum(counts.values()) / patch_count if patch_count else float("nan"),
        "class_entropy_bits": entropy_bits(counts),
        "source_path": rel(source),
        "note": "Counts are patch-level annotation occurrences; overlapping patches can repeat a biological cell.",
    }
    for name in CLASSES:
        record[name.lower() + "_count"] = counts[name]
        record[name.lower() + "_frequency"] = freqs[name]
    return record


def build_compositions():
    records = []
    shifts = []

    # KLT pooled and per-slide composition uses the exact KLT train/test CSVs.
    for fold in FOLDS:
        manifest_path = klt_manifest_path(fold)
        manifest = load_manifest(manifest_path)
        dataset = data_dir_from_manifest(manifest_path)
        per_split = {}
        for split in ["train", "test"]:
            count_source = dataset / "cell_count_{}.csv".format(split)
            rows = load_cell_counts(count_source)
            grouped, patches = group_counts_by_slide(rows)
            per_split[split] = (grouped, patches)
            for slide in sorted(grouped):
                tissue = slide.rsplit("_", 1)[0].title()
                records.append(composition_record(
                    "KLT", tissue, fold, split, slide, patches[slide], grouped[slide], count_source
                ))
            pooled_counts = {name: sum(grouped[s][name] for s in grouped) for name in CLASSES}
            records.append(composition_record(
                "KLT", "Pooled KLT", fold, split, "+".join(sorted(grouped)), len(rows), pooled_counts, count_source
            ))

        train_grouped, _ = per_split["train"]
        test_grouped, _ = per_split["test"]
        for tissue in KLT_TISSUES + ["Pooled KLT"]:
            if tissue == "Pooled KLT":
                train_counts = {name: sum(x[name] for x in train_grouped.values()) for name in CLASSES}
                test_counts = {name: sum(x[name] for x in test_grouped.values()) for name in CLASSES}
            else:
                train_slide = next(s for s in train_grouped if s.startswith(tissue.lower() + "_"))
                test_slide = next(s for s in test_grouped if s.startswith(tissue.lower() + "_"))
                train_counts = train_grouped[train_slide]
                test_counts = test_grouped[test_slide]
            p, q = frequencies(train_counts), frequencies(test_counts)
            shifts.append({
                "record_type": "distribution_shift",
                "scope": "KLT",
                "tissue": tissue,
                "fold": fold,
                "split": "train_to_test",
                "total_variation": total_variation(p, q),
                "jensen_shannon_bits": js_divergence_bits(p, q),
                "rare_or_absent_train_present_test": "; ".join(rare_train_present_test(train_counts, test_counts)) or "none",
                "source_path": "{}; {}".format(rel(dataset / "cell_count_train.csv"), rel(dataset / "cell_count_test.csv")),
                "note": "Descriptive distribution distances; no significance test.",
            })

    # Tissue-specific manifests are the composition authority for Part B.
    tissue_records = []
    tissue_shifts = []
    for tissue in TISSUES:
        for fold in FOLDS:
            path = tissue_manifest_path(tissue, fold)
            manifest = load_manifest(path)
            pair = {}
            for split in ["train", "test"]:
                counts = {name: int(manifest["class_counts_per_split"][split][name]) for name in CLASSES}
                slide = manifest["train_slide_ids" if split == "train" else "test_slide_ids"][0]
                patches = int(manifest["patch_counts"][split])
                pair[split] = counts
                tissue_records.append(composition_record(
                    "tissue-specific", tissue, fold, split, slide, patches, counts, path
                ))
            p, q = frequencies(pair["train"]), frequencies(pair["test"])
            tissue_shifts.append({
                "record_type": "distribution_shift",
                "scope": "tissue-specific",
                "tissue": tissue,
                "fold": fold,
                "split": "train_to_test",
                "total_variation": total_variation(p, q),
                "jensen_shannon_bits": js_divergence_bits(p, q),
                "rare_or_absent_train_present_test": "; ".join(rare_train_present_test(pair["train"], pair["test"])) or "none",
                "source_path": rel(path),
                "note": "Descriptive distribution distances; no significance test.",
            })
    return records, shifts, tissue_records, tissue_shifts


def aggregate_inference(path, slide_prefix=None):
    with Path(path).open() as handle:
        payload = json.load(handle)
    dataset = payload["dataset"]
    image_items = list(payload["image_metrics"].items())
    if slide_prefix is not None:
        image_items = [(name, item) for name, item in image_items if name.startswith(slide_prefix + "__")]
        if not image_items:
            raise RuntimeError("No images for slide prefix {} in {}".format(slide_prefix, path))
    confusion = [[0 for _ in ALL_CLASSES] for _ in ALL_CLASSES]
    unpaired_true = [0 for _ in ALL_CLASSES]
    unpaired_pred = [0 for _ in ALL_CLASSES]
    for _, image in image_items:
        stats = image["detection_stats"]
        for i, row in enumerate(stats["paired_confusion"]):
            for j, value in enumerate(row):
                confusion[i][j] += int(value)
        for i, value in enumerate(stats["unpaired_true_counts"]):
            unpaired_true[i] += int(value)
        for i, value in enumerate(stats["unpaired_pred_counts"]):
            unpaired_pred[i] += int(value)

    per_class = {}
    f1_values = []
    predicted = {}
    for idx, name in enumerate(CLASSES, 1):
        matched_true = sum(confusion[idx][1:])
        matched_pred = sum(confusion[row][idx] for row in range(1, len(ALL_CLASSES)))
        tp = confusion[idx][idx]
        f1 = 2.0 * tp / (matched_true + matched_pred) if matched_true + matched_pred else float("nan")
        if matched_true:
            f1_values.append(f1)
        pred_count = matched_pred + unpaired_pred[idx]
        predicted[name] = pred_count
        per_class[name] = {
            "f1": f1,
            "matched_support": matched_true,
            "total_true_support": matched_true + unpaired_true[idx],
            "predicted_count": pred_count,
        }
    pred_total = sum(predicted.values())
    for name in CLASSES:
        per_class[name]["predicted_frequency"] = predicted[name] / pred_total if pred_total else float("nan")

    offdiag = []
    for i, true_name in enumerate(CLASSES, 1):
        for j, pred_name in enumerate(CLASSES, 1):
            if i != j:
                offdiag.append((confusion[i][j], true_name, pred_name))
    offdiag.sort(reverse=True)
    paired = sum(sum(row) for row in confusion)
    false_negative = sum(unpaired_true)
    false_positive = sum(unpaired_pred)
    subset_f1det = 2.0 * paired / (2.0 * paired + false_positive + false_negative)
    use_dataset = slide_prefix is None
    return {
        "n_images": len(image_items),
        "Dice": float(dataset["Binary-Cell-Dice-Mean"]) if use_dataset else mean([x.get("Dice") for _, x in image_items]),
        "Jaccard": float(dataset["Binary-Cell-Jacard-Mean"]) if use_dataset else mean([x.get("Jaccard") for _, x in image_items]),
        "bPQ": float(dataset["bPQ"]) if use_dataset else mean([x.get("bPQ") for _, x in image_items]),
        "mPQ": float(dataset["mPQ"]) if use_dataset else mean([x.get("mPQ") for _, x in image_items]),
        "F1det": float(dataset["f1_detection"]) if use_dataset else subset_f1det,
        "F1type": mean(f1_values),
        "confusion": confusion,
        "unpaired_true": unpaired_true,
        "unpaired_pred": unpaired_pred,
        "per_class": per_class,
        "dominant_confusion_count": offdiag[0][0],
        "dominant_confusion_true": offdiag[0][1],
        "dominant_confusion_pred": offdiag[0][2],
    }


def verify_checkpoint10(json_path, expected_patches=None):
    log_path = Path(json_path).with_name("inference.log")
    if not log_path.is_file():
        return False, "missing inference.log"
    text = log_path.read_text(errors="replace")
    if not re.search(r"checkpoint_10\.pth", text):
        return False, "inference.log does not resolve checkpoint_10.pth"
    if expected_patches is not None:
        with Path(json_path).open() as handle:
            n_images = len(json.load(handle).get("image_metrics", {}))
        if n_images != expected_patches:
            return False, "TEST coverage {}/{}".format(n_images, expected_patches)
    return True, "checkpoint_10 TEST"


def load_master_klt_sources():
    sources = {}
    with (REPORT_ROOT / "workshop_master_results.csv").open(newline="") as handle:
        for row in csv.DictReader(handle):
            if (row["protocol"] == "held_out_slide_KLT" and row["backbone"] == "CellViT-SAM-H"
                    and row["method"] in ("Selected PEFT", "FullFT") and row["seed"] in ("42", "43", "44")):
                sources[(row["method"], row["fold"], int(row["seed"]))] = REPO / row["source_inference_json"]
    expected = {(m, f, s) for m in ("Selected PEFT", "FullFT") for f in FOLDS for s in (42, 43, 44)}
    if set(sources) != expected:
        raise RuntimeError("KLT source matrix mismatch: {}".format(sorted(set(sources) ^ expected)))
    return sources


def build_klt_results():
    sources = load_master_klt_sources()
    results = {}
    rows = []
    scopes = KLT_TISSUES + ["Pooled KLT"]
    for key in sorted(sources):
        method, fold, seed = key
        path = sources[key]
        expected = load_manifest(klt_manifest_path(fold))["patch_counts"]["test"]
        ok, reason = verify_checkpoint10(path, expected)
        if not ok:
            raise RuntimeError("Noncanonical KLT source {}: {}".format(path, reason))
        test_slides = load_manifest(klt_manifest_path(fold))["test_slide_ids"]
        for test_scope in scopes:
            prefix = None if test_scope == "Pooled KLT" else next(
                slide for slide in test_slides if slide.startswith(test_scope.lower() + "_")
            )
            agg = aggregate_inference(path, prefix)
            results[(method, fold, seed, test_scope)] = agg
            base = {
                "scope": "KLT",
                "tissue": test_scope,
                "backbone": "CellViT-SAM-H",
                "method": method,
                "fold": fold,
                "seed": seed,
                "source_path": rel(path),
                "checkpoint_policy": reason,
            }
            metric_row = dict(base, record_type="klt_seed_metric", patch_count=agg["n_images"])
            metric_row.update({name: agg[name] for name in ["Dice", "Jaccard", "bPQ", "mPQ", "F1det", "F1type"]})
            rows.append(metric_row)
            for name in CLASSES:
                item = agg["per_class"][name]
                rows.append(dict(base, record_type="klt_per_class", class_name=name,
                                 class_f1=item["f1"], matched_support=item["matched_support"],
                                 total_true_support=item["total_true_support"],
                                 predicted_count=item["predicted_count"],
                                 predicted_frequency=item["predicted_frequency"]))
            for i, true_name in enumerate(ALL_CLASSES):
                for j, pred_name in enumerate(ALL_CLASSES):
                    rows.append(dict(base, record_type="klt_confusion", true_class=true_name,
                                     predicted_class=pred_name, count=agg["confusion"][i][j],
                                     note="Rows=true, columns=predicted; paired matches only."))

    summary = {}
    for test_scope in scopes:
        for method in ("Selected PEFT", "FullFT"):
            for fold in FOLDS:
                vals = [results[(method, fold, seed, test_scope)] for seed in (42, 43, 44)]
                item = {}
                for metric in ["Dice", "Jaccard", "bPQ", "mPQ", "F1det", "F1type"]:
                    item[metric + "_mean"] = mean([x[metric] for x in vals])
                    item[metric + "_sd"] = sample_sd([x[metric] for x in vals])
                summary[(method, fold, test_scope)] = item
                rows.append(dict({"record_type": "klt_fold_summary", "scope": "KLT", "tissue": test_scope,
                                  "backbone": "CellViT-SAM-H", "method": method, "fold": fold,
                                  "seed": "42;43;44"}, **item))
                for cls in CLASSES:
                    fs = [x["per_class"][cls]["f1"] for x in vals]
                    rows.append({"record_type": "klt_per_class_fold_summary", "scope": "KLT", "tissue": test_scope,
                                 "backbone": "CellViT-SAM-H", "method": method, "fold": fold,
                                 "seed": "42;43;44", "class_name": cls, "class_f1_mean": mean(fs),
                                 "class_f1_sd": sample_sd(fs)})
            diff = {metric + "_difference_B_minus_A": summary[(method, "B", test_scope)][metric + "_mean"] - summary[(method, "A", test_scope)][metric + "_mean"]
                    for metric in ["Dice", "Jaccard", "bPQ", "mPQ", "F1det", "F1type"]}
            rows.append(dict({"record_type": "klt_fold_difference", "scope": "KLT", "tissue": test_scope,
                              "backbone": "CellViT-SAM-H", "method": method, "fold": "B-A",
                              "seed": "42;43;44"}, **diff))
    return results, summary, rows


TISSUE_SPECS = [
    ("CellViT-SAM-H", "Selected PEFT", "sthelar40x_{t}_5class_slideind_fold{f}_lora_adaptformer_r8_a8_red16_heads_e10_seed42"),
    ("CellViT-256", "Selected PEFT", "sthelar40x_{t}_5class_slideind_fold{f}_cellvit256_lora_adaptformer_r8_a8_red16_heads_e10_seed42"),
]
FULLFT_SPECS = [
    ("CellViT-SAM-H", "FullFT", "sthelar40x_{t}_5class_slideind_fold{f}_fullft_lr1e-5_e10_seed42"),
    ("CellViT-256", "FullFT", "sthelar40x_{t}_5class_slideind_fold{f}_cellvit256_fullft_lr1e-5_e10_seed42"),
]


def discover_tissue_source(template, tissue, fold):
    run_dir = REPO / "run" / template.format(t=tissue.lower(), f=fold)
    candidates = sorted(run_dir.glob("log/*/inference_results.json"))
    if not candidates:
        return None
    # The latest complete log directory is canonical when retries exist.
    expected = load_manifest(tissue_manifest_path(tissue, fold))["patch_counts"]["test"]
    good = []
    for path in candidates:
        ok, _ = verify_checkpoint10(path, expected)
        if ok:
            good.append(path)
    return good[-1] if good else None


def ordinal_ranks(items, metric):
    ordered = sorted(items, key=lambda x: x[metric], reverse=True)
    return {id(item): rank for rank, item in enumerate(ordered, 1)}


def build_tissue_results(include_fullft=False):
    specs = list(TISSUE_SPECS) + (list(FULLFT_SPECS) if include_fullft else [])
    results = []
    missing = []
    for backbone, method, template in specs:
        for tissue in TISSUES:
            for fold in FOLDS:
                path = discover_tissue_source(template, tissue, fold)
                if path is None:
                    missing.append((backbone, method, tissue, fold))
                    continue
                agg = aggregate_inference(path)
                results.append({"backbone": backbone, "method": method, "tissue": tissue,
                                "fold": fold, "seed": 42, "source": path, **agg})

    rows = []
    for backbone, method, _ in specs:
        subset = [x for x in results if x["backbone"] == backbone and x["method"] == method]
        ranks = {metric: ordinal_ranks(subset, metric) for metric in ["bPQ", "mPQ", "F1det", "F1type"]}
        f1det_median = statistics.median([x["F1det"] for x in subset]) if subset else float("nan")
        f1type_median = statistics.median([x["F1type"] for x in subset]) if subset else float("nan")
        for item in subset:
            item["strong_detection_weak_typing"] = item["F1det"] >= f1det_median and item["F1type"] < f1type_median
            base = {"record_type": "tissue_result", "scope": "tissue-specific", "backbone": backbone,
                    "method": method, "tissue": item["tissue"], "fold": item["fold"], "seed": 42,
                    "source_path": rel(item["source"]), "checkpoint_policy": "checkpoint_10 TEST",
                    "patch_count": item["n_images"], "Dice": item["Dice"], "Jaccard": item["Jaccard"],
                    "bPQ": item["bPQ"], "mPQ": item["mPQ"], "F1det": item["F1det"], "F1type": item["F1type"],
                    "bPQ_rank": ranks["bPQ"][id(item)], "mPQ_rank": ranks["mPQ"][id(item)],
                    "F1det_rank": ranks["F1det"][id(item)], "F1type_rank": ranks["F1type"][id(item)],
                    "strong_detection_weak_typing": item["strong_detection_weak_typing"],
                    "dominant_confusion": "{}->{}".format(item["dominant_confusion_true"], item["dominant_confusion_pred"]),
                    "dominant_confusion_count": item["dominant_confusion_count"],
                    "note": "Strong/weak flag is within-backbone median-based and descriptive."}
            rows.append(base)
            for name in CLASSES:
                cls = item["per_class"][name]
                rows.append({"record_type": "tissue_per_class", "scope": "tissue-specific", "backbone": backbone,
                             "method": method, "tissue": item["tissue"], "fold": item["fold"], "seed": 42,
                             "class_name": name, "class_f1": cls["f1"], "matched_support": cls["matched_support"],
                             "total_true_support": cls["total_true_support"],
                             "predicted_count": cls["predicted_count"], "predicted_frequency": cls["predicted_frequency"],
                             "source_path": rel(item["source"])})

        # Reciprocal-fold tissue means are ranked only for complete A+B pairs.
        paired_means = []
        for tissue in TISSUES:
            pair = [x for x in subset if x["tissue"] == tissue]
            if {x["fold"] for x in pair} != {"A", "B"}:
                continue
            mean_item = {"tissue": tissue}
            for metric in ["bPQ", "mPQ", "F1det", "F1type"]:
                mean_item[metric] = mean([x[metric] for x in pair])
            paired_means.append(mean_item)
        mean_ranks = {metric: ordinal_ranks(paired_means, metric) for metric in ["bPQ", "mPQ", "F1det", "F1type"]}
        for item in paired_means:
            rows.append({"record_type": "tissue_reciprocal_fold_mean", "scope": "tissue-specific",
                         "backbone": backbone, "method": method, "tissue": item["tissue"], "fold": "A+B mean", "seed": 42,
                         **{m: item[m] for m in ["bPQ", "mPQ", "F1det", "F1type"]},
                         **{m + "_rank": mean_ranks[m][id(item)] for m in ["bPQ", "mPQ", "F1det", "F1type"]},
                         "note": "Unweighted descriptive mean of reciprocal fold directions; only complete A+B tissue pairs ranked."})

    for backbone, method, tissue, fold in missing:
        rows.append({"record_type": "tissue_missing", "scope": "tissue-specific", "backbone": backbone,
                     "method": method, "tissue": tissue, "fold": fold, "seed": 42,
                     "note": "No full-coverage checkpoint_10 TEST JSON found; excluded."})
    return results, missing, rows


DIAG_FIELDS = [
    "record_type", "scope", "backbone", "method", "tissue", "fold", "split", "slide_id", "seed",
    "class_name", "true_class", "predicted_class", "count", "patch_count", "annotated_cell_count",
    "nuclei_per_patch", "class_entropy_bits",
    "immune_count", "immune_frequency", "stromal_count", "stromal_frequency",
    "epithelial_count", "epithelial_frequency", "melanocyte_count", "melanocyte_frequency",
    "other_count", "other_frequency", "total_variation", "jensen_shannon_bits",
    "rare_or_absent_train_present_test", "Dice", "Jaccard", "bPQ", "mPQ", "F1det", "F1type",
    "Dice_mean", "Dice_sd", "Jaccard_mean", "Jaccard_sd", "bPQ_mean", "bPQ_sd", "mPQ_mean", "mPQ_sd",
    "F1det_mean", "F1det_sd", "F1type_mean", "F1type_sd",
    "Dice_difference_B_minus_A", "Jaccard_difference_B_minus_A", "bPQ_difference_B_minus_A",
    "mPQ_difference_B_minus_A", "F1det_difference_B_minus_A", "F1type_difference_B_minus_A",
    "class_f1", "class_f1_mean", "class_f1_sd", "matched_support", "total_true_support",
    "predicted_count", "predicted_frequency", "bPQ_rank", "mPQ_rank", "F1det_rank", "F1type_rank",
    "strong_detection_weak_typing", "dominant_confusion", "dominant_confusion_count",
    "checkpoint_policy", "source_path", "note",
]


def write_diag_csv(rows):
    with DIAG_CSV.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=DIAG_FIELDS, extrasaction="ignore")
        writer.writeheader()
        writer.writerows([
            {key: ("" if isinstance(value, float) and not math.isfinite(value) else value)
             for key, value in row.items()}
            for row in rows
        ])


def composition_table(lines, title, records, shifts):
    lines.extend(["### " + title, "",
                  "| Tissue/scope | Fold | Split | Slide(s) | Patches | Annotated cells | Nuclei/patch | Entropy (bits) | Immune | Stromal | Epithelial | Melanocyte | Other |",
                  "|---|:---:|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|"])
    for r in records:
        cells = []
        for name in CLASSES:
            cells.append("{:,} ({})".format(r[name.lower() + "_count"], pct(r[name.lower() + "_frequency"])))
        lines.append("| {} | {} | {} | `{}` | {:,} | {:,} | {} | {} | {} |".format(
            r["tissue"], r["fold"], r["split"].upper(), r["slide_id"], r["patch_count"],
            r["annotated_cell_count"], fmt(r["nuclei_per_patch"], 2), fmt(r["class_entropy_bits"], 3), " | ".join(cells)))
    lines.extend(["", "| Tissue/scope | Fold | TVD | JSD (bits) | Absent/rare in train but present in test |",
                  "|---|:---:|---:|---:|---|"])
    for r in shifts:
        lines.append("| {} | {} | {} | {} | {} |".format(r["tissue"], r["fold"],
                     fmt(r["total_variation"], 3), fmt(r["jensen_shannon_bits"], 3),
                     r["rare_or_absent_train_present_test"]))
    lines.append("")


def confusion_md(matrix):
    lines = ["| True \\ Pred | " + " | ".join(ALL_CLASSES) + " |", "|---|" + "---:|" * len(ALL_CLASSES)]
    for name, row in zip(ALL_CLASSES, matrix):
        lines.append("| {} | {} |".format(name, " | ".join("{:,}".format(v) for v in row)))
    return lines


def write_diag_md(klt_comps, klt_shifts, tissue_comps, tissue_shifts, klt_results, klt_summary,
                  tissue_results, tissue_missing, include_fullft=False):
    samh_complete = len([x for x in tissue_results if x["backbone"] == "CellViT-SAM-H" and x["method"] == "Selected PEFT"])
    cv256_complete = len([x for x in tissue_results if x["backbone"] == "CellViT-256" and x["method"] == "Selected PEFT"])
    missing_peft = ["{} {} {}".format(*x) for x in tissue_missing if x[1] == "Selected PEFT"]
    lines = [
        "# Workshop slide- and tissue-dependent diagnostics",
        "",
        "Generated from repository-local manifests and canonical inference artifacts on 2026-09-01. This is a read-only, descriptive scientific analysis. No run, checkpoint, configuration, adapter, or scheduler state was changed.",
        "",
        "## Scope and definitions",
        "",
        "Canonical completion requires an `inference_results.json` whose `inference.log` explicitly loads `checkpoint_10.pth` and whose `image_metrics` keys cover the complete manifest TEST set. Cell counts are **annotation occurrences in patches**, not unique cells: overlapping patches can repeat a biological cell. The train partition excludes the spatial validation partition. Entropy is Shannon entropy in bits over the five STHELAR classes (maximum log2(5)=2.322). ‘Rare’ is declared as <1% of training annotation occurrences. TVD and JSD (base 2) are descriptive distances on foreground-class frequency vectors; neither is a significance test.",
        "",
        "Slide identifiers establish held-out slides, not patient independence: the repository does not provide a slide-to-patient mapping. Folds are reciprocal directions, not biological replicates, and seeds are stochastic repeats within a fixed fold.",
        "",
        "## Part A — KLT Fold A versus Fold B",
        "",
    ]
    composition_table(lines, "KLT composition: every train and held-out slide plus pooled KLT", klt_comps, klt_shifts)

    klt_test_scopes = KLT_TISSUES + ["Pooled KLT"]
    lines.extend(["### Canonical SAM-H seed-level TEST metrics", "",
                  "| Test scope | Method | Fold | Seed | bPQ | mPQ | F1det | F1type |",
                  "|---|---|:---:|---:|---:|---:|---:|---:|"])
    for test_scope in klt_test_scopes:
        for method in ("Selected PEFT", "FullFT"):
            for fold in FOLDS:
                for seed in (42, 43, 44):
                    r = klt_results[(method, fold, seed, test_scope)]
                    lines.append("| {} | {} | {} | {} | {} | {} | {} | {} |".format(
                        test_scope, method, fold, seed, fmt(r["bPQ"]), fmt(r["mPQ"]), fmt(r["F1det"]), fmt(r["F1type"])))
    lines.extend(["", "### Within-fold seed variation and reciprocal-fold difference", "",
                  "| Test scope | Method | Fold | bPQ mean ± sample SD | mPQ mean ± sample SD | F1det mean ± sample SD | F1type mean ± sample SD |",
                  "|---|---|:---:|---:|---:|---:|---:|"])
    for test_scope in klt_test_scopes:
        for method in ("Selected PEFT", "FullFT"):
            for fold in FOLDS:
                r = klt_summary[(method, fold, test_scope)]
                lines.append("| {} | {} | {} | {} ± {} | {} ± {} | {} ± {} | {} ± {} |".format(
                    test_scope, method, fold, fmt(r["bPQ_mean"]), fmt(r["bPQ_sd"]), fmt(r["mPQ_mean"]), fmt(r["mPQ_sd"]),
                    fmt(r["F1det_mean"]), fmt(r["F1det_sd"]), fmt(r["F1type_mean"]), fmt(r["F1type_sd"])))
            a, b = klt_summary[(method, "A", test_scope)], klt_summary[(method, "B", test_scope)]
            lines.append("| {} | {} | **B−A** | {:+.4f} | {:+.4f} | {:+.4f} | {:+.4f} |".format(
                test_scope, method, b["bPQ_mean"] - a["bPQ_mean"], b["mPQ_mean"] - a["mPQ_mean"],
                b["F1det_mean"] - a["F1det_mean"], b["F1type_mean"] - a["F1type_mean"]))

    lines.extend(["", "### Per-class F1 and support", "",
                  "Support is shown as matched true / matched-plus-unmatched true. F1 uses the paired foreground confusion matrix; zero matched-support classes are omitted from F1type.", "",
                  "| Test scope | Method | Fold | Seed | Immune | Stromal | Epithelial | Melanocyte | Other |",
                  "|---|---|:---:|---:|---:|---:|---:|---:|---:|"])
    for test_scope in klt_test_scopes:
        for method in ("Selected PEFT", "FullFT"):
            for fold in FOLDS:
                for seed in (42, 43, 44):
                    agg = klt_results[(method, fold, seed, test_scope)]
                    cells = []
                    for name in CLASSES:
                        c = agg["per_class"][name]
                        cells.append("{} ({:,}/{:,})".format(fmt(c["f1"]), c["matched_support"], c["total_true_support"]))
                    lines.append("| {} | {} | {} | {} | {} |".format(test_scope, method, fold, seed, " | ".join(cells)))
    lines.extend(["", "| Test scope | Method | Fold | Class | F1 mean ± sample SD | B−A mean F1 |",
                  "|---|---|:---:|---|---:|---:|"])
    for test_scope in klt_test_scopes:
        for method in ("Selected PEFT", "FullFT"):
            for fold in FOLDS:
                for name in CLASSES:
                    vals = [klt_results[(method, fold, s, test_scope)]["per_class"][name]["f1"] for s in (42, 43, 44)]
                    diff = mean([klt_results[(method, "B", s, test_scope)]["per_class"][name]["f1"] for s in (42, 43, 44)]) - mean([klt_results[(method, "A", s, test_scope)]["per_class"][name]["f1"] for s in (42, 43, 44)])
                    lines.append("| {} | {} | {} | {} | {} ± {} | {} |".format(test_scope, method, fold, name,
                                 fmt(mean(vals)), fmt(sample_sd(vals)), fmt(diff) if fold == "B" else "--"))

    lines.extend(["", "### Predicted foreground-class frequencies", "",
                  "Counts include paired and unmatched foreground predictions; background/untyped predictions are excluded from the denominator.", "",
                  "| Test scope | Method | Fold | Seed | Immune | Stromal | Epithelial | Melanocyte | Other |",
                  "|---|---|:---:|---:|---:|---:|---:|---:|---:|"])
    for test_scope in klt_test_scopes:
        for method in ("Selected PEFT", "FullFT"):
            for fold in FOLDS:
                for seed in (42, 43, 44):
                    agg = klt_results[(method, fold, seed, test_scope)]
                    cells = ["{:,} ({})".format(agg["per_class"][n]["predicted_count"], pct(agg["per_class"][n]["predicted_frequency"])) for n in CLASSES]
                    lines.append("| {} | {} | {} | {} | {} |".format(test_scope, method, fold, seed, " | ".join(cells)))

    lines.extend(["", "### Paired confusion matrices", "",
                  "Rows are true labels, columns predicted labels. Background is retained exactly as stored; F1type excludes its row and column.", ""])
    for test_scope in klt_test_scopes:
        for method in ("Selected PEFT", "FullFT"):
            for fold in FOLDS:
                for seed in (42, 43, 44):
                    lines.extend(["#### {} — {} — Fold {} — seed {}".format(test_scope, method, fold, seed), ""])
                    lines.extend(confusion_md(klt_results[(method, fold, seed, test_scope)]["confusion"]))
                    lines.append("")

    # Descriptive KLT association summary, computed from the observed values.
    shift_map = {(r["tissue"], r["fold"]): r for r in klt_shifts}
    comp_map = {(r["tissue"], r["fold"], r["split"]): r for r in klt_comps}
    lines.extend([
        "### Descriptive Fold A/B correspondence", "",
        "| Test scope | Train→test TVD A / B | TEST nuclei/patch A / B | Method | B−A mPQ | B−A F1type | Dominant paired off-diagonal A / B |",
        "|---|---:|---:|---|---:|---:|---|",
    ])

    def dominant_pair(test_scope, method, fold):
        matrix = [[0 for _ in ALL_CLASSES] for _ in ALL_CLASSES]
        for seed in (42, 43, 44):
            source = klt_results[(method, fold, seed, test_scope)]["confusion"]
            for i in range(len(ALL_CLASSES)):
                for j in range(len(ALL_CLASSES)):
                    matrix[i][j] += source[i][j]
        candidates = []
        for i, true_name in enumerate(CLASSES, 1):
            row_total = sum(matrix[i][1:])
            for j, pred_name in enumerate(CLASSES, 1):
                if i != j:
                    candidates.append((matrix[i][j], true_name, pred_name, matrix[i][j] / row_total if row_total else float("nan")))
        count, true_name, pred_name, rate = max(candidates)
        return "{}→{} {:,} ({})".format(true_name, pred_name, count, pct(rate, 1))

    for test_scope in klt_test_scopes:
        shift_a, shift_b = shift_map[(test_scope, "A")], shift_map[(test_scope, "B")]
        density_a = comp_map[(test_scope, "A", "test")]["nuclei_per_patch"]
        density_b = comp_map[(test_scope, "B", "test")]["nuclei_per_patch"]
        for method in ("Selected PEFT", "FullFT"):
            a, b = klt_summary[(method, "A", test_scope)], klt_summary[(method, "B", test_scope)]
            lines.append("| {} | {} / {} | {} / {} | {} | {:+.4f} | {:+.4f} | {} / {} |".format(
                test_scope, fmt(shift_a["total_variation"], 3), fmt(shift_b["total_variation"], 3),
                fmt(density_a, 2), fmt(density_b, 2), method,
                b["mPQ_mean"] - a["mPQ_mean"], b["F1type_mean"] - a["F1type_mean"],
                dominant_pair(test_scope, method, "A"), dominant_pair(test_scope, method, "B")))
    lines.append("")
    for fold in FOLDS:
        p = comp_map[("Pooled KLT", fold, "test")]
        s = shift_map[("Pooled KLT", fold)]
        lines.append("- Fold {} has pooled train→test TVD {} and JSD {} bits; its held-out TEST density is {} nuclei/patch and entropy is {} bits.".format(
            fold, fmt(s["total_variation"], 3), fmt(s["jensen_shannon_bits"], 3), fmt(p["nuclei_per_patch"], 2), fmt(p["class_entropy_bits"], 3)))
    lines.extend([
        "- Melanocyte is absent from both KLT training and TEST in both directions, so it does not explain an A/B difference and is excluded from present-class F1type. Other is present but is the least prevalent pooled class in both directions.",
        "- No KLT test class is wholly missing from its reciprocal training set. The only declared rare→present case is Other in Liver Fold B training (0.35%); therefore a newly appearing class is not a general explanation for the fold changes.",
        "- FullFT’s mean F1type rises from {} in A to {} in B (B−A {:+.4f}); Selected PEFT changes from {} to {} (B−A {:+.4f}). These changes occur alongside reciprocal class-prevalence and density changes, but two slides per tissue do not isolate a cause.".format(
            fmt(klt_summary[("FullFT", "A", "Pooled KLT")]["F1type_mean"]), fmt(klt_summary[("FullFT", "B", "Pooled KLT")]["F1type_mean"]),
            klt_summary[("FullFT", "B", "Pooled KLT")]["F1type_mean"] - klt_summary[("FullFT", "A", "Pooled KLT")]["F1type_mean"],
            fmt(klt_summary[("Selected PEFT", "A", "Pooled KLT")]["F1type_mean"]), fmt(klt_summary[("Selected PEFT", "B", "Pooled KLT")]["F1type_mean"]),
            klt_summary[("Selected PEFT", "B", "Pooled KLT")]["F1type_mean"] - klt_summary[("Selected PEFT", "A", "Pooled KLT")]["F1type_mean"]),
    ])
    for method in ("Selected PEFT", "FullFT"):
        class_diffs = []
        for name in CLASSES:
            a = mean([klt_results[(method, "A", s, "Pooled KLT")]["per_class"][name]["f1"] for s in (42, 43, 44)])
            b = mean([klt_results[(method, "B", s, "Pooled KLT")]["per_class"][name]["f1"] for s in (42, 43, 44)])
            if math.isfinite(a) and math.isfinite(b):
                class_diffs.append((abs(b - a), name, b - a))
        class_diffs.sort(reverse=True)
        text = ", ".join("{} {:+.3f}".format(name, diff) for _, name, diff in class_diffs[:3])
        lines.append("- The largest mean per-class F1 shifts for {} are {}. The seed-level matrices above show the corresponding confusion pairs; this is descriptive coincidence, not attribution.".format(method, text))

    lines.extend(["", "## Part B — tissue-specific difficulty", ""])
    composition_table(lines, "Tissue-specific slide composition (shared by both backbones)", tissue_comps, tissue_shifts)
    lines.extend(["### Complete seed42 Selected PEFT results and completed-direction ranks", "",
                  "Ranks are descending within each backbone over currently complete tissue×fold directions. They are descriptive; incomplete directions are not assigned a rank.", "",
                  "| Backbone | Tissue | Fold | bPQ (rank) | mPQ (rank) | F1det (rank) | F1type (rank) | Type F1/support by class | Dominant paired confusion | Strong detection / weak typing flag |",
                  "|---|---|:---:|---:|---:|---:|---:|---|---|:---:|"])
    for backbone in ("CellViT-SAM-H", "CellViT-256"):
        subset = [x for x in tissue_results if x["backbone"] == backbone and x["method"] == "Selected PEFT"]
        ranks = {m: ordinal_ranks(subset, m) for m in ["bPQ", "mPQ", "F1det", "F1type"]}
        for r in sorted(subset, key=lambda x: (x["tissue"], x["fold"])):
            class_text = "; ".join("{} {} ({:,})".format(n, fmt(r["per_class"][n]["f1"], 3), r["per_class"][n]["total_true_support"]) for n in CLASSES if r["per_class"][n]["matched_support"])
            lines.append("| {} | {} | {} | {} ({}) | {} ({}) | {} ({}) | {} ({}) | {} | {}→{} ({:,}) | {} |".format(
                backbone, r["tissue"], r["fold"], fmt(r["bPQ"]), ranks["bPQ"][id(r)],
                fmt(r["mPQ"]), ranks["mPQ"][id(r)], fmt(r["F1det"]), ranks["F1det"][id(r)],
                fmt(r["F1type"]), ranks["F1type"][id(r)], class_text,
                r["dominant_confusion_true"], r["dominant_confusion_pred"], r["dominant_confusion_count"],
                "yes" if r.get("strong_detection_weak_typing") else "no"))

    if include_fullft:
        lines.extend(["", "### Matched Kidney/Liver/Tonsil PEFT versus FullFT diagnostics", "",
                      "Only exact tissue/fold/seed42 pairs with identical TEST patch-ID sets are included. Per-class support is matched-plus-unmatched true support; predicted frequencies include paired and unmatched foreground predictions.", "",
                      "| Backbone | Tissue | Fold | Method | bPQ | mPQ | F1det | F1type | Per-class F1 / true support | Predicted foreground frequency | Dominant paired confusion |",
                      "|---|---|:---:|---|---:|---:|---:|---:|---|---|---|"])
        matched = [x for x in tissue_results
                   if x["tissue"] in KLT_TISSUES and x["method"] in ("Selected PEFT", "FullFT")]
        for backbone in ("CellViT-SAM-H", "CellViT-256"):
            for tissue in KLT_TISSUES:
                for fold in FOLDS:
                    pair = [x for x in matched if x["backbone"] == backbone and x["tissue"] == tissue and x["fold"] == fold]
                    if {x["method"] for x in pair} != {"Selected PEFT", "FullFT"}:
                        continue
                    peft = next(x for x in pair if x["method"] == "Selected PEFT")
                    fullft = next(x for x in pair if x["method"] == "FullFT")
                    with peft["source"].open() as handle:
                        peft_ids = set(json.load(handle)["image_metrics"])
                    with fullft["source"].open() as handle:
                        fullft_ids = set(json.load(handle)["image_metrics"])
                    if peft_ids != fullft_ids:
                        raise RuntimeError("Non-matched TEST patch IDs: {} {} {}".format(backbone, tissue, fold))
                    for r in (peft, fullft):
                        class_text = "; ".join("{} {} / {:,}".format(
                            name, fmt(r["per_class"][name]["f1"], 3), r["per_class"][name]["total_true_support"]
                        ) for name in CLASSES)
                        pred_text = "; ".join("{} {}".format(
                            name, pct(r["per_class"][name]["predicted_frequency"], 1)
                        ) for name in CLASSES)
                        lines.append("| {} | {} | {} | {} | {} | {} | {} | {} | {} | {} | {}->{} ({:,}) |".format(
                            backbone, tissue, fold, r["method"], fmt(r["bPQ"]), fmt(r["mPQ"]),
                            fmt(r["F1det"]), fmt(r["F1type"]), class_text, pred_text,
                            r["dominant_confusion_true"], r["dominant_confusion_pred"], r["dominant_confusion_count"]))

        lines.extend(["", "### Matched tissue paired confusion matrices", "",
                      "Rows are true labels and columns predicted labels; counts are paired matches only.", ""])
        for backbone in ("CellViT-SAM-H", "CellViT-256"):
            for tissue in KLT_TISSUES:
                for fold in FOLDS:
                    for method in ("Selected PEFT", "FullFT"):
                        candidates = [x for x in matched if x["backbone"] == backbone and x["tissue"] == tissue
                                      and x["fold"] == fold and x["method"] == method]
                        if not candidates:
                            continue
                        lines.extend(["#### {} — {} — {} — Fold {}".format(backbone, tissue, method, fold), ""])
                        lines.extend(confusion_md(candidates[0]["confusion"]))
                        lines.append("")

    lines.extend(["", "### Reciprocal-fold tissue mean ranks", "",
                  "Only tissues with both A and B complete are included; means are unweighted across the two directions.", "",
                  "| Backbone | Tissue | bPQ mean (rank) | mPQ mean (rank) | F1det mean (rank) | F1type mean (rank) |",
                  "|---|---|---:|---:|---:|---:|"])
    for backbone in ("CellViT-SAM-H", "CellViT-256"):
        subset = [x for x in tissue_results if x["backbone"] == backbone and x["method"] == "Selected PEFT"]
        means = []
        for tissue in TISSUES:
            pair = [x for x in subset if x["tissue"] == tissue]
            if {x["fold"] for x in pair} == {"A", "B"}:
                means.append({"tissue": tissue, **{m: mean([x[m] for x in pair]) for m in ["bPQ", "mPQ", "F1det", "F1type"]}})
        ranks = {m: ordinal_ranks(means, m) for m in ["bPQ", "mPQ", "F1det", "F1type"]}
        for r in sorted(means, key=lambda x: ranks["mPQ"][id(x)]):
            lines.append("| {} | {} | {} ({}) | {} ({}) | {} ({}) | {} ({}) |".format(
                backbone, r["tissue"], fmt(r["bPQ"]), ranks["bPQ"][id(r)], fmt(r["mPQ"]), ranks["mPQ"][id(r)],
                fmt(r["F1det"]), ranks["F1det"][id(r)], fmt(r["F1type"]), ranks["F1type"][id(r)]))

    lines.extend(["", "### Detection–typing separation and class contribution", "",
                  "‘Strong detection / weak typing’ is operationalized only for triage: F1det at or above the median and F1type below the median among complete directions of the same backbone. It is not a biological or statistical threshold.", ""])
    for backbone in ("CellViT-SAM-H", "CellViT-256"):
        flagged = [x for x in tissue_results if x["backbone"] == backbone and x["method"] == "Selected PEFT" and x.get("strong_detection_weak_typing")]
        lines.append("- {}: {}.".format(backbone, "; ".join("{} {} (F1det {}, F1type {})".format(x["tissue"], x["fold"], fmt(x["F1det"]), fmt(x["F1type"])) for x in flagged) or "none"))
    lines.extend([
        "- The complete-result table reports every present class’s F1 and support. Low typing values are usually associated with one or more low-F1 classes rather than uniform failure across all classes; the dominant off-diagonal pair identifies the largest paired confusion by count. Other has low support and often low F1; Kidney additionally shows pronounced direction-specific Immune/Stromal/Epithelial confusion. Skin is the only tissue here with nonzero Melanocyte annotations, and its available rows must be interpreted with that distinct class composition.",
        "- Train→test TVD/JSD are properties of the tissue/fold data and therefore shared by SAM-H and CellViT-256. They should not be duplicated as independent observations across backbones.",
        "",
        "### Completion boundary and FullFT-ready schema",
        "",
        "The current artifact audit includes {}/18 SAM-H and {}/18 CellViT-256 Selected PEFT directions. {}The frozen `workshop_master_results.csv` predates several CellViT-256 completions, so completion was re-established from each inference log and exact TEST coverage rather than copied from its older status cells.".format(
            samh_complete, cv256_complete,
            ("Excluded for missing canonical evidence: {}. ".format(", ".join(missing_peft))) if missing_peft else "No Selected PEFT direction is missing. "),
        "",
        ("Matched Kidney/Liver/Tonsil FullFT rows are included through the same checkpoint-10 aggregation and confusion schema. FullFT is not extrapolated to the other six tissues."
         if include_fullft else
         "FullFT rows are not included in this snapshot; use `--include-fullft` only when canonical matched artifacts are available."),
        "",
        "## Paper-safe interpretations",
        "",
        "- Reciprocal Fold A/B directions differ in class prevalence, train→test distribution distance, and nuclei density; the observed metric changes occur alongside those measured data changes.",
        "- KLT Melanocyte is absent in train and TEST in both folds and is excluded from present-class F1type. Other is present but least prevalent and frequently has low typing F1.",
        "- FullFT has a larger observed B−A mean F1type change than Selected PEFT in the three-seed KLT results. The largest per-class shifts and confusion matrices are reported above; they do not establish why the change occurred.",
        "- Several tissue directions retain comparatively strong detection while typing is weak under the declared within-backbone median rule. Their per-class F1/support and dominant confusions localize the descriptive typing loss.",
        "- Tissue difficulty is direction-dependent: reciprocal slides can change both class composition and metric ranking. Fold directions are not interchangeable replicates.",
        "- These results support slide-held-out wording only. They do not support causal, statistical-significance, patient-level, equivalence, or non-inferiority claims.",
        "",
        "## Machine-readable record types",
        "",
        "`slide_tissue_diagnostics.csv` is long-form and append-safe. `slide_composition` and `distribution_shift` store data composition; `klt_seed_metric`, `klt_per_class`, `klt_confusion`, `klt_fold_summary`, and `klt_fold_difference` store Part A; `tissue_result`, `tissue_per_class`, `tissue_reciprocal_fold_mean`, and `tissue_missing` store Part B. Every evidence row carries a source path where applicable.",
        "",
    ])
    DIAG_MD.write_text("\n".join(lines))


HISTORICAL_METHODS = [
    ("Frozen CellViT", "none", [
        (42, "run/sthelar40x_kidney_liver_tonsil_5class_spatial_margin128_freeze_e10_seed42_CLEAN/log/2026-06-24T195646_sthelar40x_kidney_liver_tonsil_5class_spatial_margin128_freeze_e10_seed42_CLEAN/inference_results.json")]),
    ("LoRA", "frozen decoder", [
        (42, "run/sthelar40x_kidney_liver_tonsil_5class_spatial_margin128_lora_r8_a8_decoder_frozen_lr5e-5_e10_seed42_CLEAN/log/2026-06-24T220904_sthelar40x_kidney_liver_tonsil_5class_spatial_margin128_lora_r8_a8_decoder_frozen_lr5e-5_e10_seed42_CLEAN/inference_results.json")]),
    ("AdaptFormer", "frozen decoder", [
        (42, "run/sthelar40x_kidney_liver_tonsil_5class_spatial_margin128_adaptformer_red16_decoder_frozen_lr5e-5_e10_seed42_CLEAN/log/2026-06-25T001346_sthelar40x_kidney_liver_tonsil_5class_spatial_margin128_adaptformer_red16_decoder_frozen_lr5e-5_e10_seed42_CLEAN/inference_results.json")]),
    ("VeRA", "frozen decoder", [
        (42, "run/sthelar40x_kidney_liver_tonsil_5class_spatial_margin128_vera_r16_a16_decoder_frozen_lr5e-5_e10_seed42_CLEAN/log/2026-06-25T070650_sthelar40x_kidney_liver_tonsil_5class_spatial_margin128_vera_r16_a16_decoder_frozen_lr5e-5_e10_seed42_CLEAN/inference_results.json")]),
    ("Decoder conv adapters", "conv adapters", [
        (42, "run/sthelar40x_kidney_liver_tonsil_5class_spatial_margin128_frozen_encoder_decoder_conv_adapters_lr5e-5_e10_seed42_CLEAN/log/2026-06-24T201215_sthelar40x_kidney_liver_tonsil_5class_spatial_margin128_frozen_encoder_decoder_conv_adapters_lr5e-5_e10_seed42_CLEAN/inference_results.json")]),
    ("FullFT", "all", [
        (42, "run/sthelar40x_kidney_liver_tonsil_5class_spatial_margin128_fullft_lr1e-5_e10_seed42_CLEAN/log/2026-06-24T195748_sthelar40x_kidney_liver_tonsil_5class_spatial_margin128_fullft_lr1e-5_e10_seed42_CLEAN/inference_results.json"),
        (43, "run/sthelar40x_kidney_liver_tonsil_5class_spatial_margin128_fullft_lr1e-5_e10_seed43_KEEPALL_FISHER/log/2026-06-26T002217_sthelar40x_kidney_liver_tonsil_5class_spatial_margin128_fullft_lr1e-5_e10_seed43_KEEPALL_FISHER/inference_results.json")]),
    ("LoRA+AdaptFormer", "heads only", [
        (42, "run/sthelar40x_kidney_liver_tonsil_5class_spatial_margin128_lora_adaptformer_r8_a8_red16_decoder_heads_only_lr5e-5_e10_seed42_CLEAN/log/2026-06-25T113302_sthelar40x_kidney_liver_tonsil_5class_spatial_margin128_lora_adaptformer_r8_a8_red16_decoder_heads_only_lr5e-5_e10_seed42_CLEAN/inference_results.json"),
        (43, "run/sthelar40x_kidney_liver_tonsil_5class_spatial_margin128_lora_adaptformer_r8_a8_red16_decoder_heads_only_lr5e-5_e10_seed43_CLEAN/log/2026-06-26T002217_sthelar40x_kidney_liver_tonsil_5class_spatial_margin128_lora_adaptformer_r8_a8_red16_decoder_heads_only_lr5e-5_e10_seed43_CLEAN/inference_results.json")]),
    ("LoRA+AdaptFormer", "last stage", [
        (42, "run/sthelar40x_kidney_liver_tonsil_5class_spatial_margin128_lora_adaptformer_r8_a8_red16_decoder_last_stage_lr5e-5_e10_seed42_CLEAN/log/2026-06-25T185008_sthelar40x_kidney_liver_tonsil_5class_spatial_margin128_lora_adaptformer_r8_a8_red16_decoder_last_stage_lr5e-5_e10_seed42_CLEAN/inference_results.json"),
        (43, "run/sthelar40x_kidney_liver_tonsil_5class_spatial_margin128_lora_adaptformer_r8_a8_red16_decoder_last_stage_lr5e-5_e10_seed43_CLEAN/log/2026-06-27T201436_sthelar40x_kidney_liver_tonsil_5class_spatial_margin128_lora_adaptformer_r8_a8_red16_decoder_last_stage_lr5e-5_e10_seed43_CLEAN/inference_results.json")]),
    ("LoRA+AdaptFormer", "conv adapters", [
        (42, "run/sthelar40x_kidney_liver_tonsil_5class_spatial_margin128_lora_adaptformer_r8_a8_red16_decoder_conv_adapters_lr5e-5_e10_seed42_CLEAN/log/2026-06-25T102125_sthelar40x_kidney_liver_tonsil_5class_spatial_margin128_lora_adaptformer_r8_a8_red16_decoder_conv_adapters_lr5e-5_e10_seed42_CLEAN/inference_results.json"),
        (43, "run/sthelar40x_kidney_liver_tonsil_5class_spatial_margin128_lora_adaptformer_r8_a8_red16_decoder_conv_adapters_lr5e-5_e10_seed43_CLEAN/log/2026-06-26T002217_sthelar40x_kidney_liver_tonsil_5class_spatial_margin128_lora_adaptformer_r8_a8_red16_decoder_conv_adapters_lr5e-5_e10_seed43_CLEAN/inference_results.json")]),
    ("VeRA+AdaptFormer", "heads only", [
        (43, "run/sthelar40x_kidney_liver_tonsil_5class_spatial_margin128_vera_adaptformer_r16_a16_red16_decoder_heads_only_lr5e-5_e10_seed43_CLEAN/log/2026-06-28T022341_sthelar40x_kidney_liver_tonsil_5class_spatial_margin128_vera_adaptformer_r16_a16_red16_decoder_heads_only_lr5e-5_e10_seed43_CLEAN/inference_results.json")]),
    ("VeRA+AdaptFormer", "conv adapters", [
        (42, "run/sthelar40x_kidney_liver_tonsil_5class_spatial_margin128_vera_adaptformer_r16_a16_red16_decoder_conv_adapters_lr5e-5_e10_seed42_CLEAN/log/2026-06-25T102125_sthelar40x_kidney_liver_tonsil_5class_spatial_margin128_vera_adaptformer_r16_a16_red16_decoder_conv_adapters_lr5e-5_e10_seed42_CLEAN/inference_results.json"),
        (43, "run/sthelar40x_kidney_liver_tonsil_5class_spatial_margin128_vera_adaptformer_r16_a16_red16_decoder_conv_adapters_lr5e-5_e10_seed43_CLEAN/log/2026-06-27T235829_sthelar40x_kidney_liver_tonsil_5class_spatial_margin128_vera_adaptformer_r16_a16_red16_decoder_conv_adapters_lr5e-5_e10_seed43_CLEAN/inference_results.json")]),
]


AUDIT_FIELDS = ["record_type", "method", "decoder_scope", "seed", "comparable_seed_count", "checkpoint_policy",
                "test_patch_count", "Dice", "bPQ", "mPQ", "F1det", "Dice_mean", "Dice_sample_sd",
                "bPQ_mean", "bPQ_sample_sd", "mPQ_mean", "mPQ_sample_sd", "F1det_mean", "F1det_sample_sd",
                "source_inference_json", "source_inference_log", "json_sha256", "provenance_flag", "note"]


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def build_ablation_audit():
    evidence = []
    summaries = []
    csv_rows = []
    for method, decoder, seeds in HISTORICAL_METHODS:
        items = []
        for seed, source in seeds:
            path = REPO / source
            ok, reason = verify_checkpoint10(path)
            if not ok:
                raise RuntimeError("Noncanonical historical evidence {}: {}".format(path, reason))
            agg = aggregate_inference(path)
            item = {"method": method, "decoder_scope": decoder, "seed": seed, "source": path,
                    "log": path.with_name("inference.log"), "sha256": sha256(path), **agg}
            items.append(item)
            evidence.append(item)
            flag = "FROZEN_TAXONOMY_SPECIAL_CASE" if method == "Frozen CellViT" else ("SINGLE_CANONICAL_SEED" if len(seeds) == 1 else "COMPARABLE_CANONICAL_SEED")
            row = {"record_type": "seed_evidence", "method": method, "decoder_scope": decoder, "seed": seed,
                   "comparable_seed_count": len(seeds), "checkpoint_policy": reason, "test_patch_count": agg["n_images"],
                   "Dice": agg["Dice"], "bPQ": agg["bPQ"], "mPQ": "" if method == "Frozen CellViT" else agg["mPQ"],
                   "F1det": agg["F1det"], "source_inference_json": rel(path), "source_inference_log": rel(path.with_name("inference.log")),
                   "json_sha256": item["sha256"], "provenance_flag": flag,
                   "note": "Frozen taxonomy-dependent mPQ is excluded." if method == "Frozen CellViT" else "Exact checkpoint-10 TEST evidence."}
            csv_rows.append(row)
        summary = {"method": method, "decoder_scope": decoder, "n": len(items), "items": items}
        for metric in ["Dice", "bPQ", "mPQ", "F1det"]:
            values = [] if method == "Frozen CellViT" and metric == "mPQ" else [x[metric] for x in items]
            summary[metric + "_mean"] = mean(values)
            summary[metric + "_sd"] = sample_sd(values)
        summaries.append(summary)
        flag = "HETEROGENEOUS_TAXONOMY_SPECIAL_CASE" if method == "Frozen CellViT" else ("SINGLE_RUN_NO_SD" if len(items) == 1 else "HOMOGENEOUS_WITHIN_ROW")
        csv_rows.append({"record_type": "method_summary", "method": method, "decoder_scope": decoder,
                         "seed": ";".join(str(x["seed"]) for x in items), "comparable_seed_count": len(items),
                         "checkpoint_policy": "checkpoint_10 TEST only",
                         "Dice_mean": summary["Dice_mean"], "Dice_sample_sd": summary["Dice_sd"] if len(items) >= 2 else "",
                         "bPQ_mean": summary["bPQ_mean"], "bPQ_sample_sd": summary["bPQ_sd"] if len(items) >= 2 else "",
                         "mPQ_mean": summary["mPQ_mean"] if math.isfinite(summary["mPQ_mean"]) else "",
                         "mPQ_sample_sd": summary["mPQ_sd"] if len(items) >= 2 else "",
                         "F1det_mean": summary["F1det_mean"], "F1det_sample_sd": summary["F1det_sd"] if len(items) >= 2 else "",
                         "provenance_flag": flag,
                         "note": "Sample SD is blank for n<2; model_best evidence is never mixed."})
    return evidence, summaries, csv_rows


def write_audit_csv(rows):
    with AUDIT_CSV.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=AUDIT_FIELDS, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def metric_mean_sd(summary, metric):
    m = summary[metric + "_mean"]
    sd = summary[metric + "_sd"]
    if not math.isfinite(m):
        return "--"
    return "{} ± {}".format(fmt(m), fmt(sd)) if summary["n"] >= 2 else fmt(m)


def write_audit_md(evidence, summaries):
    lines = [
        "# Workshop historical KLT ablation seed audit",
        "",
        "This audit covers the 11 historical within-slide KLT ablation rows reproduced in the workshop materials. It uses only canonical checkpoint-10 TEST evidence. `model_best` results are not substituted or mixed, and sample SD is reported only for methods with at least two genuinely comparable canonical seeds.",
        "",
        "## Method-level audit", "",
        "| Method | Decoder | Comparable canonical seeds | Dice | bPQ | mPQ | F1det | Provenance |",
        "|---|---|---:|---:|---:|---:|---:|---|",
    ]
    for s in summaries:
        provenance = "taxonomy special case; mPQ excluded" if s["method"] == "Frozen CellViT" else ("single run; no SD" if s["n"] == 1 else "homogeneous checkpoint-10 TEST seeds within row")
        lines.append("| {} | {} | {} | {} | {} | {} | {} | {} |".format(
            s["method"], s["decoder_scope"], s["n"], metric_mean_sd(s, "Dice"), metric_mean_sd(s, "bPQ"),
            metric_mean_sd(s, "mPQ"), metric_mean_sd(s, "F1det"), provenance))
    lines.extend([
        "",
        "All ± values are mean ± sample SD (n−1). A bare value is a single canonical run, not a zero-variance estimate. Frozen’s class-aware output is not taxonomy-compatible with the STHELAR labels, so its mPQ remains blank even though the historical JSON contains a numeric value.",
        "",
        "## Exact seed-level checkpoint-10 TEST evidence", "",
        "| Method | Decoder | Seed | TEST patches | Dice | bPQ | mPQ | F1det | JSON SHA-256 | Canonical JSON / inference log |",
        "|---|---|---:|---:|---:|---:|---:|---:|---|---|",
    ])
    for e in evidence:
        m = "--" if e["method"] == "Frozen CellViT" else fmt(e["mPQ"])
        lines.append("| {} | {} | {} | {:,} | {} | {} | {} | {} | `{}` | `{}` / `{}` |".format(
            e["method"], e["decoder_scope"], e["seed"], e["n_images"], fmt(e["Dice"]), fmt(e["bPQ"]), m,
            fmt(e["F1det"]), e["sha256"], rel(e["source"]), rel(e["log"])))
    lines.extend([
        "",
        "Each listed inference log explicitly resolves inference to `checkpoint_10.pth`. The hash identifies the exact TEST JSON audited here; checkpoint retention or adapter export after evaluation does not change that evidence identity.",
        "",
        "## Provenance heterogeneity flags", "",
        "- **Table-wide replication depth is heterogeneous:** five rows have one canonical seed, five have two, and VeRA+AdaptFormer heads-only has one. Cross-row comparisons therefore do not share the same uncertainty basis.",
        "- **Frozen is a taxonomy special case:** bPQ/F1det are usable class-agnostic quantities, but the pretrained output taxonomy is not a valid STHELAR typing baseline; mPQ/type metrics are excluded.",
        "- **Historical generation is heterogeneous:** the table combines original seed42 controls with later seed43 extensions. Within each two-seed row, the audited protocol, dataset split, epoch-10 policy, and TEST scope are comparable; the FullFT seed43 run’s `KEEPALL_FISHER` suffix reflects extra artifact retention, not a different TEST checkpoint policy.",
        "- **Single-seed rows carry no estimated seed uncertainty:** no SD is supplied or implied for Frozen, LoRA, AdaptFormer, VeRA, decoder-conv adapters, or VeRA+AdaptFormer heads-only.",
        "- **No checkpoint-policy mixing:** no `model_best` metric appears in either the method summaries or seed evidence rows.",
        "",
        "## Paper-safe interpretation", "",
        "The historical ablation supports descriptive checkpoint-10 TEST comparisons among the listed methods. Only the five two-seed method rows permit a sample SD, while the remaining rows are single-run point estimates. The heterogeneous replication depth and Frozen taxonomy mismatch should be disclosed; these data do not support significance, equivalence, non-inferiority, or causal claims.",
        "",
    ])
    AUDIT_MD.write_text("\n".join(lines))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--include-fullft", action="store_true",
                        help="also include any full-coverage tissue-specific FullFT checkpoint-10 TEST rows")
    args = parser.parse_args()

    klt_comps, klt_shifts, tissue_comps, tissue_shifts = build_compositions()
    klt_results, klt_summary, klt_rows = build_klt_results()
    tissue_results, tissue_missing, tissue_rows = build_tissue_results(args.include_fullft)
    diag_rows = klt_comps + klt_shifts + tissue_comps + tissue_shifts + klt_rows + tissue_rows
    write_diag_csv(diag_rows)
    write_diag_md(klt_comps, klt_shifts, tissue_comps, tissue_shifts, klt_results, klt_summary,
                  tissue_results, tissue_missing, args.include_fullft)

    evidence, summaries, audit_rows = build_ablation_audit()
    write_audit_csv(audit_rows)
    write_audit_md(evidence, summaries)
    for path in [DIAG_MD, DIAG_CSV, AUDIT_MD, AUDIT_CSV]:
        print(rel(path))


if __name__ == "__main__":
    main()
