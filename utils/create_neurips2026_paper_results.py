#!/usr/bin/env python3
"""Build the canonical paper-results package from completed runs.

This script is deliberately read-only with respect to runs and checkpoints.  It
only reads saved final-test inference records and the existing efficiency audit.
"""

import csv
import json
import math
import os
import statistics
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib-paper-results")

from utils.analysis.type_assignment_analysis import (  # noqa: E402
    CANONICAL_CLASSES,
    aggregate_detection_stats,
    canonical_ids,
    extract_slide_id,
    finite_mean,
    load_nuclei_types,
    plot_confusion_png,
    type_metrics,
)


OUT = ROOT / "reports" / "neurips2026_paper_results"
CONF_OUT = OUT / "confusion_matrices"
NA = "NA"
PRIMARY_METRICS = ("Dice", "Jaccard", "bPQ", "mPQ", "F1det", "F1type")
CLASS_FIELDS = ("f1", "precision", "recall", "matched_true_support",
                "matched_pred_support", "unmatched_true", "unmatched_pred",
                "total_true_support", "total_pred_support")


def rel(path):
    return str(Path(path).resolve().relative_to(ROOT))


def one(pattern):
    matches = sorted(ROOT.glob(pattern))
    if len(matches) != 1:
        raise RuntimeError("Expected exactly one match for {!r}; got {}: {}".format(
            pattern, len(matches), matches))
    return matches[0]


def run_specs():
    specs = []
    for fold in ("A", "B"):
        specs.append(dict(backbone="SAM-H", method="Frozen", seed=42, fold=fold,
                          path=one("run/sthelar40x_klt_5class_slideind_fold{}_frozen_seed42/eval/frozen_class_agnostic_results.json".format(fold))))
        for method, token in (
            ("LP", "lp_final_heads"),
            ("Selected PEFT", "lora_adaptformer_r8_a8_red16_heads"),
            ("NT-header1", "lora_adaptformer_r8_a8_red16_nt_header1_np_hv_heads"),
            ("FullFT", "fullft_lr1e-5"),
        ):
            for seed in (42, 43):
                specs.append(dict(backbone="SAM-H", method=method, seed=seed, fold=fold,
                                  path=one("run/sthelar40x_klt_5class_slideind_fold{}_{}_e10_seed{}/log/*/inference_results.json".format(fold, token, seed))))
        specs.append(dict(backbone="CellViT-256", method="Frozen", seed=42, fold=fold,
                          path=one("run/sthelar40x_klt_5class_slideind_fold{}_cellvit256_frozen_seed42/eval/frozen_class_agnostic_results.json".format(fold))))
        for method, token in (
            ("LP", "cellvit256_lp_final_heads"),
            ("Selected PEFT", "cellvit256_lora_adaptformer_r8_a8_red16_heads"),
            ("FullFT", "cellvit256_fullft_lr1e-5"),
        ):
            specs.append(dict(backbone="CellViT-256", method=method, seed=42, fold=fold,
                              path=one("run/sthelar40x_klt_5class_slideind_fold{}_{}_e10_seed42/log/*/inference_results.json".format(fold, token))))
    return specs


def within_specs():
    mapping = [
        ("LP", 42, "run/sthelar40x_kidney_liver_tonsil_5class_spatial_margin128_klt_final_heads_only_frozen_encoder_decoder_e10_seed42/log/*/inference_results.json"),
        ("Selected PEFT", 42, "run/sthelar40x_kidney_liver_tonsil_5class_spatial_margin128_lora_adaptformer_r8_a8_red16_decoder_heads_only_lr5e-5_e10_seed42_CLEAN/log/*/inference_results.json"),
        ("Selected PEFT", 43, "run/sthelar40x_kidney_liver_tonsil_5class_spatial_margin128_lora_adaptformer_r8_a8_red16_decoder_heads_only_lr5e-5_e10_seed43_CLEAN/log/*/inference_results.json"),
        ("FullFT", 42, "run/sthelar40x_kidney_liver_tonsil_5class_spatial_margin128_fullft_lr1e-5_e10_seed42_CLEAN/log/*/inference_results.json"),
        ("FullFT", 43, "run/sthelar40x_kidney_liver_tonsil_5class_spatial_margin128_fullft_lr1e-5_e10_seed43_KEEPALL_FISHER/log/*/inference_results.json"),
    ]
    return [dict(backbone="SAM-H", method=m, seed=s, protocol="within-slide", path=one(p))
            for m, s, p in mapping]


def load_json(path):
    with Path(path).open() as handle:
        return json.load(handle)


def class_aware(spec, data):
    # Older completed within-slide files predate the explicit metric_scope tag;
    # their saved dataset mPQ and six-class detection confusion establish that
    # they are class-aware.  Frozen is always forced to class-agnostic.
    return spec["method"] != "Frozen" and (
        data.get("metric_scope") == "class_aware" or data.get("dataset", {}).get("mPQ") is not None
    )


def analyze(spec):
    data = load_json(spec["path"])
    dataset = data["dataset"]
    aware = class_aware(spec, data)
    result = dict(spec)
    result["source"] = rel(spec["path"])
    result["data"] = data
    result["class_aware"] = aware
    result["metrics"] = {
        "Dice": dataset.get("Binary-Cell-Dice-Mean"),
        "Jaccard": dataset.get("Binary-Cell-Jacard-Mean"),
        "bPQ": dataset.get("bPQ"),
        "mPQ": dataset.get("mPQ") if aware else None,
        "F1det": dataset.get("f1_detection"),
        "F1type": None,
    }
    result["class_rows"] = []
    result["confusion"] = None
    result["unpaired_true"] = None
    result["unpaired_pred"] = None
    if aware:
        run_dir = spec["path"].parent
        nuclei_types = load_nuclei_types(run_dir, data)
        ids = canonical_ids(nuclei_types)
        n_classes = max(nuclei_types.values()) + 1
        records = list(data.get("image_metrics", {}).values())
        conf_all, utr_all, upr_all, found = aggregate_detection_stats(records, n_classes)
        if found != len(records):
            raise RuntimeError("Missing detection stats in {}: {}/{}".format(spec["path"], found, len(records)))
        summary, rows, conf, utr, upr = type_metrics(conf_all, utr_all, upr_all, ids)
        result["class_rows"] = enrich_class_rows(rows)
        present_f1 = [row["f1"] for row in result["class_rows"]
                      if row["matched_true_support"] > 0]
        result["metrics"]["F1type"] = mean(present_f1)
        result["confusion"] = conf
        result["unpaired_true"] = utr
        result["unpaired_pred"] = upr
        # Detection is class-agnostic and therefore uses all six saved slots,
        # including type-0 predictions.  F1type/per-class results intentionally
        # use only the five foreground taxonomy classes.
        matched_all = int(conf_all.sum())
        unmatched_true_all = int(utr_all.sum())
        unmatched_pred_all = int(upr_all.sum())
        det_den = 2 * matched_all + unmatched_true_all + unmatched_pred_all
        det_from_stats = 2.0 * matched_all / det_den if det_den else None
        if not math.isclose(det_from_stats, result["metrics"]["F1det"], rel_tol=1e-9, abs_tol=1e-9):
            raise RuntimeError("Detection F1 mismatch for {}".format(spec["path"]))
    return result


def enrich_class_rows(rows):
    out = []
    for row in rows:
        row = dict(row)
        row["total_true_support"] = row["matched_true_support"] + row["unmatched_true"]
        row["total_pred_support"] = row["matched_pred_support"] + row["unmatched_pred"]
        out.append(row)
    return out


def normalize_value(value):
    if value is None:
        return NA
    if isinstance(value, float):
        if not math.isfinite(value):
            return NA
        return "{:.10f}".format(value).rstrip("0").rstrip(".")
    return value


def write_csv(path, rows, fields=None):
    path.parent.mkdir(parents=True, exist_ok=True)
    if fields is None:
        fields = []
        seen = set()
        for row in rows:
            for key in row:
                if key not in seen:
                    fields.append(key)
                    seen.add(key)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(
            handle, fieldnames=fields, extrasaction="ignore", lineterminator="\n"
        )
        writer.writeheader()
        for row in rows:
            writer.writerow({key: normalize_value(row.get(key)) for key in fields})


def mean(values):
    values = [float(x) for x in values if x is not None and math.isfinite(float(x))]
    return sum(values) / len(values) if values else None


def sample_sd(values):
    values = [float(x) for x in values if x is not None and math.isfinite(float(x))]
    return statistics.stdev(values) if len(values) > 1 else None


def flattened_values(result):
    values = dict(result["metrics"])
    by_class = {r["class"]: r for r in result["class_rows"]}
    for class_name in CANONICAL_CLASSES:
        token = class_name.lower()
        row = by_class.get(class_name, {})
        for field in ("f1", "matched_true_support", "total_true_support"):
            values["{}_{}".format(token, field)] = row.get(field)
    return values


def wide_fold_table(results, backbone, methods):
    selected = [r for r in results if r["backbone"] == backbone and r["method"] in methods]
    value_names = list(PRIMARY_METRICS)
    for class_name in CANONICAL_CLASSES:
        for field in ("f1", "matched_true_support", "total_true_support"):
            value_names.append("{}_{}".format(class_name.lower(), field))
    rows = []
    for method in methods:
        method_rows = [r for r in selected if r["method"] == method]
        seeds = sorted(set(r["seed"] for r in method_rows))
        for seed in seeds:
            row = {"method": method, "seed": seed, "row_type": "seed_result",
                   "aggregation_definition": "single completed seed; Fold A/B kept separate"}
            for fold in ("A", "B"):
                found = [r for r in method_rows if r["seed"] == seed and r["fold"] == fold]
                if len(found) != 1:
                    raise RuntimeError("Missing/duplicate {} {} Fold {}".format(method, seed, fold))
                values = flattened_values(found[0])
                row["fold_{}_n_seeds".format(fold)] = 1
                row["fold_{}_test_slides".format(fold)] = "kidney_s1;liver_s1;tonsil_s1" if fold == "A" else "kidney_s0;liver_s0;tonsil_s0"
                row["fold_{}_source".format(fold)] = found[0]["source"]
                for name in value_names:
                    row["fold_{}_{}".format(fold, name)] = values.get(name)
            rows.append(row)

        for agg_name, func in (("mean_across_seeds", mean), ("sd_across_seeds", sample_sd)):
            row = {"method": method, "seed": "MEAN" if agg_name.startswith("mean") else "SD",
                   "row_type": agg_name,
                   "aggregation_definition": "computed independently within each fold; sample SD (n-1)"}
            for fold in ("A", "B"):
                fold_rows = [r for r in method_rows if r["fold"] == fold]
                row["fold_{}_n_seeds".format(fold)] = len(fold_rows)
                row["fold_{}_test_slides".format(fold)] = "kidney_s1;liver_s1;tonsil_s1" if fold == "A" else "kidney_s0;liver_s0;tonsil_s0"
                row["fold_{}_source".format(fold)] = "completed seeds: {}".format(";".join(str(r["seed"]) for r in fold_rows))
                for name in value_names:
                    row["fold_{}_{}".format(fold, name)] = func([flattened_values(r).get(name) for r in fold_rows])
            if agg_name == "mean_across_seeds":
                for name in value_names:
                    row["mean_of_fold_means_{}".format(name)] = mean([
                        row.get("fold_A_{}".format(name)), row.get("fold_B_{}".format(name))])
            rows.append(row)
    return rows


def subset_records(result, slide_id):
    return [record for name, record in result["data"].get("image_metrics", {}).items()
            if extract_slide_id(name) == slide_id]


def analyze_records(result, records):
    out = {
        "num_patches": len(records),
        "bPQ": finite_mean([r.get("bPQ") for r in records]),
        "mPQ": finite_mean([r.get("mPQ") for r in records]) if result["class_aware"] else None,
        "F1det": None,
        "F1type": None,
        "class_rows": [],
    }
    if not records:
        return out
    # Six entries are fixed by the saved STHELAR 5-class detection-stat schema.
    conf, utr, upr, found = aggregate_detection_stats(records, 6)
    matched = int(conf.sum())
    unmatched_true = int(utr.sum())
    unmatched_pred = int(upr.sum())
    denominator = 2 * matched + unmatched_true + unmatched_pred
    out["F1det"] = 2.0 * matched / denominator if denominator else None
    if result["class_aware"]:
        ids = {"Epithelial": 3, "Immune": 1, "Stromal": 2, "Melanocyte": 4, "Other": 5}
        summary, rows, _, _, _ = type_metrics(conf, utr, upr, ids)
        out["class_rows"] = enrich_class_rows(rows)
        out["F1type"] = mean([row["f1"] for row in out["class_rows"]
                              if row["matched_true_support"] > 0])
    return out


def per_slide_tables(results):
    include = []
    for r in results:
        if r["backbone"] == "SAM-H" and r["method"] in ("Selected PEFT", "FullFT", "NT-header1", "Frozen"):
            include.append(r)
        elif r["backbone"] == "CellViT-256" and r["method"] in ("Selected PEFT", "FullFT", "Frozen"):
            include.append(r)
    slide_rows, class_rows = [], []
    for result in include:
        slide_ids = sorted(set(extract_slide_id(x) for x in result["data"].get("image_metrics", {})))
        if len(slide_ids) != 3:
            raise RuntimeError("Expected three held-out slides in {}".format(result["source"]))
        for slide_id in slide_ids:
            stats = analyze_records(result, subset_records(result, slide_id))
            row = {"protocol": "slide-independent", "backbone": result["backbone"],
                   "method": result["method"], "seed": result["seed"], "fold": result["fold"],
                   "slide_id": slide_id, "tissue": slide_id.rsplit("_s", 1)[0].title(),
                   "num_patches": stats["num_patches"], "bPQ": stats["bPQ"], "mPQ": stats["mPQ"],
                   "F1det": stats["F1det"], "F1type": stats["F1type"],
                   "taxonomy_metrics_valid": result["class_aware"], "source": result["source"]}
            by_class = {x["class"]: x for x in stats["class_rows"]}
            for class_name in CANONICAL_CLASSES:
                cr = by_class.get(class_name, {})
                row["{}_f1".format(class_name.lower())] = cr.get("f1")
                row["{}_matched_true_support".format(class_name.lower())] = cr.get("matched_true_support")
                row["{}_total_true_support".format(class_name.lower())] = cr.get("total_true_support")
                long_row = {"protocol": "slide-independent", "scope": "slide", "backbone": result["backbone"],
                            "method": result["method"], "seed": result["seed"], "fold": result["fold"],
                            "slide_id": slide_id, "class": class_name,
                            "taxonomy_metrics_valid": result["class_aware"], "source": result["source"]}
                for field in CLASS_FIELDS:
                    long_row[field] = cr.get(field)
                class_rows.append(long_row)
            slide_rows.append(row)
    return slide_rows, class_rows


def fold_class_rows(results):
    rows = []
    for result in results:
        by_class = {x["class"]: x for x in result["class_rows"]}
        for class_name in CANONICAL_CLASSES:
            cr = by_class.get(class_name, {})
            row = {"protocol": "slide-independent", "scope": "fold", "backbone": result["backbone"],
                   "method": result["method"], "seed": result["seed"], "fold": result["fold"],
                   "slide_id": "ALL_HELD_OUT_SLIDES", "class": class_name,
                   "taxonomy_metrics_valid": result["class_aware"], "source": result["source"]}
            for field in CLASS_FIELDS:
                row[field] = cr.get(field)
            rows.append(row)
    return rows


def write_matrix_csv(path, matrix, normalized=False):
    values = matrix.astype(float)
    if normalized:
        sums = values.sum(axis=1, keepdims=True)
        values = np.divide(values, sums, out=np.zeros_like(values), where=sums != 0)
    rows = []
    for label, line in zip(CANONICAL_CLASSES, values):
        row = {"true_label": label}
        for pred, value in zip(CANONICAL_CLASSES, line):
            row[pred] = float(value) if normalized else int(value)
        rows.append(row)
    write_csv(path, rows, ["true_label"] + list(CANONICAL_CLASSES))


def confusion_artifacts(results):
    selected = [r for r in results if r["backbone"] == "SAM-H" and r["method"] in
                ("Selected PEFT", "FullFT", "NT-header1")]
    manifest = []
    aggregate = defaultdict(lambda: np.zeros((5, 5), dtype=np.int64))
    for result in selected:
        stem = "{}_seed{}_fold{}".format(result["method"].lower().replace(" ", "_").replace("-", "_"), result["seed"], result["fold"])
        raw = CONF_OUT / "individual" / (stem + "_raw.csv")
        norm = CONF_OUT / "individual" / (stem + "_row_normalized.csv")
        png = CONF_OUT / "individual" / (stem + "_row_normalized.png")
        write_matrix_csv(raw, result["confusion"], False)
        write_matrix_csv(norm, result["confusion"], True)
        png.parent.mkdir(parents=True, exist_ok=True)
        plot_confusion_png(png, result["confusion"], "{} | seed {} | Fold {}".format(result["method"], result["seed"], result["fold"]))
        aggregate[result["method"]] += result["confusion"]
        manifest.append({"scope": "individual_fold_seed", "method": result["method"], "seed": result["seed"],
                         "fold": result["fold"], "raw_csv": rel(raw), "normalized_csv": rel(norm),
                         "png": rel(png), "source": result["source"]})
    for method, matrix in aggregate.items():
        stem = method.lower().replace(" ", "_").replace("-", "_") + "_all_seeds_folds"
        raw = CONF_OUT / "aggregated_descriptive" / (stem + "_raw.csv")
        norm = CONF_OUT / "aggregated_descriptive" / (stem + "_row_normalized.csv")
        png = CONF_OUT / "aggregated_descriptive" / (stem + "_row_normalized.png")
        write_matrix_csv(raw, matrix, False)
        write_matrix_csv(norm, matrix, True)
        png.parent.mkdir(parents=True, exist_ok=True)
        plot_confusion_png(png, matrix, "{} | descriptive aggregate of seeds/folds".format(method))
        manifest.append({"scope": "descriptive_aggregate_repeated_observations", "method": method,
                         "seed": "42;43", "fold": "A;B", "raw_csv": rel(raw),
                         "normalized_csv": rel(norm), "png": rel(png),
                         "source": "sum of individual saved matched-nuclei matrices; not an inferential replicate"})
    write_csv(CONF_OUT / "manifest.csv", manifest)
    (CONF_OUT / "README.md").write_text(
        "# Matched-nuclei confusion matrices\n\n"
        "Rows are true types and columns are predicted types in the fixed order: Epithelial, Immune, Stromal, Melanocyte, Other. "
        "Each individual fold/seed matrix is preserved as raw counts, row-normalized fractions, and a paper-ready PNG. "
        "The aggregated versions are descriptive sums over repeated observations from both seeds and folds; they are not statistical replicates.\n"
    )


def efficiency_lookup():
    rows = list(csv.DictReader((ROOT / "reports/efficiency_audit/matched_fold_measurements.csv").open()))
    return {(r["backbone"], r["method"], int(r["seed"]), r["fold"]): r for r in rows}


def audited_adapter_sizes():
    rows = list(csv.DictReader((ROOT / "reports/efficiency_audit/paper_ready_efficiency.csv").open()))
    out = {}
    for row in rows:
        if int(row["seed"]) != 42:
            continue
        key = (row["backbone"], row["method"])
        out[key] = {
            "bytes": int(float(row["adapter_only_weight_bytes"])) if row.get("adapter_only_weight_bytes") else None,
            "mib": float(row["adapter_only_weight_mib"]) if row.get("adapter_only_weight_mib") else None,
        }
    return out


def matched_backbone_table(results):
    lookup = efficiency_lookup()
    adapter_sizes = audited_adapter_sizes()
    rows = []
    wanted = [("SAM-H", "Selected PEFT"), ("SAM-H", "FullFT"),
              ("CellViT-256", "Selected PEFT"), ("CellViT-256", "FullFT")]
    for backbone, method in wanted:
        fold_rows = []
        for fold in ("A", "B"):
            result = next(r for r in results if r["backbone"] == backbone and r["method"] == method and r["seed"] == 42 and r["fold"] == fold)
            eff = lookup[(backbone, method, 42, fold)]
            row = {"backbone": backbone, "method": method, "seed": 42, "fold": fold,
                   "row_type": "matched_fold", "held_out_slides": "kidney_s1;liver_s1;tonsil_s1" if fold == "A" else "kidney_s0;liver_s0;tonsil_s0",
                   "Dice": result["metrics"]["Dice"], "bPQ": result["metrics"]["bPQ"],
                   "mPQ": result["metrics"]["mPQ"], "F1det": result["metrics"]["F1det"],
                   "F1type": result["metrics"]["F1type"],
                   "trainable_parameters": int(eff["trainable_parameters"]),
                   "trainable_percent": float(eff["trainable_percent"]),
                   "training_peak_allocated_gib": float(eff["training_peak_allocated_gib"]),
                   "training_wall_seconds": float(eff["training_wall_seconds"]),
                   "training_wall_hours": float(eff["training_wall_seconds"]) / 3600.0,
                   "adapter_only_bytes": adapter_sizes[(backbone, method)]["bytes"],
                   "adapter_only_mib": adapter_sizes[(backbone, method)]["mib"],
                   "inference_peak_allocated_gib": float(eff["inference_peak_allocated_gib"]),
                   "inference_patches_per_second": float(eff["inference_patches_per_second"]),
                   "gpu_model": eff["gpu_model"], "training_batch_size": int(eff["training_batch_size"]),
                   "inference_batch_size": int(eff["inference_batch_size"]), "amp_policy": eff["amp_policy"],
                   "test_source": result["source"],
                   "efficiency_source": "reports/efficiency_audit/matched_fold_measurements.csv; adapter size from reports/efficiency_audit/paper_ready_efficiency.csv"}
            rows.append(row)
            fold_rows.append(row)
        avg = {"backbone": backbone, "method": method, "seed": 42, "fold": "A+B",
               "row_type": "mean_of_fold_values", "held_out_slides": "six complete held-out slides; folds not treated as replicates",
               "gpu_model": fold_rows[0]["gpu_model"], "training_batch_size": fold_rows[0]["training_batch_size"],
               "inference_batch_size": fold_rows[0]["inference_batch_size"], "amp_policy": fold_rows[0]["amp_policy"],
               "test_source": "mean of the two explicit matched fold rows",
               "efficiency_source": "reports/efficiency_audit/matched_fold_measurements.csv; adapter size from reports/efficiency_audit/paper_ready_efficiency.csv"}
        for field in ("Dice", "bPQ", "mPQ", "F1det", "F1type", "trainable_parameters", "trainable_percent",
                      "training_peak_allocated_gib", "training_wall_seconds", "training_wall_hours", "adapter_only_bytes",
                      "adapter_only_mib", "inference_peak_allocated_gib", "inference_patches_per_second"):
            avg[field] = mean([x.get(field) for x in fold_rows])
        rows.append(avg)
    return rows


def within_table(results):
    slide_lookup = {(r["method"], r["seed"], r["fold"]): r for r in results if r["backbone"] == "SAM-H"}
    rows = []
    for spec in within_specs():
        within = analyze(spec)
        fold_a = slide_lookup[(spec["method"], spec["seed"], "A")]
        fold_b = slide_lookup[(spec["method"], spec["seed"], "B")]
        for metric in ("Dice", "bPQ", "mPQ", "F1det", "F1type"):
            slide_mean = mean([fold_a["metrics"][metric], fold_b["metrics"][metric]])
            rows.append({"backbone": "SAM-H", "method": spec["method"], "seed": spec["seed"], "metric": metric,
                         "within_slide": within["metrics"][metric], "slide_independent_fold_A": fold_a["metrics"][metric],
                         "slide_independent_fold_B": fold_b["metrics"][metric], "slide_independent_mean_of_fold_values": slide_mean,
                         "absolute_delta_slideind_minus_within": slide_mean - within["metrics"][metric],
                         "within_test_composition": "4,653 spatially held-out patches sampled from all six KLT slides",
                         "slideind_test_composition": "Fold A: 23,494 complete-slide patches from s1; Fold B: 26,335 complete-slide patches from s0",
                         "composition_matched": False, "within_source": within["source"],
                         "slideind_fold_A_source": fold_a["source"], "slideind_fold_B_source": fold_b["source"]})
    return rows


def fmt(x, digits=3):
    return "NA" if x is None else ("{:.%df}" % digits).format(x)


def fact_sheet():
    text = """# Authoritative training and reproducibility fact sheet

Status date: 2026-08-26 (public endpoints checked 2026-08-25). Values below were traced to the completed timestamped configs, executable code, saved efficiency metadata, materialization manifests, or immutable checkpoint metadata. `UNKNOWN` means the requested value could not be verified; it is not inferred.

## Methods values

| Field | Exact verified value | Provenance |
|---|---|---|
| Input patch size | 256 × 256 RGB pixels | Timestamped run `config.yaml`, `data.input_shape` |
| Magnification | 40× | Timestamped run `config.yaml`, `data.magnification` |
| Slide-independent folds | Fold A trains/validates on kidney/liver/tonsil s0 and tests on all s1 patches; Fold B reverses s0/s1 | Fold `split_manifest.yaml` and `split_validation.json` |
| Train/validation policy | Within training slides only: spatial x-axis 85% train / 15% validation | `utils/generate_slide_independent_fold.py`; fold manifests |
| Test policy | Whole held-out slides; Fold A 23,494 test patches, Fold B 26,335; primary unfiltered `dataset`/`image_metrics` outputs only | Fold manifests and final `inference_results.json` |
| 128-pixel margin | Inherited from the earlier source materialization's spatial x-boundary exclusion bands. The new materialization reuses the packed payload and applies the margin to the within-training-slide train/validation boundary; train/test independence is by slide ID, not by this margin. | `utils/generate_slide_independent_fold.py`; source/fold manifests |
| Optimizer | AdamW, betas=(0.85, 0.85) | Timestamped configs; `base_ml/base_experiment.py` |
| Learning rate | LP 5×10⁻⁵; Selected PEFT/NT-header1 5×10⁻⁵; FullFT 1×10⁻⁵ | Timestamped configs |
| Weight decay | 0.01, inherited from the PyTorch 2.5.1 `torch.optim.AdamW` default because configs omit `weight_decay` and code forwards only configured kwargs | Configs; `base_ml/base_experiment.py`; verified local PyTorch signature |
| Scheduler | ExponentialLR, gamma=0.95 (code default; configs select `exponential` and omit gamma) | `experiment_cellvit_pannuke.py`, `get_scheduler` |
| Training batch size | 4 | Timestamped configs / efficiency audit |
| Inference batch size | 16 | Timestamped configs / efficiency audit |
| Augmentations | Each p=0.5: RandomRotate90, horizontal flip, vertical flip, downscale(scale=0.2), blur(limit=10), Gaussian noise(var_limit=10), color jitter(scale_setting=0.25, scale_color=0.1), superpixels, zoom blur, random-sized crop, elastic transform | Timestamped configs |
| Normalization | Per RGB channel mean=(0.5,0.5,0.5), SD=(0.5,0.5,0.5) | Timestamped configs |
| NP losses | xentropy_loss weight 1 + dice_loss weight 1, static | Default loss code; configs contain no override |
| HV losses | mse_loss_maps weight 1 + msge_loss_maps weight 1, static | Default loss code; configs contain no override |
| NT losses | xentropy_loss weight 1 + dice_loss weight 1, static | Default loss code; configs contain no override |
| Tissue loss | CrossEntropyLoss weight 1, static, one KLT tissue output | Default loss code and configs |
| Regression loss | None | Config/code branch not enabled |
| Class imbalance handling | Cell-presence WeightedRandomSampler, gamma=0.85, replacement=True, `num_samples=len(train_dataset)`, seeded generator. Per-class presence weight `k/(gamma*n_c+(1-gamma)*k)` and image weights from the gamma-weighted sum; absent-class-free patches receive the smallest nonzero image weight. | `cell_segmentation/datasets/pannuke.py`; `experiment_cellvit_pannuke.py`; configs |
| AMP policy | CUDA autocast FP16 for train/inference; GradScaler during training | Trainer code and saved efficiency audit |
| Epochs | 10 | Timestamped configs |
| Checkpoint policy | Save best and last, evaluate every epoch, retain final epoch under retention policy. Canonical tests here explicitly used fixed `checkpoint_10.pth`; no epoch was selected using test performance. | Configs, inference logs, `reports/checkpoint_selection_policy_audit.md` |
| Seed protocol | SAM-H Frozen seed 42; SAM-H LP/Selected PEFT/NT-header1/FullFT seeds 42 and 43; CellViT-256 all requested methods seed 42. Seed effects are summarized within each fold; folds are not seed replicates. | Completed-run inventory |
| Hardware | Matched runs: NVIDIA A100-SXM4-40GB | Saved efficiency metadata |
| Software | PyTorch 2.5.1+cu121; CUDA 12.1; Python used by project environment 3.9.25 | Saved efficiency metadata / verified environment |

## Parameter counts

| Backbone/method | Total | Trainable | Trainable % |
|---|---:|---:|---:|
| SAM-H Frozen | 699,741,149 | 0 | 0 |
| SAM-H LP | 699,736,523 | 650 | 0.0000928921 |
| SAM-H Selected PEFT | 707,644,395 | 7,908,779 | 1.117620525 |
| SAM-H NT-header1 | 707,644,395 | 7,945,835 | 1.122857053 |
| SAM-H FullFT | 699,736,523 | 699,736,523 | 100 |
| CellViT-256 Frozen | 46,750,349 | 0 | 0 |
| CellViT-256 LP | 46,743,419 | 650 | 0.0013905701 |
| CellViT-256 Selected PEFT | 47,116,967 | 374,198 | 0.7941894902 |
| CellViT-256 FullFT | 46,743,419 | 46,743,419 | 100 |

## Exact Selected PEFT definition

- LoRA rank 8, alpha 8, dropout 0, applied to Q and V slices of the fused QKV projection in every transformer encoder block.
- AdaptFormer after each encoder-block MLP, bottleneck reduction 16, GELU activation.
- Trainable final 1×1 NP, HV and NT decoder heads. SAM-H also has the one-output tissue classifier trainable under the canonical policy; CellViT-256 does not train its encoder classification head.
- Decoder bodies and pretrained backbone weights remain frozen. The verified deployable adapter additionally carries the 105 mutable normalization buffers required for exact reconstruction, although buffers are not trainable parameters.
- NT-header1 is a prespecified typing variant that additionally trains the preceding NT header convolution; it is reported separately and is not relabeled as Selected PEFT.

## Reproducibility and release status

| Item | Current factual status | Anonymous-PDF treatment |
|---|---|---|
| Git repository | The migration snapshot records its source revision in `reports/cluster_migration_snapshot.md`. Public code includes PEFT modules and within-slide release configs; the completed slide-independent seed/fold work is carried by this snapshot. | Keep repository ownership and author history outside anonymous manuscripts when required by venue policy. |
| Code required for PEFT | Present locally in `models/adapters/`, adapter application/loading utilities, training experiment code, and release verification scripts. A prior PEFT-capable revision is public; the exact new slide-independent configs/results package is not public. | Public repository link reveals identity; do not link anonymously. |
| Configs | Completed-run timestamped configs and local `configs/slide_exp/` exist. Public GitHub/Hugging Face configs cover the earlier within-slide release, not this final slide-independent package. | Public links reveal account identity; omit. |
| External adapter release | An earlier public, ungated release contains 12 within-slide KLT/tissue-specific adapter packages. **The new slide-independent Fold A/B adapters audited here are not in that remote release.** | Keep account namespaces outside anonymous manuscripts when required by venue policy. |
| Pretrained checkpoint provenance | Local required base `models/pretrained/CellViT-SAM-H-x40.pth`, 2,799,315,941 bytes, SHA256 `b324c10fddb0f80f5ab03a0459453a4c4848866934daf63435b46749a6b278cf`; README attributes it to official CellViT-SAM-H x40 and records a Google Drive download. Base is not redistributed. Exact immutable upstream release identifier/version: **UNKNOWN**. | Upstream CellViT link is identity-safe; the project README link is not. Cite CellViT normally, but do not expose the author-owned repository/HF namespace. |

No visibility, publication, checkpoint, or remote state was changed during this audit.
"""
    (OUT / "reproducibility_fact_sheet.md").write_text(text)
    rows = []
    for line in text.splitlines():
        if line.startswith("|") and not line.startswith("|---"):
            cells = [x.strip() for x in line.strip("|").split("|")]
            if len(cells) == 3 and cells[0] not in ("Field", "Item"):
                rows.append({"field": cells[0], "verified_value_or_status": cells[1], "provenance_or_anonymous_treatment": cells[2]})
    write_csv(OUT / "reproducibility_fact_sheet.csv", rows)


def summary_report(results, matched_rows, within_rows, slide_rows):
    means = {(r["backbone"], r["method"]): r for r in matched_rows if r["row_type"] == "mean_of_fold_values"}
    samh_peft = means[("SAM-H", "Selected PEFT")]
    samh_ft = means[("SAM-H", "FullFT")]
    cell_peft = means[("CellViT-256", "Selected PEFT")]
    cell_ft = means[("CellViT-256", "FullFT")]

    sam_all = [r for r in results if r["backbone"] == "SAM-H"]
    method_fold_means = {}
    for method in ("Frozen", "LP", "Selected PEFT", "NT-header1", "FullFT"):
        method_fold_means[method] = {}
        for metric in PRIMARY_METRICS:
            fold_means = [mean([r["metrics"][metric] for r in sam_all if r["method"] == method and r["fold"] == fold]) for fold in ("A", "B")]
            method_fold_means[method][metric] = mean(fold_means)

    slide_main = [r for r in slide_rows if r["seed"] == 42 and r["method"] in ("Selected PEFT", "FullFT")]
    tissue_bpq = defaultdict(list)
    for row in slide_main:
        tissue_bpq[row["tissue"]].append(row["bPQ"])
    tissue_bpq_mean = {k: mean(v) for k, v in tissue_bpq.items()}

    def delta(a, b, field):
        return a[field] - b[field]

    sam_fold = {}
    for fold in ("A", "B"):
        sam_fold[fold] = mean([r["metrics"]["bPQ"] for r in results if r["backbone"] == "SAM-H" and r["method"] in ("Selected PEFT", "FullFT") and r["seed"] == 42 and r["fold"] == fold])

    nt_cls = {}
    peft_cls = {}
    for method, holder in (("NT-header1", nt_cls), ("Selected PEFT", peft_cls)):
        for class_name in ("Epithelial", "Other"):
            fold_seed_means = []
            for fold in ("A", "B"):
                fold_seed_means.append(mean([
                    next(x["f1"] for x in r["class_rows"] if x["class"] == class_name)
                    for r in results
                    if r["backbone"] == "SAM-H" and r["method"] == method and r["fold"] == fold
                ]))
            holder[class_name] = mean(fold_seed_means)

    text = """# Final STHELAR-Adapt paper-results summary

This package uses completed, fixed-final-epoch (`checkpoint_10.pth`) test outputs only. Validation metrics are excluded. Fold A and Fold B are reported separately; seed means/SDs are calculated inside each fold, followed by the mean of the two fold means. Folds and seeds are never pooled as interchangeable replicates. `F1type` follows the existing paper definition: unweighted macro-F1 over foreground classes with nonzero matched true support, computed on matched nuclei only (four KLT classes; Melanocyte is excluded because its matched true support is zero). Per-class support columns distinguish matched support from matched+unmatched total true support. No statistical significance is claimed.

## A. Facts safe to state strongly

- The new KLT evaluation is slide-independent: Fold A holds out all s1 slides and Fold B holds out all s0 slides for kidney, liver and tonsil.
- Frozen results are strictly class-agnostic. Dice, Jaccard, bPQ and F1 detection are valid; mPQ, matched-only F1 type and per-class values are `NA`.
- SAM-H Selected PEFT trains 7,908,779/707,644,395 parameters (1.1176%); CellViT-256 Selected PEFT trains 374,198/47,116,967 (0.7942%).
- On matched seed-42 A100 runs, SAM-H PEFT uses less training allocated VRAM than SAM-H FullFT ({sam_p_vram:.3f} vs {sam_f_vram:.3f} GiB) and a 31,844,024-byte verified adapter instead of a complete model.
- CellViT-256 PEFT uses less training allocated VRAM than CellViT-256 FullFT ({cell_p_vram:.3f} vs {cell_f_vram:.3f} GiB), with a 1,623,604-byte verified adapter.
- Adapter sizes above are genuine serialized deployable state dictionaries and are distinct from training checkpoint and complete inference-model sizes. Base+adapter loading was previously verified against each canonical checkpoint with exact state equality and deterministic forward equality.

## B. Descriptive findings requiring cautious wording

- SAM-H Fold A/B asymmetry remains visible: seed-42 Selected PEFT/FullFT mean bPQ is {fold_a:.3f} in Fold A and {fold_b:.3f} in Fold B. This is a held-out-slide composition effect as well as a possible model effect.
- Across the four seed-42 PEFT/FullFT backbone-method combinations, per-slide mean bPQ is Kidney {kidney:.3f}, Liver {liver:.3f}, and Tonsil {tonsil:.3f}. {tonsil_clause}
- NT-header1 changes the matched-only typing balance descriptively: the mean of the two within-fold seed means is Epithelial F1 {nt_epi:.3f} vs {p_epi:.3f} for Selected PEFT, and Other F1 {nt_other:.3f} vs {p_other:.3f}. This is a trade-off description, not an inferential claim.
- Other-class F1 is low relative to the stronger foreground classes in several fold/seed matrices; inspect `per_class_metrics.csv` and the preserved confusion matrices when wording this point.
- Within-slide versus slide-independent deltas are descriptive only because test composition changes from 4,653 spatial patches sampled from all six slides to complete held-out slides (23,494/26,335 patches by fold).

## C. Claims not supported by these data

- No claim of statistical significance, equivalence, non-inferiority, or superiority.
- No claim that slide independence improves a model, even where class-agnostic metrics increase; the test sets differ in both slide identity and patch composition.
- No interpretation of validation metrics and no assertion that epoch 10 is optimal.
- No use of folds as independent seed replicates or of patches as independent samples for significance testing.
- No claim that Frozen has taxonomy-dependent performance.
- No claim that parameter efficiency guarantees wall-clock efficiency. CellViT-256 PEFT is slower than CellViT-256 FullFT on the matched runs ({cell_p_h:.3f} vs {cell_f_h:.3f} h), despite training only 0.7942% of its parameters. SAM-H PEFT is faster than FullFT ({sam_p_h:.3f} vs {sam_f_h:.3f} h), but the wall-time reduction is much smaller than the parameter reduction.
- No claim that the currently public adapters reproduce the new slide-independent results; the public Hugging Face packages are from the earlier within-slide release.

## D. Exact suggested numerical statements

1. “Across the two slide-independent folds at seed 42, SAM-H Selected PEFT obtained Dice {sp_dice:.3f}, bPQ {sp_bpq:.3f}, mPQ {sp_mpq:.3f}, F1 detection {sp_det:.3f}, and matched-only present-class macro-F1 type {sp_type:.3f} (mean of Fold A/B values).”
2. “The corresponding SAM-H FullFT values were Dice {sf_dice:.3f}, bPQ {sf_bpq:.3f}, mPQ {sf_mpq:.3f}, F1 detection {sf_det:.3f}, and matched-only present-class macro-F1 type {sf_type:.3f}.”
3. “SAM-H Selected PEFT trained 1.118% of parameters and reduced peak allocated training memory by {sam_vram_abs:.3f} GiB ({sam_vram_rel:.1f}%) relative to FullFT; wall time decreased by {sam_time_abs:.3f} h ({sam_time_rel:.1f}%).”
4. “Across matched seed-42 folds, CellViT-256 Selected PEFT obtained bPQ {cp_bpq:.3f} versus {cf_bpq:.3f} for CellViT-256 FullFT, while using {cp_vram:.3f} versus {cf_vram:.3f} GiB peak allocated training memory.”
5. “CellViT-256 PEFT did not provide wall-clock savings: its mean training time was {cp_h:.3f} h versus {cf_h:.3f} h for FullFT ({cell_time_pct:+.1f}%).”
6. “The verified adapter-only artifacts occupied 31,844,024 bytes (30.369 MiB) for SAM-H and 1,623,604 bytes (1.548 MiB) for CellViT-256; these values do not include the shared pretrained base.”

## File guide

- `main_samh_slideind.csv`: all valid SAM-H seeds/folds plus within-fold seed mean/SD and mean of fold means.
- `cellvit256_slideind.csv`: CellViT-256 seed-42 Fold A/B, with Frozen taxonomy fields as `NA`.
- `matched_backbone_comparison.csv`: exactly matched seed-42 Fold A/B test and pre-audited A100 efficiency values.
- `within_vs_slideind.csv`: seed-matched protocol values and deltas with composition warnings.
- `per_slide_metrics.csv`, `per_class_metrics.csv`: descriptive held-out-slide and matched-nuclei typing audit.
- `confusion_matrices/`: raw, row-normalized, and PNG individual matrices plus explicitly descriptive aggregates.
""".format(
        sam_p_vram=samh_peft["training_peak_allocated_gib"], sam_f_vram=samh_ft["training_peak_allocated_gib"],
        cell_p_vram=cell_peft["training_peak_allocated_gib"], cell_f_vram=cell_ft["training_peak_allocated_gib"],
        fold_a=sam_fold["A"], fold_b=sam_fold["B"], kidney=tissue_bpq_mean.get("Kidney"),
        liver=tissue_bpq_mean.get("Liver"), tonsil=tissue_bpq_mean.get("Tonsil"),
        tonsil_clause=("Tonsil is descriptively the lowest of the three tissues in this aggregation." if tissue_bpq_mean.get("Tonsil") == min(tissue_bpq_mean.values()) else "Tonsil is not the lowest tissue in this aggregation, so a general tonsil-difficulty claim is not supported."),
        nt_epi=nt_cls["Epithelial"], p_epi=peft_cls["Epithelial"], nt_other=nt_cls["Other"], p_other=peft_cls["Other"],
        cell_p_h=cell_peft["training_wall_hours"], cell_f_h=cell_ft["training_wall_hours"],
        sam_p_h=samh_peft["training_wall_hours"], sam_f_h=samh_ft["training_wall_hours"],
        sp_dice=samh_peft["Dice"], sp_bpq=samh_peft["bPQ"], sp_mpq=samh_peft["mPQ"], sp_det=samh_peft["F1det"], sp_type=samh_peft["F1type"],
        sf_dice=samh_ft["Dice"], sf_bpq=samh_ft["bPQ"], sf_mpq=samh_ft["mPQ"], sf_det=samh_ft["F1det"], sf_type=samh_ft["F1type"],
        sam_vram_abs=samh_ft["training_peak_allocated_gib"] - samh_peft["training_peak_allocated_gib"],
        sam_vram_rel=100 * (samh_ft["training_peak_allocated_gib"] - samh_peft["training_peak_allocated_gib"]) / samh_ft["training_peak_allocated_gib"],
        sam_time_abs=samh_ft["training_wall_hours"] - samh_peft["training_wall_hours"],
        sam_time_rel=100 * (samh_ft["training_wall_hours"] - samh_peft["training_wall_hours"]) / samh_ft["training_wall_hours"],
        cp_bpq=cell_peft["bPQ"], cf_bpq=cell_ft["bPQ"], cp_vram=cell_peft["training_peak_allocated_gib"], cf_vram=cell_ft["training_peak_allocated_gib"],
        cp_h=cell_peft["training_wall_hours"], cf_h=cell_ft["training_wall_hours"],
        cell_time_pct=100 * (cell_peft["training_wall_hours"] - cell_ft["training_wall_hours"]) / cell_ft["training_wall_hours"])
    (OUT / "final_results_summary.md").write_text(text)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    results = [analyze(spec) for spec in run_specs()]
    samh = wide_fold_table(results, "SAM-H", ("Frozen", "LP", "Selected PEFT", "NT-header1", "FullFT"))
    cell = wide_fold_table(results, "CellViT-256", ("Frozen", "LP", "Selected PEFT", "FullFT"))
    write_csv(OUT / "main_samh_slideind.csv", samh)
    write_csv(OUT / "cellvit256_slideind.csv", cell)

    matched = matched_backbone_table(results)
    write_csv(OUT / "matched_backbone_comparison.csv", matched)
    within = within_table(results)
    write_csv(OUT / "within_vs_slideind.csv", within)

    slide_rows, slide_class_rows = per_slide_tables(results)
    write_csv(OUT / "per_slide_metrics.csv", slide_rows)
    write_csv(OUT / "per_class_metrics.csv", fold_class_rows(results) + slide_class_rows)
    confusion_artifacts(results)
    fact_sheet()
    summary_report(results, matched, within, slide_rows)

    manifest = []
    for path in sorted(OUT.rglob("*")):
        if path.is_file() and path != OUT / "package_manifest.csv":
            manifest.append({"path": rel(path), "bytes": path.stat().st_size})
    write_csv(OUT / "package_manifest.csv", manifest)
    print("Wrote {} files under {}".format(len(manifest) + 1, OUT))


if __name__ == "__main__":
    main()
