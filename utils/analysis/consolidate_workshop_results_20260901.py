#!/usr/bin/env python3
"""Generate the 2026-09-01 read-only workshop paper-results package.

The script reads canonical repository artifacts and writes only paper-facing
reports. It never invokes Slurm and never changes a run, config, result JSON,
checkpoint, adapter, or manifest.
"""

from __future__ import annotations

import csv
import hashlib
import json
import math
import statistics
from collections import defaultdict
from pathlib import Path

from workshop_slide_tissue_diagnostics import (
    CLASSES,
    FOLDS,
    KLT_TISSUES,
    REPO,
    REPORT_ROOT,
    TISSUES,
    aggregate_inference,
    build_ablation_audit,
    discover_tissue_source,
    fmt,
    klt_manifest_path,
    load_manifest,
    mean,
    rel,
    sample_sd,
    tissue_manifest_path,
    verify_checkpoint10,
)


MASTER_MD = REPORT_ROOT / "workshop_master_results.md"
MASTER_CSV = REPORT_ROOT / "workshop_master_results.csv"
LATEX = REPORT_ROOT / "workshop_latex_tables.tex"
MATCHED_MD = REPORT_ROOT / "tissue_peft_vs_fullft_slideind.md"
MATCHED_CSV = REPORT_ROOT / "tissue_peft_vs_fullft_slideind.csv"
NINE_MD = REPORT_ROOT / "nine_tissue_peft_slideind.md"
NINE_CSV = REPORT_ROOT / "nine_tissue_peft_slideind.csv"

SNAPSHOT = "2026-09-01"

# Read-only Slurm snapshot taken immediately before generation.
LIVE_STATUS = {
    "1570493": "RUNNING",
    "1570494": "RUNNING",
    "1570495": "COMPLETED",
    "1570496": "COMPLETED",
    "1570497": "RUNNING",
    "1570498": "RUNNING",
}

FULLFT_JOB_IDS = {
    ("CellViT-SAM-H", "Breast", "A"): "1596567_0",
    ("CellViT-SAM-H", "Breast", "B"): "1596567_1",
    ("CellViT-SAM-H", "Colon", "A"): "1596567_2",
    ("CellViT-SAM-H", "Colon", "B"): "1596567_3",
    ("CellViT-SAM-H", "Kidney", "A"): "1586683",
    ("CellViT-SAM-H", "Kidney", "B"): "1586684",
    ("CellViT-SAM-H", "Liver", "A"): "1586685",
    ("CellViT-SAM-H", "Liver", "B"): "1586686",
    ("CellViT-SAM-H", "Lung", "A"): "1596567_4",
    ("CellViT-SAM-H", "Lung", "B"): "1596567_5",
    ("CellViT-SAM-H", "Ovary", "A"): "1596567_6",
    ("CellViT-SAM-H", "Ovary", "B"): "1596567_7",
    ("CellViT-SAM-H", "Pancreatic", "A"): "1596567_8",
    ("CellViT-SAM-H", "Pancreatic", "B"): "1596567_9",
    ("CellViT-SAM-H", "Skin", "A"): "1596567_10",
    ("CellViT-SAM-H", "Skin", "B"): "1596567_11",
    ("CellViT-SAM-H", "Tonsil", "A"): "1586687",
    ("CellViT-SAM-H", "Tonsil", "B"): "1586688",
    ("CellViT-256", "Kidney", "A"): "1586689",
    ("CellViT-256", "Kidney", "B"): "1586690",
    ("CellViT-256", "Liver", "A"): "1586691",
    ("CellViT-256", "Liver", "B"): "1586692",
    ("CellViT-256", "Tonsil", "A"): "1586693",
    ("CellViT-256", "Tonsil", "B"): "1586694",
}

TISSUE_TEMPLATES = {
    ("CellViT-SAM-H", "Selected PEFT"): "sthelar40x_{t}_5class_slideind_fold{f}_lora_adaptformer_r8_a8_red16_heads_e10_seed42",
    ("CellViT-256", "Selected PEFT"): "sthelar40x_{t}_5class_slideind_fold{f}_cellvit256_lora_adaptformer_r8_a8_red16_heads_e10_seed42",
    ("CellViT-SAM-H", "FullFT"): "sthelar40x_{t}_5class_slideind_fold{f}_fullft_lr1e-5_e10_seed42",
    ("CellViT-256", "FullFT"): "sthelar40x_{t}_5class_slideind_fold{f}_cellvit256_fullft_lr1e-5_e10_seed42",
}

EXTRA_MASTER_FIELDS = [
    "test_patches", "test_true_support", "train_slide", "test_slide", "test_patch_id_sha256",
    "training_wall_seconds", "training_wall_hours", "training_peak_allocated_gib",
    "inference_peak_allocated_gib", "inference_patches_per_second", "retained_model_state_bytes",
    "model_state_kind", "source_efficiency_json", "source_inference_efficiency_json",
]


def finite(value):
    return isinstance(value, (int, float)) and math.isfinite(value)


def nfmt(value, digits=9):
    if not finite(value):
        return ""
    return ("{:." + str(digits) + "f}").format(value)


def read_json(path):
    with Path(path).open() as handle:
        return json.load(handle)


def patch_id_hash(path):
    ids = sorted(read_json(path)["image_metrics"])
    return hashlib.sha256(("\n".join(ids) + "\n").encode()).hexdigest(), ids


def load_resources(source, method):
    run_dir = Path(source).parent
    training_path = run_dir / "efficiency_metrics.json"
    inference_path = run_dir / "inference_efficiency_metrics.json"
    training = read_json(training_path)
    inference = read_json(inference_path)
    if training.get("completed") is not True or inference.get("completed") is not True:
        raise RuntimeError("Incomplete efficiency metadata: {}".format(run_dir))
    state_bytes = None
    state_kind = None
    if method == "Selected PEFT":
        retention_path = run_dir / "adapter_checkpoint_retention_metadata.json"
        if retention_path.is_file():
            retention = read_json(retention_path)
            if not str(retention.get("status", "")).startswith("adapter_verified"):
                raise RuntimeError("Unverified adapter retention: {}".format(retention_path))
            state_bytes = int(retention["adapter_bytes"])
            if not (REPO / retention["adapter_path"]).is_file():
                raise RuntimeError("Retained adapter is missing: {}".format(retention["adapter_path"]))
            state_kind = "verified adapter safetensors"
        else:
            adapter = REPO / "adapters" / "slide_exp_safetensors" / run_dir.parent.parent.name / "adapter_model.safetensors"
            if adapter.is_file():
                state_bytes = adapter.stat().st_size
                state_kind = "adapter safetensors"
    elif method == "FullFT":
        state_bytes = int(training["checkpoint_size_bytes"])
        state_kind = "canonical FullFT checkpoint size (provenance; weight may be retention-deleted)"
    return {
        "trainable_params": int(training["trainable_parameters"]),
        "total_params": int(training["total_parameters"]),
        "trainable_pct": float(training["trainable_percentage"]),
        "training_wall_seconds": float(training["training_time_seconds"]),
        "training_wall_hours": float(training["training_time_hours"]),
        "training_peak_allocated_gib": float(training["peak_cuda_memory_allocated_gib"]),
        "inference_peak_allocated_gib": float(inference["peak_cuda_memory_allocated_gib"]),
        "inference_patches_per_second": float(inference["patches_per_second"]),
        "retained_model_state_bytes": state_bytes,
        "model_state_kind": state_kind,
        "source_efficiency_json": rel(training_path),
        "source_inference_efficiency_json": rel(inference_path),
    }


def tissue_source(backbone, method, tissue, fold):
    path = discover_tissue_source(TISSUE_TEMPLATES[(backbone, method)], tissue, fold)
    if path is None:
        raise RuntimeError("Missing canonical tissue source: {} {} {} {}".format(backbone, method, tissue, fold))
    return path


def result_record(backbone, method, tissue, fold, source):
    manifest = load_manifest(tissue_manifest_path(tissue, fold))
    expected = int(manifest["patch_counts"]["test"])
    ok, reason = verify_checkpoint10(source, expected)
    if not ok:
        raise RuntimeError("Noncanonical source {}: {}".format(source, reason))
    agg = aggregate_inference(source)
    digest, ids = patch_id_hash(source)
    resources = load_resources(source, method)
    total_true = sum(agg["per_class"][name]["total_true_support"] for name in CLASSES)
    record = {
        "backbone": backbone, "method": method, "tissue": tissue, "fold": fold, "seed": 42,
        "source": Path(source), "checkpoint_policy": reason, "test_patches": len(ids),
        "test_true_support": total_true, "train_slide": manifest["train_slide_ids"][0],
        "test_slide": manifest["test_slide_ids"][0], "test_patch_id_sha256": digest,
        **{metric: agg[metric] for metric in ("Dice", "Jaccard", "bPQ", "mPQ", "F1det", "F1type")},
        "per_class": agg["per_class"], "confusion": agg["confusion"],
        "dominant_confusion": "{}->{}".format(agg["dominant_confusion_true"], agg["dominant_confusion_pred"]),
        "dominant_confusion_count": agg["dominant_confusion_count"], **resources,
    }
    return record


def build_tissue_records():
    peft = []
    fullft = []
    for backbone in ("CellViT-SAM-H", "CellViT-256"):
        for tissue in TISSUES:
            for fold in FOLDS:
                source = tissue_source(backbone, "Selected PEFT", tissue, fold)
                peft.append(result_record(backbone, "Selected PEFT", tissue, fold, source))
        fullft_tissues = TISSUES if backbone == "CellViT-SAM-H" else KLT_TISSUES
        for tissue in fullft_tissues:
            for fold in FOLDS:
                source = tissue_source(backbone, "FullFT", tissue, fold)
                fullft.append(result_record(backbone, "FullFT", tissue, fold, source))
    return peft, fullft


def build_matched(peft, fullft):
    pmap = {(x["backbone"], x["tissue"], x["fold"]): x for x in peft}
    fmap = {(x["backbone"], x["tissue"], x["fold"]): x for x in fullft}
    rows = []
    for key in sorted(fmap, key=lambda x: (("CellViT-SAM-H", "CellViT-256").index(x[0]), TISSUES.index(x[1]), x[2])):
        p, f = pmap[key], fmap[key]
        if p["test_patch_id_sha256"] != f["test_patch_id_sha256"]:
            raise RuntimeError("PEFT/FullFT TEST set mismatch: {}".format(key))
        row = {
            "backbone": key[0], "tissue": key[1], "fold": key[2], "seed": 42,
            "train_slide": p["train_slide"], "test_slide": p["test_slide"],
            "matched_test_patches": p["test_patches"], "test_patch_id_sha256": p["test_patch_id_sha256"],
            "peft_source_inference_json": rel(p["source"]), "fullft_source_inference_json": rel(f["source"]),
        }
        for metric in ("bPQ", "mPQ", "F1det", "F1type"):
            row["peft_" + metric] = p[metric]
            row["fullft_" + metric] = f[metric]
            row["delta_peft_minus_fullft_" + metric] = p[metric] - f[metric]
            row["absolute_delta_" + metric] = abs(p[metric] - f[metric])
        row["relative_mpq_recovery_peft_over_fullft"] = p["mPQ"] / f["mPQ"] if f["mPQ"] else float("nan")
        for prefix, item in (("peft", p), ("fullft", f)):
            for name in ("trainable_params", "total_params", "trainable_pct", "training_wall_seconds",
                         "training_wall_hours", "training_peak_allocated_gib", "inference_peak_allocated_gib",
                         "inference_patches_per_second", "retained_model_state_bytes", "model_state_kind",
                         "source_efficiency_json", "source_inference_efficiency_json"):
                row[prefix + "_" + name] = item[name]
        rows.append(row)
    return rows


def write_matched(rows):
    fields = list(rows[0])
    with MATCHED_CSV.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    lines = [
        "# Tissue-specific Selected PEFT versus FullFT — reciprocal complete-slide protocol",
        "",
        "All rows are seed42 and use canonical checkpoint-10 TEST evidence. Each pair was required to have identical tissue, fold, train slide, held-out slide, and exact TEST patch-ID hash. Deltas are signed PEFT−FullFT; recovery is PEFT mPQ / FullFT mPQ. Resource measurements are run-specific A100 measurements, not KLT efficiency-audit averages.",
        "",
        "| Backbone | Tissue | Fold | N | PEFT bPQ | FullFT bPQ | Δ bPQ | PEFT mPQ | FullFT mPQ | Δ mPQ | mPQ recovery | PEFT F1det | FullFT F1det | Δ F1det | PEFT F1type | FullFT F1type | Δ F1type |",
        "|---|---|:---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for r in rows:
        lines.append("| {backbone} | {tissue} | {fold} | {matched_test_patches} | {peft_bPQ:.6f} | {fullft_bPQ:.6f} | {delta_peft_minus_fullft_bPQ:+.6f} | {peft_mPQ:.6f} | {fullft_mPQ:.6f} | {delta_peft_minus_fullft_mPQ:+.6f} | {relative_mpq_recovery_peft_over_fullft:.4f} | {peft_F1det:.6f} | {fullft_F1det:.6f} | {delta_peft_minus_fullft_F1det:+.6f} | {peft_F1type:.6f} | {fullft_F1type:.6f} | {delta_peft_minus_fullft_F1type:+.6f} |".format(**r))
    samh = [r for r in rows if r["backbone"] == "CellViT-SAM-H"]
    if len(samh) == 18:
        lines.extend(["", "## Descriptive summary over 18 matched SAM-H tissue×fold conditions", "",
                      "These are descriptive condition-level summaries. Tissue×fold conditions are not treated as independent biological replicates; no significance, equivalence, or non-inferiority test is performed.", "",
                      "| Metric | PEFT mean | FullFT mean | Mean absolute delta | PEFT higher (of 18) |",
                      "|---|---:|---:|---:|---:|"])
        for metric in ("bPQ", "mPQ", "F1det", "F1type"):
            lines.append("| {} | {:.6f} | {:.6f} | {:.6f} | {} |".format(
                metric, mean([r["peft_" + metric] for r in samh]),
                mean([r["fullft_" + metric] for r in samh]),
                mean([r["absolute_delta_" + metric] for r in samh]),
                sum(r["peft_" + metric] > r["fullft_" + metric] for r in samh)))
        recovery = sorted(r["relative_mpq_recovery_peft_over_fullft"] for r in samh)
        lines.extend(["", "mPQ recovery PEFT/FullFT: minimum {:.4f}, median {:.4f}, maximum {:.4f}.".format(
            recovery[0], statistics.median(recovery), recovery[-1])])
    lines.extend(["", "## Run-specific resource measurements", "",
                  "| Backbone | Tissue | Fold | Method | Trainable % | Train h | Peak train VRAM GiB | Model state bytes | State kind |",
                  "|---|---|:---:|---|---:|---:|---:|---:|---|"])
    for r in rows:
        for prefix, label in (("peft", "Selected PEFT"), ("fullft", "FullFT")):
            state_bytes = r[prefix + "_retained_model_state_bytes"]
            state_display = "{:,}".format(state_bytes) if state_bytes is not None else "--"
            state_kind = r[prefix + "_model_state_kind"] or "not available"
            lines.append("| {} | {} | {} | {} | {:.6f} | {:.4f} | {:.3f} | {} | {} |".format(
                r["backbone"], r["tissue"], r["fold"], label, r[prefix + "_trainable_pct"],
                r[prefix + "_training_wall_hours"], r[prefix + "_training_peak_allocated_gib"],
                state_display, state_kind))
    lines.extend(["", "## Provenance", "",
                  "Exact inference and efficiency paths, TEST patch-ID hashes, signed and absolute deltas, parameter counts, inference VRAM, and throughput are retained in `reports/tissue_peft_vs_fullft_slideind.csv`.", ""])
    MATCHED_MD.write_text("\n".join(lines))


def write_nine_tissue(peft):
    fields = ["backbone", "method", "tissue", "fold", "seed", "train_slide", "test_slide",
              "test_patches", "test_true_support", "Dice", "bPQ", "mPQ", "F1det", "F1type"]
    for cls in CLASSES:
        fields.extend([cls.lower() + "_F1", cls.lower() + "_matched_support", cls.lower() + "_total_true_support"])
    fields.extend(["dominant_confusion", "dominant_confusion_count", "checkpoint_policy", "source_inference_json"])
    rows = []
    for item in sorted(peft, key=lambda x: (("CellViT-SAM-H", "CellViT-256").index(x["backbone"]), TISSUES.index(x["tissue"]), x["fold"])):
        row = {name: item.get(name, "") for name in fields}
        row.update({"checkpoint_policy": item["checkpoint_policy"], "source_inference_json": rel(item["source"])})
        for cls in CLASSES:
            c = item["per_class"][cls]
            row[cls.lower() + "_F1"] = c["f1"]
            row[cls.lower() + "_matched_support"] = c["matched_support"]
            row[cls.lower() + "_total_true_support"] = c["total_true_support"]
        rows.append(row)
    with NINE_CSV.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    lines = ["# Nine-tissue reciprocal complete-slide Selected PEFT matrix", "",
             "Both backbones are complete for all 9 tissues × Fold A/B at seed42. Support is matched-plus-unmatched true foreground support from canonical TEST inference. Counts are patch-level nucleus occurrences, not unique biological cells.", "",
             "| Backbone | Tissue | Fold | TEST patches | True support | bPQ | mPQ | F1det | F1type | Per-class F1 / support |",
             "|---|---|:---:|---:|---:|---:|---:|---:|---:|---|"]
    for item, row in zip(sorted(peft, key=lambda x: (("CellViT-SAM-H", "CellViT-256").index(x["backbone"]), TISSUES.index(x["tissue"]), x["fold"])), rows):
        classes = "; ".join("{} {:.3f}/{:,}".format(cls, row[cls.lower() + "_F1"], row[cls.lower() + "_total_true_support"]) for cls in CLASSES)
        lines.append("| {} | {} | {} | {:,} | {:,} | {:.6f} | {:.6f} | {:.6f} | {:.6f} | {} |".format(
            item["backbone"], item["tissue"], item["fold"], item["test_patches"], item["test_true_support"],
            item["bPQ"], item["mPQ"], item["F1det"], item["F1type"], classes))
    lines.extend(["", "Matched FullFT controls cover all nine tissues for CellViT-SAM-H. CellViT-256 FullFT remains restricted to Kidney/Liver/Tonsil; no comparison is implied for its other six tissues.", ""])
    NINE_MD.write_text("\n".join(lines))


def discover_klt_source(backbone, method, fold, seed):
    cv = "_cellvit256" if backbone == "CellViT-256" else ""
    if method == "LP":
        name = "sthelar40x_klt_5class_slideind_fold{}{}_lp_final_heads_e10_seed{}".format(fold, cv, seed)
    elif method == "Selected PEFT":
        name = "sthelar40x_klt_5class_slideind_fold{}{}_lora_adaptformer_r8_a8_red16_heads_e10_seed{}".format(fold, cv, seed)
    elif method == "FullFT":
        name = "sthelar40x_klt_5class_slideind_fold{}{}_fullft_lr1e-5_e10_seed{}".format(fold, cv, seed)
    else:
        return None
    candidates = sorted((REPO / "run" / name / "log").glob("*/inference_results.json"))
    expected = int(load_manifest(klt_manifest_path(fold))["patch_counts"]["test"])
    good = [path for path in candidates if verify_checkpoint10(path, expected)[0]]
    return good[-1] if good else None


def fill_metric_row(row, source, method):
    agg = aggregate_inference(source)
    resources = load_resources(source, method)
    digest, ids = patch_id_hash(source)
    manifest = load_manifest(klt_manifest_path(row["fold"])) if row["protocol"] == "held_out_slide_KLT" else load_manifest(tissue_manifest_path(row["tissue"], row["fold"]))
    row.update({
        "status": "COMPLETED", "checkpoint_policy": "checkpoint_10 TEST",
        "Dice": nfmt(agg["Dice"]), "Jaccard": nfmt(agg["Jaccard"]), "bPQ": nfmt(agg["bPQ"]),
        "mPQ": nfmt(agg["mPQ"]), "F1det": nfmt(agg["F1det"]), "F1type": nfmt(agg["F1type"]),
        "trainable_params": str(resources["trainable_params"]), "total_params": str(resources["total_params"]),
        "trainable_pct": nfmt(resources["trainable_pct"]), "test_patches": str(len(ids)),
        "test_true_support": str(sum(agg["per_class"][c]["total_true_support"] for c in CLASSES)),
        "train_slide": ";".join(manifest["train_slide_ids"]), "test_slide": ";".join(manifest["test_slide_ids"]),
        "test_patch_id_sha256": digest, "training_wall_seconds": nfmt(resources["training_wall_seconds"], 6),
        "training_wall_hours": nfmt(resources["training_wall_hours"], 6),
        "training_peak_allocated_gib": nfmt(resources["training_peak_allocated_gib"], 6),
        "inference_peak_allocated_gib": nfmt(resources["inference_peak_allocated_gib"], 6),
        "inference_patches_per_second": nfmt(resources["inference_patches_per_second"], 6),
        "retained_model_state_bytes": str(resources["retained_model_state_bytes"] or ""),
        "model_state_kind": resources["model_state_kind"] or "", "source_efficiency_json": resources["source_efficiency_json"],
        "source_inference_efficiency_json": resources["source_inference_efficiency_json"],
        "source_inference_json": rel(source), "provenance_note": "Canonical completion re-established from checkpoint-10 TEST artifacts on {}.".format(SNAPSHOT),
    })
    for cls in CLASSES:
        prefix = cls.lower()
        c = agg["per_class"][cls]
        row[prefix + "_F1"] = nfmt(c["f1"])
        row[prefix + "_matched_support"] = str(c["matched_support"])
        row[prefix + "_total_true_support"] = str(c["total_true_support"])
    return row


def update_master_csv(peft, fullft):
    with MASTER_CSV.open(newline="") as handle:
        reader = csv.DictReader(handle)
        old_fields = reader.fieldnames
        rows = list(reader)
    rows = [r for r in rows if r["protocol"] != "held_out_slide_KLT_aggregate"]
    # Rebuild the matched tissue FullFT block from canonical sources on every
    # invocation.  This keeps the report generation idempotent when the master
    # CSV already contains rows written by an earlier consolidation pass.
    rows = [r for r in rows if not (
        r["protocol"] == "tissue_specific_held_out_slide"
        and r["method"] == "FullFT"
        and r["backbone"] in ("CellViT-SAM-H", "CellViT-256")
        and r["tissue"] in TISSUES
    )]
    tissue_map = {(x["backbone"], x["method"], x["tissue"], x["fold"]): x for x in peft + fullft}
    existing_tissue_keys = set()
    for row in rows:
        if row["protocol"] == "held_out_slide_KLT" and row["method"] in ("LP", "Selected PEFT", "FullFT"):
            source = discover_klt_source(row["backbone"], row["method"], row["fold"], int(row["seed"]))
            if source:
                fill_metric_row(row, source, row["method"])
            elif row["canonical_job_id"] in LIVE_STATUS:
                row["status"] = LIVE_STATUS[row["canonical_job_id"]]
                row["source_inference_json"] = ""
                row["provenance_note"] = "Read-only Slurm snapshot {}: {}; metrics intentionally blank.".format(SNAPSHOT, row["status"])
                for key in ("Dice", "Jaccard", "bPQ", "mPQ", "F1det", "F1type"):
                    row[key] = ""
        if row["protocol"] == "tissue_specific_held_out_slide" and row["method"] == "Selected PEFT":
            key = (row["backbone"], row["method"], row["tissue"], row["fold"])
            if key in tissue_map:
                fill_metric_row(row, tissue_map[key]["source"], row["method"])
                existing_tissue_keys.add(key)

    for item in fullft:
        key = (item["backbone"], item["method"], item["tissue"], item["fold"])
        row = {field: "" for field in old_fields + EXTRA_MASTER_FIELDS}
        row.update({"protocol": "tissue_specific_held_out_slide", "backbone": item["backbone"],
                    "method": "FullFT", "domain": item["tissue"], "tissue": item["tissue"], "seed": "42",
                    "fold": item["fold"], "canonical_job_id": FULLFT_JOB_IDS[(item["backbone"], item["tissue"], item["fold"])]})
        fill_metric_row(row, item["source"], "FullFT")
        rows.append(row)

    individual = [r for r in rows if r["protocol"] == "held_out_slide_KLT"]
    for backbone in ("CellViT-SAM-H", "CellViT-256"):
        for method in ("Frozen", "LP", "Selected PEFT", "FullFT"):
            planned = 1 if method == "Frozen" else 3
            for fold in FOLDS:
                complete = [r for r in individual if r["backbone"] == backbone and r["method"] == method
                            and r["fold"] == fold and r["status"] == "COMPLETED"]
                if not complete:
                    continue
                metrics = ["Dice", "Jaccard", "bPQ", "mPQ", "F1det", "F1type"]
                for stat in ("MEAN", "SD"):
                    if stat == "SD" and len(complete) < 2:
                        continue
                    base = complete[0]
                    row = {field: "" for field in old_fields + EXTRA_MASTER_FIELDS}
                    row.update({"protocol": "held_out_slide_KLT_aggregate", "backbone": backbone, "method": method,
                                "domain": "KLT", "tissue": "Kidney+Liver+Tonsil", "seed": stat, "fold": fold,
                                "status": "COMPLETE" if len(complete) == planned else "INCOMPLETE_PENDING_SEEDS",
                                "checkpoint_policy": "derived from checkpoint_10 TEST rows",
                                "trainable_params": base.get("trainable_params", ""), "total_params": base.get("total_params", ""),
                                "trainable_pct": base.get("trainable_pct", ""),
                                "provenance_note": "Derived within Fold {} from completed seeds {}; n={}/{}.".format(
                                    fold, ",".join(r["seed"] for r in complete), len(complete), planned)})
                    for metric in metrics:
                        vals = [float(r[metric]) for r in complete if r.get(metric) not in ("", None)]
                        if vals:
                            value = mean(vals) if stat == "MEAN" else sample_sd(vals)
                            row[metric] = nfmt(value)
                    rows.append(row)

    fields = list(old_fields)
    for field in EXTRA_MASTER_FIELDS:
        if field not in fields:
            fields.append(field)
    with MASTER_CSV.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
    return rows, fields


def klt_summaries(master_rows):
    summaries = {}
    for row in master_rows:
        if row["protocol"] == "held_out_slide_KLT_aggregate" and row["seed"] in ("MEAN", "SD"):
            summaries[(row["backbone"], row["method"], row["fold"], row["seed"])] = row
    return summaries


def display_agg(summaries, backbone, method, fold, metric):
    mean_row = summaries[(backbone, method, fold, "MEAN")]
    value = float(mean_row[metric]) if mean_row.get(metric) else None
    sd_row = summaries.get((backbone, method, fold, "SD"))
    if value is None:
        return "--"
    if sd_row and sd_row.get(metric):
        return "{:.3f} ± {:.3f}".format(value, float(sd_row[metric]))
    return "{:.3f}".format(value)


def write_master_md(master_rows, matched, peft):
    summaries = klt_summaries(master_rows)
    lines = ["# AUTHORITATIVE PAPER RESULTS — use this file for LaTeX values.", "",
             "Consolidated read-only on 2026-09-01 from canonical checkpoint-10 TEST artifacts. Running LP seed44 rows remain non-results. Fold A and Fold B are reciprocal complete-slide holdouts and are never pooled as n=6.", "",
             "## Completion inventory", "",
             "| Family | Canonical complete | Current boundary |", "|---|---:|---|",
             "| KLT SAM-H LP | 4/6 seed×fold rows | seed44 A/B RUNNING |",
             "| KLT CellViT-256 LP | 4/6 seed×fold rows | seed44 A/B RUNNING |",
             "| KLT Selected PEFT | 12/12 across both backbones | three seeds per fold |",
             "| KLT FullFT | 12/12 across both backbones | three seeds per fold |",
             "| Tissue-specific Selected PEFT | 36/36 | 9 tissues × 2 folds × 2 backbones |",
             "| Matched tissue-specific FullFT | 24/24 | SAM-H nine tissues A/B; CellViT-256 Kidney/Liver/Tonsil A/B |", "",
             "Provenance: individual canonical rows and scheduler snapshot fields are retained in `reports/workshop_master_results.csv`.", "",
             "## KLT fold-specific seed master", "",
             "Values are mean ± sample SD across currently complete stochastic seeds within each fold. Frozen is deterministic. LP is n=2/3 while seed44 is running.", "",
             "| Backbone | Method | Fold | Seeds | bPQ | mPQ | F1det | F1type |",
             "|---|---|:---:|:---:|---:|---:|---:|---:|"]
    for backbone in ("CellViT-SAM-H", "CellViT-256"):
        for method in ("Frozen", "LP", "Selected PEFT", "FullFT"):
            for fold in FOLDS:
                mean_row = summaries[(backbone, method, fold, "MEAN")]
                completed = len([r for r in master_rows if r["protocol"] == "held_out_slide_KLT" and r["backbone"] == backbone
                                 and r["method"] == method and r["fold"] == fold and r["status"] == "COMPLETED"])
                planned = 1 if method == "Frozen" else 3
                lines.append("| {} | {} | {} | {}/{} | {} | {} | {} | {} |".format(
                    backbone, method, fold, completed, planned,
                    display_agg(summaries, backbone, method, fold, "bPQ"),
                    display_agg(summaries, backbone, method, fold, "mPQ"),
                    display_agg(summaries, backbone, method, fold, "F1det"),
                    display_agg(summaries, backbone, method, fold, "F1type")))
    lines.extend(["", "Provenance: `reports/workshop_master_results.csv` retains every individual seed row, canonical job ID, checkpoint policy, and source `inference_results.json` path.", "",
                  "## Matched tissue-specific Selected PEFT versus FullFT", "",
                  "Every comparison below has an identical train slide, held-out slide, seed, fold, and TEST patch-ID set.", "",
                  "| Backbone | Tissue | Fold | N | PEFT bPQ | FullFT bPQ | PEFT mPQ | FullFT mPQ | ΔmPQ | Recovery | PEFT F1det | FullFT F1det | PEFT F1type | FullFT F1type |",
                  "|---|---|:---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|"])
    for r in matched:
        lines.append("| {backbone} | {tissue} | {fold} | {matched_test_patches} | {peft_bPQ:.4f} | {fullft_bPQ:.4f} | {peft_mPQ:.4f} | {fullft_mPQ:.4f} | {delta_peft_minus_fullft_mPQ:+.4f} | {relative_mpq_recovery_peft_over_fullft:.3f} | {peft_F1det:.4f} | {fullft_F1det:.4f} | {peft_F1type:.4f} | {fullft_F1type:.4f} |".format(**r))
    lines.extend(["", "Provenance: exact paired source paths, TEST patch-ID hashes, resource JSONs, and all metric deltas are in `reports/tissue_peft_vs_fullft_slideind.csv`.", "",
                  "## Nine-tissue Selected PEFT matrices", "",
                  "The complete numerical and per-class matrices are in `reports/nine_tissue_peft_slideind.md` and `.csv`. Matched FullFT covers all nine SAM-H tissues and Kidney/Liver/Tonsil for CellViT-256.", "",
                  "| Backbone | Directions complete | Reciprocal tissues complete |", "|---|---:|---:|",
                  "| CellViT-SAM-H | 18/18 | 9/9 |", "| CellViT-256 | 18/18 | 9/9 |", "",
                  "Provenance: `reports/nine_tissue_peft_slideind.csv` contains each canonical TEST JSON path and all per-class supports/F1 values.", "",
                  "## Matched computational context", "",
                  "The canonical seed42 A100 audit remains separate from predictive seed aggregation. SAM-H Selected PEFT trains 1.1176% of parameters with 9.362 GiB peak allocated VRAM versus FullFT 100% and 15.508 GiB; its verified adapter is about 31.8 MB versus a measured 8.40 GB FullFT training checkpoint. CellViT-256 Selected PEFT trains 0.7942% with 2.040 GiB versus FullFT 2.667 GiB; its verified adapter is about 1.61 MB versus a 561 MB FullFT checkpoint.", "",
                  "Sources: `reports/efficiency_audit/paper_ready_efficiency.csv`, run-specific efficiency JSONs in `reports/tissue_peft_vs_fullft_slideind.csv`, and verified adapter retention metadata.", "",
                  "## Historical within-slide ablation uncertainty", "",
                  "`reports/workshop_ablation_seed_audit.md` shows heterogeneous replication: only five of eleven rows have two comparable canonical seeds; the remaining six are single-seed point estimates, and Frozen has incompatible typing taxonomy. Uncertainty cannot safely be added uniformly to the main ablation table. Mean ± sample SD is defensible only for the five n=2 rows, with explicit row-wise n.", "",
                  "## PAPER-SAFE FINDINGS", "",
                  "- Under complete-slide KLT, Selected PEFT tracks FullFT closely in bPQ/F1det; typing-sensitive mPQ/F1type differences are more direction- and backbone-dependent.",
                  "- The lightweight CellViT-256 backbone shows a larger KLT bPQ gap between Selected PEFT and FullFT than SAM-H, while both retain similar F1det.",
                  "- Across the complete nine-tissue SAM-H matched block, PEFT sometimes approaches or exceeds FullFT on individual metrics/directions, but the direction-dependent pattern does not support equivalence or superiority.",
                  "- F1type and mPQ are more sensitive than F1det to reciprocal slide/tissue changes; detection is descriptively more stable.",
                  "- Fold/tissue variation is descriptively larger than seed variation for several typing metrics, but folds are different held-out slides rather than stochastic replicates.",
                  "- Class-frequency TVD/JSD, rare classes, density changes, and dominant confusion pairs align descriptively with some typing failures; they do not establish causality.",
                  "- Measured adapters provide substantial trainable-parameter and deployable-storage reductions; the matched A100 audit must remain separate from multi-seed predictive aggregation.",
                  "- NOT SUPPORTED: statistical superiority/equivalence/non-inferiority, patient-independent replication, causal explanations, or extrapolation of nine-tissue FullFT evidence to CellViT-256.", ""])
    MASTER_MD.write_text("\n".join(lines))


def latex_value(summaries, backbone, method, fold, metric):
    text = display_agg(summaries, backbone, method, fold, metric)
    return text.replace("±", r"$\pm$").replace("--", "--")


def tex_escape(text):
    return str(text).replace("%", r"\%").replace("_", r"\_")


def write_latex(master_rows, matched, peft, ablation_summaries):
    summaries = klt_summaries(master_rows)
    lines = ["% Candidate tables generated read-only from canonical checkpoint-10 TEST artifacts on 2026-09-01.",
             "% Do not overwrite the manuscript without review.",
             r"\begin{table*}[t]", r"\centering",
             r"\caption{Held-out-slide KLT results. Mean $\pm$ sample SD is computed within fold over currently complete seeds; LP is n=2/3 because seed44 is running.}",
             r"\label{tab:workshop-klt-master}", r"\scriptsize", r"\begin{tabular}{lllccccc}", r"\toprule",
             "Backbone & Method & Fold & bPQ & mPQ & $F1_{det}$ & $F1_{type}$ & Seeds " + r"\\", r"\midrule"]
    for backbone in ("CellViT-SAM-H", "CellViT-256"):
        for method in ("Frozen", "LP", "Selected PEFT", "FullFT"):
            for fold in FOLDS:
                completed = len([r for r in master_rows if r["protocol"] == "held_out_slide_KLT" and r["backbone"] == backbone and r["method"] == method and r["fold"] == fold and r["status"] == "COMPLETED"])
                planned = 1 if method == "Frozen" else 3
                lines.append("{} & {} & {} & {} & {} & {} & {} & {}/{} \\\\".format(
                    backbone.replace("CellViT-", ""), method, fold,
                    latex_value(summaries, backbone, method, fold, "bPQ"), latex_value(summaries, backbone, method, fold, "mPQ"),
                    latex_value(summaries, backbone, method, fold, "F1det"), latex_value(summaries, backbone, method, fold, "F1type"), completed, planned))
    lines.extend([r"\bottomrule", r"\end{tabular}", r"\end{table*}", "",
                  r"\begin{table*}[t]", r"\centering",
                  r"\caption{Matched tissue-specific Selected PEFT versus FullFT under reciprocal complete-slide holdout (seed42). $\Delta$ is PEFT minus FullFT.}",
                  r"\label{tab:tissue-peft-fullft}", r"\scriptsize", r"\begin{tabular}{lllrrrrrrrr}", r"\toprule",
                  "Backbone & Tissue & Fold & N & PEFT bPQ & FullFT bPQ & PEFT mPQ & FullFT mPQ & $\\Delta$mPQ & PEFT $F1_{det}$ & FullFT $F1_{det}$ " + r"\\", r"\midrule"])
    for r in matched:
        lines.append("{} & {} & {} & {} & {:.4f} & {:.4f} & {:.4f} & {:.4f} & {:+.4f} & {:.4f} & {:.4f} \\\\".format(
            r["backbone"].replace("CellViT-", ""), r["tissue"], r["fold"], r["matched_test_patches"],
            r["peft_bPQ"], r["fullft_bPQ"], r["peft_mPQ"], r["fullft_mPQ"],
            r["delta_peft_minus_fullft_mPQ"], r["peft_F1det"], r["fullft_F1det"]))
    lines.extend([r"\bottomrule", r"\end{tabular}", r"\end{table*}", ""])

    # The A100 audit is deliberately a seed42 matched measurement table, not a
    # predictive three-seed aggregation.  Read the already frozen audit CSV so
    # the LaTeX candidate remains traceable to the same source as the report.
    efficiency_path = REPORT_ROOT / "efficiency_audit" / "paper_ready_efficiency.csv"
    with efficiency_path.open(newline="") as handle:
        efficiency = list(csv.DictReader(handle))
    lines.extend([r"\begin{table*}[t]", r"\centering",
                  r"\caption{Matched A100 computational audit (seed42; Fold A/B arithmetic means). Predictive multi-seed results are reported separately.}",
                  r"\label{tab:matched-computational-audit}", r"\scriptsize",
                  r"\begin{tabular}{llrrrrrrrrr}", r"\toprule",
                  "Backbone & Method & Total params & Trainable & Trainable \\% & Train h & Train VRAM GiB & Train ckpt GiB & Adapter MiB & Infer VRAM GiB & Patches/s " + r"\\", r"\midrule"])
    order = {"Frozen": 0, "LP": 1, "Selected PEFT": 2, "FullFT": 3}
    for row in sorted(efficiency, key=lambda x: ((0 if x["backbone"] == "SAM-H" else 1), order[x["method"]])):
        def shown(key, digits=3):
            return "--" if not row[key] else ("{:,.{}f}".format(float(row[key]), digits))
        lines.append("{} & {} & {:,} & {:,} & {} & {} & {} & {} & {} & {} & {} \\\\".format(
            row["backbone"], row["method"], int(row["total_parameters"]), int(row["trainable_parameters"]),
            shown("trainable_percent", 4), shown("training_wall_hours_mean", 3),
            shown("training_peak_allocated_gib_mean", 3), shown("current_training_checkpoint_gib_mean", 3),
            shown("adapter_only_weight_mib", 3), shown("inference_peak_allocated_gib_mean", 3),
            shown("inference_patches_per_second_mean", 3)))
    lines.extend([r"\bottomrule", r"\end{tabular}", r"\end{table*}", ""])

    deployment_path = REPORT_ROOT / "efficiency_audit" / "performance_vs_deployment_storage.csv"
    with deployment_path.open(newline="") as handle:
        deployment_rows = list(csv.DictReader(handle))
    deployment = {}
    for row in deployment_rows:
        deployment.setdefault((row["strategy"], row["backbone"], row["method"]), {})[int(row["N_domains"])] = float(row["total_deployment_gib"])
    lines.extend([r"\begin{table}[t]", r"\centering",
                  r"\caption{Measured deployment storage with shared bases for PEFT (seed42 audit).}",
                  r"\label{tab:deployment-storage}", r"\scriptsize", r"\begin{tabular}{lllrrrr}", r"\toprule",
                  "Strategy & Backbone & Method & N=1 GiB & N=3 GiB & N=6 GiB & N=9 GiB " + r"\\", r"\midrule"])
    for key in sorted(deployment, key=lambda x: x[0]):
        values = deployment[key]
        lines.append("{} & {} & {} & {:.3f} & {:.3f} & {:.3f} & {:.3f} \\\\".format(
            key[0], key[1], key[2], values[1], values[3], values[6], values[9]))
    lines.extend([r"\bottomrule", r"\end{tabular}", r"\end{table}", ""])
    for backbone in ("CellViT-SAM-H", "CellViT-256"):
        lines.extend([r"\begin{table*}[t]", r"\centering",
                      r"\caption{Nine-tissue Selected PEFT reciprocal complete-slide TEST matrix for " + tex_escape(backbone) + r" (seed42).}",
                      r"\label{tab:nine-tissue-" + ("samh" if backbone.endswith("SAM-H") else "cv256") + "}",
                      r"\scriptsize", r"\begin{tabular}{llrrrr}", r"\toprule",
                      "Tissue & Fold & bPQ & mPQ & $F1_{det}$ & $F1_{type}$ " + r"\\", r"\midrule"])
        for r in sorted([x for x in peft if x["backbone"] == backbone], key=lambda x: (TISSUES.index(x["tissue"]), x["fold"])):
            lines.append("{} & {} & {:.4f} & {:.4f} & {:.4f} & {:.4f} \\\\".format(
                r["tissue"], r["fold"], r["bPQ"], r["mPQ"], r["F1det"], r["F1type"]))
        lines.extend([r"\bottomrule", r"\end{tabular}", r"\end{table*}", ""])
    lines.extend([r"\begin{table*}[t]", r"\centering",
                  r"\caption{Historical within-slide KLT ablation seed audit. Uncertainty is shown only where at least two comparable checkpoint-10 TEST seeds exist.}",
                  r"\label{tab:historical-ablation-seeds}", r"\scriptsize", r"\begin{tabular}{llrcccc}", r"\toprule",
                  "Method & Decoder & n & Dice & bPQ & mPQ & $F1_{det}$ " + r"\\", r"\midrule"])
    for s in ablation_summaries:
        def ms(metric):
            value = s[metric + "_mean"]
            if not finite(value):
                return "--"
            if s["n"] >= 2:
                return "{:.4f} $\\pm$ {:.4f}".format(value, s[metric + "_sd"])
            return "{:.4f}".format(value)
        lines.append("{} & {} & {} & {} & {} & {} & {} \\\\".format(
            tex_escape(s["method"]), tex_escape(s["decoder_scope"]), s["n"], ms("Dice"), ms("bPQ"), ms("mPQ"), ms("F1det")))
    lines.extend([r"\bottomrule", r"\end{tabular}", r"\end{table*}", ""])
    LATEX.write_text("\n".join(lines))


def main():
    peft, fullft = build_tissue_records()
    if len(peft) != 36 or len(fullft) != 24:
        raise RuntimeError("Unexpected tissue matrix sizes: PEFT {}, FullFT {}".format(len(peft), len(fullft)))
    matched = build_matched(peft, fullft)
    write_matched(matched)
    write_nine_tissue(peft)
    master_rows, _ = update_master_csv(peft, fullft)
    _, ablation_summaries, _ = build_ablation_audit()
    write_master_md(master_rows, matched, peft)
    write_latex(master_rows, matched, peft, ablation_summaries)
    for path in (MASTER_MD, MASTER_CSV, LATEX, MATCHED_MD, MATCHED_CSV, NINE_MD, NINE_CSV):
        print(rel(path))


if __name__ == "__main__":
    main()
