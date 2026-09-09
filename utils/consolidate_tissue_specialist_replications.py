#!/usr/bin/env python3
"""Consolidate verified SAM-H tissue-specialist seed43/44 replications.

The script is intentionally fail-closed.  It updates paper-facing analyses
only after all twelve runs have canonical checkpoint-10 TEST evidence and a
bitwise-verified adapter whose redundant full checkpoint was safely deleted.
It never launches jobs and never edits run artifacts.
"""

import csv
import json
import math
import re
import statistics
from pathlib import Path

import yaml

from create_tissue_specialist_failure_analysis import (
    ALL_CLASSES,
    CLASSES,
    CSV_FIELDS,
    REPO,
    aggregate_inference,
    main as rebuild_seed42_analysis,
)


MASTER_CSV = REPO / "reports" / "workshop_master_results.csv"
MASTER_MD = REPO / "reports" / "workshop_master_results.md"
LATEX = REPO / "reports" / "workshop_latex_tables.tex"
ANALYSIS_CSV = REPO / "reports" / "tissue_specialist_failure_analysis.csv"
ANALYSIS_MD = REPO / "reports" / "tissue_specialist_failure_analysis.md"

TISSUES = ("Kidney", "Liver", "Tonsil")
FOLDS = ("A", "B")
SEEDS = (42, 43, 44)
NEW_SEEDS = (43, 44)
METRICS = ("Dice", "bPQ", "mPQ", "F1det", "F1type")
MODELS = ("Generalist Selected PEFT", "Tissue-specific Selected PEFT")
JOB_IDS = {
    ("Kidney", "A", 43): "1581904",
    ("Kidney", "B", 43): "1581905",
    ("Liver", "A", 43): "1581906",
    ("Liver", "B", 43): "1581907",
    ("Tonsil", "A", 43): "1581908",
    ("Tonsil", "B", 43): "1581909",
    ("Kidney", "A", 44): "1581910",
    ("Kidney", "B", 44): "1581911",
    ("Liver", "A", 44): "1581912",
    ("Liver", "B", 44): "1581913",
    ("Tonsil", "A", 44): "1581914",
    ("Tonsil", "B", 44): "1581915",
}


def rel(path: Path) -> str:
    return str(path.resolve().relative_to(REPO.resolve()))


def run_name(tissue: str, fold: str, seed: int) -> str:
    return (
        f"sthelar40x_{tissue.lower()}_5class_slideind_fold{fold}_"
        f"lora_adaptformer_r8_a8_red16_heads_e10_seed{seed}"
    )


def load_master():
    with MASTER_CSV.open(newline="") as handle:
        reader = csv.DictReader(handle)
        return list(reader.fieldnames or []), list(reader)


def master_generalist_sources(rows):
    sources = {}
    for row in rows:
        if (
            row["protocol"] == "held_out_slide_KLT"
            and row["backbone"] == "CellViT-SAM-H"
            and row["method"] == "Selected PEFT"
            and row["status"] == "COMPLETED"
            and int(row["seed"]) in SEEDS
        ):
            sources[(row["fold"], int(row["seed"]))] = REPO / row["source_inference_json"]
    expected = {(fold, seed) for fold in FOLDS for seed in SEEDS}
    if set(sources) != expected:
        raise RuntimeError(f"Incomplete generalist source matrix: {sorted(expected - set(sources))}")
    return sources


def seed42_specialist_sources(rows):
    sources = {}
    for row in rows:
        if (
            row["protocol"] == "composition_matched_generalist_vs_specialist"
            and row["backbone"] == "CellViT-SAM-H"
            and row["method"] == "Tissue-specific Selected PEFT"
            and row["seed"] == "42"
        ):
            sources[(row["tissue"], row["fold"])] = REPO / row["source_inference_json"]
    expected = {(tissue, fold) for tissue in TISSUES for fold in FOLDS}
    if set(sources) != expected:
        raise RuntimeError(f"Incomplete seed42 specialist source matrix: {sorted(expected - set(sources))}")
    return sources


def verified_new_source(tissue: str, fold: str, seed: int) -> Path:
    name = run_name(tissue, fold, seed)
    config_path = REPO / "configs" / "slide_exp" / "training" / (
        f"training_sthelar40x_{tissue.lower()}_5class_slideind_fold{fold}_"
        f"lora_adaptformer_heads_seed{seed}.yaml"
    )
    config = yaml.safe_load(config_path.read_text())
    if config.get("random_seed") != seed or config.get("eval_checkpoint") != "checkpoint_10.pth":
        raise RuntimeError(f"Noncanonical config: {config_path}")
    candidates = []
    log_root = REPO / config["logging"]["log_dir"]
    for candidate in log_root.glob(f"*_{name}"):
        required = (
            "config.yaml",
            "inference_results.json",
            "efficiency_metrics.json",
            "inference_efficiency_metrics.json",
            "checkpoint_retention_metadata.json",
            "adapter_checkpoint_retention_metadata.json",
            "logs.log",
        )
        if all((candidate / item).is_file() for item in required):
            candidates.append(candidate)
    if len(candidates) != 1:
        raise RuntimeError(f"Expected exactly one complete run for {name}; found {candidates}")
    run_dir = candidates[0]
    generated = yaml.safe_load((run_dir / "config.yaml").read_text())
    if generated.get("random_seed") != seed or generated.get("logging", {}).get("log_comment") != name:
        raise RuntimeError(f"Run/config identity mismatch: {run_dir}")
    checkpoint_meta = json.loads((run_dir / "checkpoint_retention_metadata.json").read_text())
    if not (
        checkpoint_meta.get("status") == "retention_complete"
        and checkpoint_meta.get("training_completed") is True
        and checkpoint_meta.get("inference_completed") is True
        and checkpoint_meta.get("inference_used_checkpoint_10") is True
    ):
        raise RuntimeError(f"Checkpoint-10 TEST retention gate failed: {run_dir}")
    retention = json.loads((run_dir / "adapter_checkpoint_retention_metadata.json").read_text())
    if not (
        retention.get("status") == "adapter_verified_checkpoint_deleted"
        and retention.get("state_reconstruction") == "exact_all_tensors"
        and retention.get("forward_verification") == "exact_all_output_tensors"
        and retention.get("checkpoint_exists_after_retention") is False
    ):
        raise RuntimeError(f"Adapter retention gate failed: {run_dir}")
    checkpoint = run_dir / "checkpoints" / "checkpoint_10.pth"
    if checkpoint.exists():
        raise RuntimeError(f"Guarded checkpoint deletion was not completed: {checkpoint}")
    adapter_dir = REPO / "adapters" / "slide_exp_safetensors" / name
    verification = json.loads((adapter_dir / "verification.json").read_text())
    comparisons = verification.get("forward_output_comparison", [])
    if not (
        verification.get("status") == "ok"
        and verification.get("state_reconstruction") == "exact_all_tensors"
        and verification.get("forward_verification") == "exact_all_output_tensors"
        and len(comparisons) == 4
        and all(row.get("exact") is True and row.get("max_abs_difference") == 0.0 for row in comparisons)
        and (adapter_dir / "adapter_model.safetensors").is_file()
    ):
        raise RuntimeError(f"Adapter verification gate failed: {adapter_dir}")
    return run_dir / "inference_results.json"


def build_results(rows):
    general_sources = master_generalist_sources(rows)
    seed42_sources = seed42_specialist_sources(rows)
    specialist_sources = {
        (tissue, fold, 42): source
        for (tissue, fold), source in seed42_sources.items()
    }
    for tissue in TISSUES:
        for fold in FOLDS:
            for seed in NEW_SEEDS:
                specialist_sources[(tissue, fold, seed)] = verified_new_source(tissue, fold, seed)

    results = {}
    for tissue in TISSUES:
        for fold in FOLDS:
            for seed in SEEDS:
                general = general_sources[(fold, seed)]
                specialist = specialist_sources[(tissue, fold, seed)]
                general_json = json.loads(general.read_text())
                specialist_json = json.loads(specialist.read_text())
                names = sorted(set(general_json["image_metrics"]) & set(specialist_json["image_metrics"]))
                if not names:
                    raise RuntimeError(f"Empty patch intersection: {tissue} {fold} seed{seed}")
                results[(tissue, fold, seed, MODELS[0])] = {
                    "source": general,
                    "stats": aggregate_inference(general, names),
                    "matched": len(names),
                    "raw": len(general_json["image_metrics"]),
                }
                results[(tissue, fold, seed, MODELS[1])] = {
                    "source": specialist,
                    "stats": aggregate_inference(specialist, names),
                    "matched": len(names),
                    "raw": len(specialist_json["image_metrics"]),
                }
                for name in names:
                    left = general_json["image_metrics"][name]["detection_stats"]
                    right = specialist_json["image_metrics"][name]["detection_stats"]
                    if left.get("true_counts") != right.get("true_counts"):
                        # Older JSONs do not always save true_counts; the seed42
                        # generator independently verifies class-count equality.
                        if left.get("true_counts") is not None or right.get("true_counts") is not None:
                            raise RuntimeError(f"Matched truth mismatch: {tissue} {fold} seed{seed} {name}")
    return results


def mean_sd(values):
    values = [float(value) for value in values]
    return statistics.mean(values), statistics.stdev(values)


def pretty(value: float, digits: int = 3) -> str:
    return f"{value:.{digits}f}"


def pretty_agg(values, latex=False):
    mean, sd = mean_sd(values)
    token = r" $\pm$ " if latex else " ± "
    return pretty(mean) + token + pretty(sd)


def metric_values(results, tissue, fold, model, metric):
    return [results[(tissue, fold, seed, model)]["stats"][metric] for seed in SEEDS]


def three_seed_markdown(results) -> str:
    lines = [
        "<!-- THREE_SEED_ANALYSIS_START -->",
        "## Three-seed composition-matched analysis",
        "",
        "All values below are recomputed on the exact generalist/specialist patch-ID intersection within each tissue, fold and seed. Means and sample SDs use seeds 42/43/44 within a Fold only; Fold A and Fold B are never pooled. The per-seed differences are paired descriptively by optimization seed, but no significance, superiority, equivalence or non-inferiority claim is made.",
        "",
        "| Tissue | Fold | Matched patches/seed | Model | Dice | bPQ | mPQ | F1det | F1type |",
        "|---|:---:|---:|---|---:|---:|---:|---:|---:|",
    ]
    for tissue in TISSUES:
        for fold in FOLDS:
            matched = {results[(tissue, fold, seed, MODELS[0])]["matched"] for seed in SEEDS}
            if len(matched) != 1:
                raise RuntimeError(f"Patch intersection changed across seeds: {tissue} {fold} {matched}")
            for model in MODELS:
                lines.append(
                    "| {} | {} | {:,} | {} | {} | {} | {} | {} | {} |".format(
                        tissue,
                        fold,
                        next(iter(matched)),
                        "Generalist" if model == MODELS[0] else "Specialist",
                        *[pretty_agg(metric_values(results, tissue, fold, model, metric)) for metric in METRICS],
                    )
                )
    lines.extend([
        "",
        "### Paired specialist − generalist differences by seed",
        "",
        "| Tissue | Fold | Seed | ΔDice | ΔbPQ | ΔmPQ | ΔF1det | ΔF1type |",
        "|---|:---:|---:|---:|---:|---:|---:|---:|",
    ])
    for tissue in TISSUES:
        for fold in FOLDS:
            for seed in SEEDS:
                general = results[(tissue, fold, seed, MODELS[0])]["stats"]
                specialist = results[(tissue, fold, seed, MODELS[1])]["stats"]
                deltas = [specialist[metric] - general[metric] for metric in METRICS]
                lines.append(
                    "| {} | {} | {} | {} | {} | {} | {} | {} |".format(
                        tissue, fold, seed, *[f"{value:+.4f}" for value in deltas]
                    )
                )
    lines.extend(["", "### Descriptive robustness audit", ""])
    for tissue in TISSUES:
        by_fold = {}
        for fold in FOLDS:
            by_fold[fold] = {
                metric: [
                    results[(tissue, fold, seed, MODELS[1])]["stats"][metric]
                    - results[(tissue, fold, seed, MODELS[0])]["stats"][metric]
                    for seed in SEEDS
                ]
                for metric in METRICS
            }
        lines.append(
            f"- **{tissue}:** Fold A mean ΔF1det {statistics.mean(by_fold['A']['F1det']):+.4f}, "
            f"mean ΔF1type {statistics.mean(by_fold['A']['F1type']):+.4f}; Fold B mean ΔF1det "
            f"{statistics.mean(by_fold['B']['F1det']):+.4f}, mean ΔF1type "
            f"{statistics.mean(by_fold['B']['F1type']):+.4f}. Per-seed values above show whether signs persist."
        )
    det_abs = []
    type_abs = []
    for tissue in TISSUES:
        for fold in FOLDS:
            for seed in SEEDS:
                det_abs.append(abs(results[(tissue, fold, seed, MODELS[1])]["stats"]["F1det"] - results[(tissue, fold, seed, MODELS[0])]["stats"]["F1det"]))
                type_abs.append(abs(results[(tissue, fold, seed, MODELS[1])]["stats"]["F1type"] - results[(tissue, fold, seed, MODELS[0])]["stats"]["F1type"]))
    lines.extend([
        "",
        f"Across the 18 paired tissue/fold/seed contrasts, median |ΔF1det| is {statistics.median(det_abs):.4f} and median |ΔF1type| is {statistics.median(type_abs):.4f}. This is a descriptive stability comparison, not an inferential test.",
        "",
        "<!-- THREE_SEED_ANALYSIS_END -->",
    ])
    return "\n".join(lines)


def update_analysis(results):
    rebuild_seed42_analysis()
    text = ANALYSIS_MD.read_text()
    block = three_seed_markdown(results)
    marker = "## Machine-readable provenance"
    text = text.replace(marker, block + "\n\n" + marker)
    ANALYSIS_MD.write_text(text)

    with ANALYSIS_CSV.open(newline="") as handle:
        rows = list(csv.DictReader(handle))
    for tissue in TISSUES:
        for fold in FOLDS:
            for seed in SEEDS:
                general = results[(tissue, fold, seed, MODELS[0])]
                specialist = results[(tissue, fold, seed, MODELS[1])]
                for model, record in ((MODELS[0], general), (MODELS[1], specialist)):
                    for metric in METRICS:
                        row = {field: "" for field in CSV_FIELDS}
                        row.update(
                            record_type="three_seed_matched_metric",
                            tissue=tissue,
                            fold=fold,
                            model=model,
                            seed=seed,
                            split="matched_test",
                            metric=metric,
                            value=f"{record['stats'][metric]:.12g}",
                            matched_test_patches=record["matched"],
                            source_path=rel(record["source"]),
                            note="checkpoint_10 TEST; exact within-seed patch-ID intersection",
                        )
                        rows.append(row)
                for metric in METRICS:
                    row = {field: "" for field in CSV_FIELDS}
                    row.update(
                        record_type="three_seed_paired_delta",
                        tissue=tissue,
                        fold=fold,
                        model="Specialist minus generalist",
                        seed=seed,
                        split="matched_test",
                        metric=metric,
                        value=f"{specialist['stats'][metric] - general['stats'][metric]:.12g}",
                        matched_test_patches=general["matched"],
                        source_path=rel(specialist["source"]),
                        note=f"paired with {rel(general['source'])}",
                    )
                    rows.append(row)
            for model in MODELS:
                for metric in METRICS:
                    mean, sd = mean_sd(metric_values(results, tissue, fold, model, metric))
                    for suffix, value in (("mean", mean), ("sample_sd", sd)):
                        row = {field: "" for field in CSV_FIELDS}
                        row.update(
                            record_type="three_seed_aggregate",
                            tissue=tissue,
                            fold=fold,
                            model=model,
                            seed="42;43;44",
                            split="matched_test",
                            metric=f"{metric}_{suffix}",
                            value=f"{value:.12g}",
                            matched_test_patches=results[(tissue, fold, 42, model)]["matched"],
                            note="Fold-specific n=3; sample SD; folds not pooled",
                        )
                        rows.append(row)
    with ANALYSIS_CSV.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=CSV_FIELDS)
        writer.writeheader()
        writer.writerows(rows)


def fill_metric_fields(row, stats):
    mapping = {"Dice": "Dice", "Jaccard": "Jaccard", "bPQ": "bPQ", "mPQ": "mPQ", "F1det": "F1det", "F1type": "F1type"}
    for source, target in mapping.items():
        row[target] = f"{stats[source]:.9f}"
    for class_name in CLASSES:
        key = class_name.lower()
        values = stats["per_class"][class_name]
        row[f"{key}_F1"] = "" if not math.isfinite(values["f1"]) else f"{values['f1']:.9f}"
        row[f"{key}_matched_support"] = str(values["matched_true_support"])
        row[f"{key}_total_true_support"] = str(values["total_true_support"])


def new_master_rows(fieldnames, old_rows, results):
    general_job = {
        (row["fold"], int(row["seed"])): row["canonical_job_id"]
        for row in old_rows
        if row["protocol"] == "held_out_slide_KLT"
        and row["backbone"] == "CellViT-SAM-H"
        and row["method"] == "Selected PEFT"
        and row["status"] == "COMPLETED"
    }
    additions = []
    for tissue in TISSUES:
        for fold in FOLDS:
            for seed in NEW_SEEDS:
                specialist = results[(tissue, fold, seed, MODELS[1])]
                raw_names = sorted(json.loads(specialist["source"].read_text())["image_metrics"])
                raw_stats = aggregate_inference(specialist["source"], raw_names)
                raw = {field: "" for field in fieldnames}
                raw.update(
                    protocol="tissue_specific_held_out_slide",
                    backbone="CellViT-SAM-H",
                    method="Selected PEFT",
                    domain=tissue,
                    tissue=tissue,
                    seed=str(seed),
                    fold=fold,
                    status="COMPLETED",
                    canonical_job_id=JOB_IDS[(tissue, fold, seed)],
                    checkpoint_policy="checkpoint_10 TEST; exact verified adapter reconstruction",
                    trainable_params="7908779",
                    total_params="707644395",
                    trainable_pct="1.117620525",
                    source_inference_json=rel(specialist["source"]),
                    provenance_note="replication campaign; exact state and forward verification; full checkpoint deleted after verification",
                )
                fill_metric_fields(raw, raw_stats)
                additions.append(raw)
                for model in MODELS:
                    record = results[(tissue, fold, seed, model)]
                    derived = {field: "" for field in fieldnames}
                    derived.update(
                        protocol="composition_matched_generalist_vs_specialist",
                        backbone="CellViT-SAM-H",
                        method=model,
                        domain=tissue,
                        tissue=tissue,
                        seed=str(seed),
                        fold=fold,
                        status="COMPLETED",
                        canonical_job_id="derived",
                        checkpoint_policy="checkpoint_10 TEST; exact common patch IDs",
                        trainable_params="7908779",
                        total_params="707644395",
                        trainable_pct="1.117620525",
                        matched_patches=str(record["matched"]),
                        source_inference_json=rel(record["source"]),
                        provenance_note=(
                            f"{model}; exact intersection of generalist raw n={results[(tissue, fold, seed, MODELS[0])]['raw']} "
                            f"and specialist raw n={results[(tissue, fold, seed, MODELS[1])]['raw']}; "
                            f"generalist job {general_job[(fold, seed)]}; specialist job {JOB_IDS[(tissue, fold, seed)]}"
                        ),
                    )
                    fill_metric_fields(derived, record["stats"])
                    additions.append(derived)
    return additions


def update_master_csv(results):
    fieldnames, rows = load_master()
    rows = [
        row for row in rows
        if not (
            row["backbone"] == "CellViT-SAM-H"
            and row["tissue"] in TISSUES
            and row["seed"] in {"43", "44"}
            and row["protocol"] in {"tissue_specific_held_out_slide", "composition_matched_generalist_vs_specialist"}
        )
    ]
    rows.extend(new_master_rows(fieldnames, rows, results))
    with MASTER_CSV.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def matched_table_rows(results, latex=False):
    rows = []
    for tissue in TISSUES:
        for fold in FOLDS:
            matched = results[(tissue, fold, 42, MODELS[0])]["matched"]
            values = []
            for model in MODELS:
                for metric in ("bPQ", "mPQ", "F1det", "F1type"):
                    values.append(pretty_agg(metric_values(results, tissue, fold, model, metric), latex=latex))
            if latex:
                rows.append(f"CellViT-SAM-H & {tissue} & {fold} & {matched} & " + " & ".join(values) + r" \\")
            else:
                rows.append(f"| CellViT-SAM-H | {tissue} | {fold} | {matched:,} | " + " / ".join(values[:4]) + " | " + " / ".join(values[4:]) + " |")
    return rows


def update_master_text(results):
    text = MASTER_MD.read_text()
    text = text.replace(
        "| SAM-H tissue-specific Selected PEFT | 18 | 18 | none |",
        "| SAM-H tissue-specific Selected PEFT | 30 | 30 | none; seed42 all nine tissues plus Kidney/Liver/Tonsil seed43/44 A/B |",
    )
    start = "## Table 4 — composition-matched KLT generalist versus tissue specialist"
    end = "\n## Appendix A — historical within-slide KLT method selection"
    before, remainder = text.split(start, 1)
    _, after = remainder.split(end, 1)
    c256 = []
    for line in remainder.splitlines():
        if line.startswith("| CellViT-256 |"):
            c256.append(line)
    block = [
        start,
        "",
        "CellViT-SAM-H values are fold-specific mean ± sample SD across seeds42/43/44; CellViT-256 remains seed42. Every pair uses the exact common patch-ID intersection within seed. Folds are not pooled.",
        "",
        "| Backbone | Tissue | Fold | Matched patches | Generalist bPQ / mPQ / F1det / F1type | Specialist bPQ / mPQ / F1det / F1type |",
        "|---|---|:---:|---:|---|---|",
        *matched_table_rows(results),
        *c256,
        "",
        "Sources: individual canonical checkpoint-10 JSONs and exact intersections are listed in `reports/workshop_master_results.csv`; detailed seed-level diagnostics are in `reports/tissue_specialist_failure_analysis.csv`.",
        "",
    ]
    text = before + "\n".join(block) + end + after
    claim = "- Three-seed, fold-separated, composition-matched CellViT-SAM-H generalist-versus-specialist evidence for Kidney, Liver and Tonsil; this is descriptive robustness evidence, not a biological-replicate significance analysis."
    if claim not in text:
        text = text.replace("### SUPPORTED ONCE CURRENT JOBS FINISH", claim + "\n\n### SUPPORTED ONCE CURRENT JOBS FINISH")
    MASTER_MD.write_text(text)

    tex = LATEX.read_text()
    table_start = r"\begin{table*}[t]" + "\n" + r"\centering" + "\n" + r"\caption{Composition-matched generalist versus tissue-specialist Selected PEFT results. Every row uses the exact common patch-ID intersection.}"
    prefix, rest = tex.split(table_start, 1)
    old_table, suffix = rest.split(r"\end{table*}", 1)
    c256_tex = [line for line in old_table.splitlines() if line.startswith("CellViT-256 &")]
    table = [
        r"\begin{table*}[t]",
        r"\centering",
        r"\caption{Composition-matched generalist versus tissue-specialist Selected PEFT results. SAM-H values are mean $\pm$ sample SD across seeds 42/43/44 within each Fold; CellViT-256 values are seed42.}",
        r"\label{tab:workshop-specialist}",
        r"\scriptsize",
        r"\begin{tabular}{lllrrrrrrrrr}",
        r"\toprule",
        r"Backbone & Tissue & Fold & N & \multicolumn{4}{c}{Generalist} & \multicolumn{4}{c}{Specialist} \\",
        r" & & & & bPQ & mPQ & $F1_{det}$ & $F1_{type}$ & bPQ & mPQ & $F1_{det}$ & $F1_{type}$ \\",
        r"\midrule",
        *matched_table_rows(results, latex=True),
        *c256_tex,
        r"\bottomrule",
        r"\end{tabular}",
        r"\end{table*}",
    ]
    LATEX.write_text(prefix + "\n".join(table) + suffix)


def main():
    _, rows = load_master()
    results = build_results(rows)
    update_analysis(results)
    update_master_csv(results)
    update_master_text(results)
    print(ANALYSIS_MD.relative_to(REPO))
    print(ANALYSIS_CSV.relative_to(REPO))
    print(MASTER_MD.relative_to(REPO))
    print(MASTER_CSV.relative_to(REPO))
    print(LATEX.relative_to(REPO))


if __name__ == "__main__":
    main()
