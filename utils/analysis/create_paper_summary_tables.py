#!/usr/bin/env python3
"""Create paper-ready summary tables from completed STHELAR reports."""

from pathlib import Path

import numpy as np
import pandas as pd


RUNS_SUMMARY = Path("reports/runs_summary.csv")
TYPE_SUMMARY = Path("reports/type_assignment/type_f1_summary.csv")
OUT_DIR = Path("reports/paper_tables")
KLT_OUT = OUT_DIR / "klt_ablation_with_type_metrics.csv"
TISSUE_OUT = OUT_DIR / "tissue_specific_fullft_vs_peft_with_type_metrics.csv"
NOTES_OUT = OUT_DIR / "paper_table_notes.md"


def slug_to_method(label):
    base = label.replace("KLT ", "")
    for token in (" seed42", " seed43"):
        base = base.replace(token, "")
    if base.startswith("LoRA+AF"):
        return "LoRA+AF"
    if base.startswith("VeRA+AF"):
        return "VeRA+AF"
    if base.startswith("FullFT"):
        return "FullFT"
    if base.startswith("Frozen CellViT"):
        return "Frozen CellViT"
    return base


def decoder_scope(label):
    lowered = label.lower()
    if "heads_only" in lowered:
        return "heads_only"
    if "last_stage" in lowered:
        return "last_stage"
    if "conv_adapters" in lowered:
        return "conv_adapters"
    if "fullft" in lowered:
        return "all"
    if "frozen" in lowered:
        return "frozen"
    return ""


def read_per_class_metrics(row):
    slide_summary = row.get("slide_summary_csv")
    if not isinstance(slide_summary, str) or not slide_summary:
        return np.nan, np.nan
    path = Path(slide_summary).with_name("per_class_type_metrics.csv")
    if not path.is_file():
        return np.nan, np.nan

    per_class = pd.read_csv(path)
    if "f1" not in per_class.columns or "class" not in per_class.columns:
        return np.nan, np.nan

    matched_support = pd.to_numeric(
        per_class.get("matched_true_support", 0), errors="coerce"
    ).fillna(0)
    unmatched_support = pd.to_numeric(
        per_class.get("unmatched_true", 0), errors="coerce"
    ).fillna(0)
    true_support = matched_support + unmatched_support
    f1 = pd.to_numeric(per_class["f1"], errors="coerce")

    present = f1[true_support > 0]
    present_without_other = f1[
        (true_support > 0) & (per_class["class"].astype(str).str.lower() != "other")
    ]
    return (
        float(present.mean()) if not present.empty else np.nan,
        float(present_without_other.mean())
        if not present_without_other.empty
        else np.nan,
    )


def enrich_type_summary(type_df, runs_df):
    type_df = type_df.copy()
    runs_df = runs_df.copy()
    type_df["run_dir_key"] = type_df["run_dir"].astype(str)
    runs_df["run_dir_key"] = runs_df["run_dir"].astype(str)

    columns = [
        "run_dir_key",
        "trainable_ratio_percent",
        "effective_adapter_type",
        "adapter_type",
        "seed",
        "run_name",
    ]
    merged = type_df.merge(runs_df[columns], on="run_dir_key", how="left")

    macro_variants = merged.apply(read_per_class_metrics, axis=1, result_type="expand")
    merged["macro_type_f1_present_classes"] = macro_variants[0]
    merged["macro_type_f1_without_other"] = macro_variants[1]
    return merged


def std_or_nan(series):
    values = pd.to_numeric(series, errors="coerce").dropna()
    if len(values) <= 1:
        return np.nan
    return float(values.std(ddof=1))


def mean(series):
    return float(pd.to_numeric(series, errors="coerce").mean())


def create_klt_table(df):
    klt = df[df["cohort"] == "KLT"].copy()
    klt["method"] = klt["label"].map(slug_to_method)
    klt["decoder_scope"] = klt["label"].map(decoder_scope)

    grouped_rows = []
    for (method, scope), group in klt.groupby(["method", "decoder_scope"], sort=False):
        grouped_rows.append(
            {
                "method": method,
                "decoder_scope": scope,
                "trainable_percent": mean(group["trainable_ratio_percent"]),
                "mPQ_mean": mean(group["saved_mPQ"]),
                "mPQ_std": std_or_nan(group["saved_mPQ"]),
                "bPQ_mean": mean(group["saved_bPQ"]),
                "F1_det_mean": mean(group["saved_f1_detection"]),
                "type_accuracy_mean": mean(group["type_accuracy"]),
                "macro_type_f1_mean": mean(group["macro_f1"]),
                "weighted_type_f1_mean": mean(group["weighted_f1"]),
                "macro_type_f1_present_classes_mean": mean(
                    group["macro_type_f1_present_classes"]
                ),
                "macro_type_f1_without_other_mean": mean(
                    group["macro_type_f1_without_other"]
                ),
                "num_seeds": int(group["seed"].nunique(dropna=True)),
                "seeds": ",".join(str(int(s)) for s in sorted(group["seed"].dropna())),
            }
        )

    out = pd.DataFrame(grouped_rows)
    fullft_mpq = out.loc[
        (out["method"] == "FullFT") & (out["decoder_scope"] == "all"), "mPQ_mean"
    ].iloc[0]
    out["mPQ_recovery_vs_fullft"] = out["mPQ_mean"] / fullft_mpq

    preferred_order = {
        ("Frozen CellViT", "frozen"): 0,
        ("FullFT", "all"): 1,
        ("LoRA+AF", "heads_only"): 2,
        ("LoRA+AF", "last_stage"): 3,
        ("LoRA+AF", "conv_adapters"): 4,
        ("VeRA+AF", "heads_only"): 5,
        ("VeRA+AF", "conv_adapters"): 6,
    }
    out["_order"] = out.apply(
        lambda row: preferred_order.get((row["method"], row["decoder_scope"]), 999),
        axis=1,
    )
    out = out.sort_values("_order").drop(columns="_order")

    columns = [
        "method",
        "decoder_scope",
        "trainable_percent",
        "mPQ_mean",
        "mPQ_std",
        "bPQ_mean",
        "F1_det_mean",
        "type_accuracy_mean",
        "macro_type_f1_mean",
        "weighted_type_f1_mean",
        "mPQ_recovery_vs_fullft",
        "macro_type_f1_present_classes_mean",
        "macro_type_f1_without_other_mean",
        "num_seeds",
        "seeds",
    ]
    return out[columns]


def create_tissue_table(df):
    tissue_df = df[df["cohort"].isin(["Liver", "Kidney", "Ovary"])].copy()
    tissue_df["method"] = np.where(
        tissue_df["label"].str.contains("FullFT", case=False, na=False),
        "FullFT",
        np.where(
            tissue_df["label"].str.contains("LoRA\\+AF heads_only", case=False, na=False),
            "PEFT",
            "",
        ),
    )
    tissue_df = tissue_df[tissue_df["method"].isin(["FullFT", "PEFT"])]

    rows = []
    for tissue, group in tissue_df.groupby("cohort", sort=False):
        methods = {row["method"]: row for _, row in group.iterrows()}
        if "FullFT" not in methods or "PEFT" not in methods:
            continue
        fullft = methods["FullFT"]
        peft = methods["PEFT"]
        rows.append(
            {
                "tissue": tissue,
                "FullFT_mPQ": fullft["saved_mPQ"],
                "PEFT_mPQ": peft["saved_mPQ"],
                "mPQ_recovery": peft["saved_mPQ"] / fullft["saved_mPQ"],
                "FullFT_F1_det": fullft["saved_f1_detection"],
                "PEFT_F1_det": peft["saved_f1_detection"],
                "FullFT_type_accuracy": fullft["type_accuracy"],
                "PEFT_type_accuracy": peft["type_accuracy"],
                "FullFT_macro_type_f1": fullft["macro_f1"],
                "PEFT_macro_type_f1": peft["macro_f1"],
                "FullFT_macro_type_f1_present_classes": fullft[
                    "macro_type_f1_present_classes"
                ],
                "PEFT_macro_type_f1_present_classes": peft[
                    "macro_type_f1_present_classes"
                ],
                "FullFT_macro_type_f1_without_other": fullft[
                    "macro_type_f1_without_other"
                ],
                "PEFT_macro_type_f1_without_other": peft[
                    "macro_type_f1_without_other"
                ],
            }
        )
    return pd.DataFrame(rows)


def rounded(df):
    df = df.copy()
    numeric = df.select_dtypes(include=[np.number]).columns
    df[numeric] = df[numeric].round(6)
    return df


def fmt(value, digits=3):
    if pd.isna(value):
        return "NA"
    return f"{float(value):.{digits}f}"


def write_notes(klt, tissue):
    best_peft = klt[
        (klt["method"] != "FullFT") & (klt["method"] != "Frozen CellViT")
    ].sort_values("mPQ_mean", ascending=False).iloc[0]
    fullft = klt[(klt["method"] == "FullFT") & (klt["decoder_scope"] == "all")].iloc[0]
    frozen = klt[klt["method"] == "Frozen CellViT"].iloc[0]
    best_tissue = tissue.sort_values("mPQ_recovery", ascending=False).iloc[0]
    worst_tissue = tissue.sort_values("mPQ_recovery", ascending=True).iloc[0]

    notes = [
        "# Paper table notes",
        "",
        (
            f"- KLT FullFT is the reference row: mean mPQ {fmt(fullft['mPQ_mean'])}, "
            f"bPQ {fmt(fullft['bPQ_mean'])}, detection F1 {fmt(fullft['F1_det_mean'])}, "
            f"and type accuracy {fmt(fullft['type_accuracy_mean'])} across "
            f"{int(fullft['num_seeds'])} seeds."
        ),
        (
            f"- The strongest KLT parameter-efficient row by mPQ is "
            f"{best_peft['method']} {best_peft['decoder_scope']}, reaching "
            f"{fmt(best_peft['mPQ_mean'])} mPQ, or "
            f"{fmt(best_peft['mPQ_recovery_vs_fullft'] * 100, 1)}% of FullFT, "
            f"with {fmt(best_peft['trainable_percent'])}% trainable parameters."
        ),
        (
            f"- Frozen CellViT keeps detection usable but has near-zero type assignment "
            f"on KLT: macro type F1 {fmt(frozen['macro_type_f1_mean'])} and "
            f"mPQ recovery {fmt(frozen['mPQ_recovery_vs_fullft'] * 100, 1)}%."
        ),
        (
            "- Macro type F1 excluding zero-support classes is included because "
            "Melanocyte has zero true support in these 5-class final evaluations; "
            "this avoids penalizing methods for an absent class."
        ),
        (
            "- Macro type F1 without Other is also included over present non-Other "
            "classes, separating core epithelial/immune/stromal behavior from the "
            "heterogeneous Other label."
        ),
        (
            f"- Tissue-specific PEFT recovers the most FullFT mPQ for "
            f"{best_tissue['tissue']} ({fmt(best_tissue['mPQ_recovery'] * 100, 1)}%) "
            f"and the least for {worst_tissue['tissue']} "
            f"({fmt(worst_tissue['mPQ_recovery'] * 100, 1)}%)."
        ),
        (
            "- The ovary tissue-specific rows should be discussed with the slide "
            "dominance caveat from the slide-wise report, since aggregate ovary "
            "metrics are dominated by one test slide."
        ),
    ]
    NOTES_OUT.write_text("\n".join(notes) + "\n")


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    runs_df = pd.read_csv(RUNS_SUMMARY)
    type_df = pd.read_csv(TYPE_SUMMARY)
    enriched = enrich_type_summary(type_df, runs_df)

    klt = create_klt_table(enriched)
    tissue = create_tissue_table(enriched)

    klt_output = klt.drop(columns=["num_seeds", "seeds"])
    rounded(klt_output).to_csv(KLT_OUT, index=False)
    rounded(tissue).to_csv(TISSUE_OUT, index=False)
    write_notes(klt, tissue)

    print(f"Wrote {KLT_OUT}")
    print(f"Wrote {TISSUE_OUT}")
    print(f"Wrote {NOTES_OUT}")


if __name__ == "__main__":
    main()
