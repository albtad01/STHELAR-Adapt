#!/usr/bin/env python3
"""Generate final paper-ready type-assignment metric tables."""

from pathlib import Path

import numpy as np
import pandas as pd


RUNS_SUMMARY = Path("reports/runs_summary.csv")
TYPE_SUMMARY = Path("reports/type_assignment/type_f1_summary.csv")
OUT_DIR = Path("reports/paper_tables")
KLT_OUT = OUT_DIR / "klt_ablation_final_with_type_metrics.csv"
TISSUE_OUT = OUT_DIR / "tissue_specific_final_with_type_metrics.csv"
INTERPRETATION_OUT = OUT_DIR / "type_metric_interpretation.md"


def method_from_label(label):
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
    lowered = str(label).lower()
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


def mean(series):
    if isinstance(series, pd.DataFrame):
        series = series.iloc[:, 0]
    return float(pd.to_numeric(series, errors="coerce").mean())


def std_or_nan(series):
    values = pd.to_numeric(series, errors="coerce").dropna()
    if len(values) <= 1:
        return np.nan
    return float(values.std(ddof=1))


def load_per_class(row):
    slide_summary = row.get("slide_summary_csv")
    if not isinstance(slide_summary, str) or not slide_summary:
        return {}
    path = Path(slide_summary).with_name("per_class_type_metrics.csv")
    if not path.is_file():
        return {}
    df = pd.read_csv(path)
    if "class" not in df.columns or "f1" not in df.columns:
        return {}

    matched = pd.to_numeric(df.get("matched_true_support", 0), errors="coerce").fillna(0)
    unmatched = pd.to_numeric(df.get("unmatched_true", 0), errors="coerce").fillna(0)
    df = df.copy()
    df["true_support"] = matched + unmatched
    df["f1"] = pd.to_numeric(df["f1"], errors="coerce")

    present = df.loc[df["true_support"] > 0, "f1"]
    present_without_other = df.loc[
        (df["true_support"] > 0) & (df["class"].astype(str).str.lower() != "other"),
        "f1",
    ]

    out = {
        "macro_type_f1_present_classes": float(present.mean())
        if not present.empty
        else np.nan,
        "macro_type_f1_without_other": float(present_without_other.mean())
        if not present_without_other.empty
        else np.nan,
    }
    for _, class_row in df.iterrows():
        class_key = str(class_row["class"]).lower()
        out[f"{class_key}_f1"] = float(class_row["f1"])
        out[f"{class_key}_true_support"] = float(class_row["true_support"])
    return out


def enrich(type_df, runs_df):
    type_df = type_df.copy()
    runs_df = runs_df.copy()
    type_df["run_dir_key"] = type_df["run_dir"].astype(str)
    runs_df["run_dir_key"] = runs_df["run_dir"].astype(str)
    merged = type_df.merge(
        runs_df[["run_dir_key", "trainable_ratio_percent", "seed", "run_name"]],
        on="run_dir_key",
        how="left",
    )

    per_class = merged.apply(load_per_class, axis=1, result_type="expand")
    merged = pd.concat([merged, per_class], axis=1)
    return merged


def klt_grouped(enriched):
    klt = enriched[enriched["cohort"] == "KLT"].copy()
    klt["method"] = klt["label"].map(method_from_label)
    klt["decoder_scope"] = klt["label"].map(decoder_scope)

    rows = []
    for (method, scope), group in klt.groupby(["method", "decoder_scope"], sort=False):
        rows.append(
            {
                "method": method,
                "decoder_scope": scope,
                "trainable_percent": mean(group["trainable_ratio_percent"]),
                "mPQ_mean": mean(group["saved_mPQ"]),
                "mPQ_std": std_or_nan(group["saved_mPQ"]),
                "bPQ_mean": mean(group["saved_bPQ"]),
                "F1_detection_mean": mean(group["saved_f1_detection"]),
                "type_accuracy_mean": mean(group["type_accuracy"]),
                "macro_type_f1_mean": mean(group["macro_f1"]),
                "macro_type_f1_present_classes_mean": mean(
                    group["macro_type_f1_present_classes"]
                ),
                "macro_type_f1_without_other_mean": mean(
                    group["macro_type_f1_without_other"]
                ),
                "other_f1_mean": mean(group["other_f1"]),
                "weighted_type_f1_mean": mean(group["weighted_f1"]),
                "num_seeds": int(group["seed"].nunique(dropna=True)),
            }
        )

    out = pd.DataFrame(rows)
    fullft_mpq = out.loc[
        (out["method"] == "FullFT") & (out["decoder_scope"] == "all"), "mPQ_mean"
    ].iloc[0]
    out["mPQ_recovery_vs_fullft"] = out["mPQ_mean"] / fullft_mpq

    order = {
        ("Frozen CellViT", "frozen"): 0,
        ("FullFT", "all"): 1,
        ("LoRA+AF", "heads_only"): 2,
        ("LoRA+AF", "last_stage"): 3,
        ("LoRA+AF", "conv_adapters"): 4,
        ("VeRA+AF", "heads_only"): 5,
        ("VeRA+AF", "conv_adapters"): 6,
    }
    out["_order"] = out.apply(
        lambda row: order.get((row["method"], row["decoder_scope"]), 999),
        axis=1,
    )
    return out.sort_values("_order").drop(columns="_order")


def klt_output_table(klt):
    columns = [
        "method",
        "decoder_scope",
        "trainable_percent",
        "mPQ_mean",
        "mPQ_std",
        "bPQ_mean",
        "F1_detection_mean",
        "type_accuracy_mean",
        "macro_type_f1_mean",
        "macro_type_f1_present_classes_mean",
        "macro_type_f1_without_other_mean",
        "mPQ_recovery_vs_fullft",
    ]
    return klt[columns]


def tissue_output_table(enriched):
    tissue_df = enriched[enriched["cohort"].isin(["Liver", "Kidney", "Ovary"])].copy()
    tissue_df["method_kind"] = np.where(
        tissue_df["label"].str.contains("FullFT", case=False, na=False),
        "FullFT",
        np.where(
            tissue_df["label"].str.contains("LoRA\\+AF heads_only", case=False, na=False),
            "PEFT",
            "",
        ),
    )
    tissue_df = tissue_df[tissue_df["method_kind"].isin(["FullFT", "PEFT"])]

    rows = []
    for tissue, group in tissue_df.groupby("cohort", sort=False):
        methods = {row["method_kind"]: row for _, row in group.iterrows()}
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
                "FullFT_F1_detection": fullft["saved_f1_detection"],
                "PEFT_F1_detection": peft["saved_f1_detection"],
                "FullFT_type_accuracy": fullft["type_accuracy"],
                "PEFT_type_accuracy": peft["type_accuracy"],
                "FullFT_macro_type_f1_present_classes": fullft[
                    "macro_type_f1_present_classes"
                ],
                "PEFT_macro_type_f1_present_classes": peft[
                    "macro_type_f1_present_classes"
                ],
            }
        )
    return pd.DataFrame(rows)


def round_numeric(df):
    df = df.copy()
    numeric = df.select_dtypes(include=[np.number]).columns
    df[numeric] = df[numeric].round(6)
    return df


def fmt(value, digits=3):
    if pd.isna(value):
        return "NA"
    return f"{float(value):.{digits}f}"


def latex_escape(value):
    return str(value).replace("_", "\\_").replace("%", "\\%")


def latex_table(klt):
    display = klt_output_table(klt).copy()
    display["Method"] = display["method"]
    display["Decoder"] = display["decoder_scope"]
    display["Train %"] = display["trainable_percent"].map(lambda v: fmt(v, 2))
    display["mPQ"] = display["mPQ_mean"].map(lambda v: fmt(v, 3))
    display["bPQ"] = display["bPQ_mean"].map(lambda v: fmt(v, 3))
    display["F1_det"] = display["F1_detection_mean"].map(lambda v: fmt(v, 3))
    display["Type Acc."] = display["type_accuracy_mean"].map(lambda v: fmt(v, 3))

    cols = ["Method", "Decoder", "Train %", "mPQ", "bPQ", "F1_det", "Type Acc."]
    lines = [
        "\\begin{tabular}{llccccc}",
        "\\toprule",
        "Method & Decoder & Train \\% & mPQ & bPQ & F1\\_det & Type Acc. \\\\",
        "\\midrule",
    ]
    for _, row in display.iterrows():
        lines.append(" & ".join(latex_escape(row[col]) for col in cols) + " \\\\")
    lines.extend(["\\bottomrule", "\\end{tabular}"])
    return "\n".join(lines)


def write_interpretation(klt, tissue):
    fullft = klt[(klt["method"] == "FullFT") & (klt["decoder_scope"] == "all")].iloc[0]
    frozen = klt[klt["method"] == "Frozen CellViT"].iloc[0]
    heads = klt[(klt["method"] == "LoRA+AF") & (klt["decoder_scope"] == "heads_only")].iloc[0]
    last = klt[(klt["method"] == "LoRA+AF") & (klt["decoder_scope"] == "last_stage")].iloc[0]
    best_acc = klt.sort_values("type_accuracy_mean", ascending=False).iloc[0]
    best_macro = klt.sort_values("macro_type_f1_present_classes_mean", ascending=False).iloc[0]
    best_peft_mpq = klt[
        (klt["method"] != "FullFT") & (klt["method"] != "Frozen CellViT")
    ].sort_values("mPQ_mean", ascending=False).iloc[0]
    tissue_best = tissue.sort_values("mPQ_recovery", ascending=False).iloc[0]
    tissue_worst = tissue.sort_values("mPQ_recovery", ascending=True).iloc[0]

    bullets = [
        (
            f"KLT FullFT remains the reference ceiling with mPQ {fmt(fullft['mPQ_mean'])}, "
            f"bPQ {fmt(fullft['bPQ_mean'])}, F1_detection {fmt(fullft['F1_detection_mean'])}, "
            f"and type accuracy {fmt(fullft['type_accuracy_mean'])} across "
            f"{int(fullft['num_seeds'])} seeds."
        ),
        (
            f"LoRA+AF heads_only remains the best default when type accuracy is considered: "
            f"it has the highest mean type accuracy ({fmt(heads['type_accuracy_mean'])}) "
            f"and highest F1_detection ({fmt(heads['F1_detection_mean'])}), while retaining "
            f"{fmt(heads['mPQ_recovery_vs_fullft'] * 100, 1)}% of FullFT mPQ with "
            f"{fmt(heads['trainable_percent'])}% trainable parameters."
        ),
        (
            f"LoRA+AF last_stage is stronger for macro type-F1 and Other-class behavior: "
            f"present-class macro F1 is {fmt(last['macro_type_f1_present_classes_mean'])} "
            f"versus {fmt(heads['macro_type_f1_present_classes_mean'])} for heads_only, "
            f"and Other F1 is {fmt(last['other_f1_mean'])} versus {fmt(heads['other_f1_mean'])}."
        ),
        (
            f"The best non-FullFT mPQ row is {best_peft_mpq['method']} "
            f"{best_peft_mpq['decoder_scope']} at mPQ {fmt(best_peft_mpq['mPQ_mean'])}; "
            f"the best present-class macro F1 row overall is {best_macro['method']} "
            f"{best_macro['decoder_scope']} at {fmt(best_macro['macro_type_f1_present_classes_mean'])}."
        ),
        (
            f"Frozen CellViT preserves nucleus detection but fails type assignment: "
            f"F1_detection {fmt(frozen['F1_detection_mean'])} and bPQ {fmt(frozen['bPQ_mean'])}, "
            f"but mPQ {fmt(frozen['mPQ_mean'])}, type accuracy {fmt(frozen['type_accuracy_mean'])}, "
            f"and macro type-F1 {fmt(frozen['macro_type_f1_mean'])}."
        ),
        (
            f"Across tissues with both final FullFT and PEFT results, LoRA+AF heads_only "
            f"recovers the most mPQ on {tissue_best['tissue']} "
            f"({fmt(tissue_best['mPQ_recovery'] * 100, 1)}%) and the least on "
            f"{tissue_worst['tissue']} ({fmt(tissue_worst['mPQ_recovery'] * 100, 1)}%)."
        ),
        (
            "Melanocyte has zero true support in these 5-class final evaluations, so "
            "`macro_type_f1_present_classes` excludes zero-support classes; otherwise "
            "macro F1 would include an absent-class zero."
        ),
        (
            "The Other/Unknown category is heterogeneous and materially affects macro-F1; "
            "`macro_type_f1_without_other` is provided to separate core epithelial, immune, "
            "and stromal behavior from the catch-all class."
        ),
    ]

    frozen_sentence = (
        "Frozen CellViT transfers nucleus detection reasonably well "
        f"(F1_detection={fmt(frozen['F1_detection_mean'])}, bPQ={fmt(frozen['bPQ_mean'])}) "
        "but essentially fails STHELAR type assignment "
        f"(mPQ={fmt(frozen['mPQ_mean'])}, type accuracy={fmt(frozen['type_accuracy_mean'])}, "
        f"macro type-F1={fmt(frozen['macro_type_f1_mean'])}), showing that detection "
        "transfer alone is insufficient for cell-type labeling."
    )

    lines = ["# Type metric interpretation", ""]
    lines.extend(f"- {bullet}" for bullet in bullets)
    lines.extend(
        [
            "",
            "## Suggested Frozen CellViT sentence",
            "",
            frozen_sentence,
            "",
            "## Compact KLT LaTeX table suggestion",
            "",
            "```latex",
            latex_table(klt),
            "```",
        ]
    )
    INTERPRETATION_OUT.write_text("\n".join(lines) + "\n")


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    runs_df = pd.read_csv(RUNS_SUMMARY)
    type_df = pd.read_csv(TYPE_SUMMARY)
    enriched = enrich(type_df, runs_df)
    klt = klt_grouped(enriched)
    tissue = tissue_output_table(enriched)

    round_numeric(klt_output_table(klt)).to_csv(KLT_OUT, index=False)
    round_numeric(tissue).to_csv(TISSUE_OUT, index=False)
    write_interpretation(klt, tissue)

    print(f"Wrote {KLT_OUT}")
    print(f"Wrote {TISSUE_OUT}")
    print(f"Wrote {INTERPRETATION_OUT}")


if __name__ == "__main__":
    main()
