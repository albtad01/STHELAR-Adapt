#!/usr/bin/env python3
"""Create slide-wise dominance summaries from type-assignment outputs."""

import argparse
import os
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")

import numpy as np
import pandas as pd

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


DEFAULT_REPORT_DIR = Path("reports/type_assignment")
DEFAULT_TYPE_F1 = DEFAULT_REPORT_DIR / "type_f1_summary.csv"
DEFAULT_SUMMARY_CSV = DEFAULT_REPORT_DIR / "slide_dominance_summary.csv"
DEFAULT_REPORT_MD = DEFAULT_REPORT_DIR / "slide_dominance_report.md"
DEFAULT_BARPLOT = DEFAULT_REPORT_DIR / "slide_dominance_barplot.png"

METRICS = {
    "mPQ": "mPQ",
    "bPQ": "bPQ",
    "detection_f1": "F1_detection",
    "type_accuracy": "type_accuracy",
    "type_macro_f1": "type_macro_f1",
}
DOMINANCE_THRESHOLD = 0.70
TISSUE_COHORTS = {"Liver", "Kidney", "Ovary"}


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--report-dir", type=Path, default=DEFAULT_REPORT_DIR)
    parser.add_argument("--type-f1-summary", type=Path, default=DEFAULT_TYPE_F1)
    parser.add_argument("--summary-csv", type=Path, default=DEFAULT_SUMMARY_CSV)
    parser.add_argument("--report-md", type=Path, default=DEFAULT_REPORT_MD)
    parser.add_argument("--barplot", type=Path, default=DEFAULT_BARPLOT)
    return parser.parse_args()


def read_type_f1(path):
    if not path.is_file():
        return {}
    df = pd.read_csv(path)
    metadata = {}
    for order, row in df.reset_index(drop=True).iterrows():
        slide_path = str(row.get("slide_summary_csv", ""))
        if not slide_path:
            continue
        metadata[Path(slide_path).as_posix()] = {
            "order": order,
            "label": row.get("label", ""),
            "cohort": row.get("cohort", ""),
            "matching": row.get("matching", ""),
            "run_dir": row.get("run_dir", ""),
        }
    return metadata


def as_float(series):
    return pd.to_numeric(series, errors="coerce")


def metric_stats(values):
    values = as_float(values).dropna()
    if values.empty:
        return {
            "min": np.nan,
            "mean": np.nan,
            "max": np.nan,
            "std": np.nan,
            "cv": np.nan,
        }
    mean = float(values.mean())
    std = float(values.std(ddof=0))
    return {
        "min": float(values.min()),
        "mean": mean,
        "max": float(values.max()),
        "std": std,
        "cv": std / mean if mean else np.nan,
    }


def dominant_row(df, fraction_col):
    fractions = as_float(df[fraction_col])
    if fractions.dropna().empty:
        return "", np.nan
    idx = fractions.idxmax()
    return str(df.loc[idx, "slide_id"]), float(fractions.loc[idx])


def summarize_slide_file(path, metadata):
    df = pd.read_csv(path)
    run_key = path.parent.name
    path_key = path.as_posix()
    meta = metadata.get(path_key, {})

    if "slide_id" not in df.columns:
        df["slide_id"] = ""
    df["slide_id"] = df["slide_id"].fillna("").astype(str)

    missing_slide_ids = int((df["slide_id"].str.strip() == "").sum())
    slide_ids_available = missing_slide_ids == 0
    dom_patch_slide, dom_patch_fraction = dominant_row(df, "patch_fraction")
    dom_matched_slide, dom_matched_fraction = dominant_row(df, "matched_fraction")
    slide_dominated = (
        dom_patch_fraction > DOMINANCE_THRESHOLD
        or dom_matched_fraction > DOMINANCE_THRESHOLD
    )

    row = {
        "run_label": meta.get("label") or run_key,
        "cohort": meta.get("cohort") or infer_cohort(run_key),
        "run_key": run_key,
        "matching": meta.get("matching", ""),
        "num_test_slides_or_regions": len(df),
        "slide_ids_available": slide_ids_available,
        "missing_slide_id_rows": missing_slide_ids,
        "dominant_patch_slide_or_region": dom_patch_slide,
        "dominant_patch_fraction": dom_patch_fraction,
        "dominant_matched_slide_or_region": dom_matched_slide,
        "dominant_matched_fraction": dom_matched_fraction,
        "slide_dominated": bool(slide_dominated),
        "slide_summary_csv": path_key,
        "run_dir": meta.get("run_dir", ""),
    }

    for source_name, output_name in METRICS.items():
        stats = metric_stats(df[source_name]) if source_name in df.columns else metric_stats([])
        row[f"{output_name}_min"] = stats["min"]
        row[f"{output_name}_mean"] = stats["mean"]
        row[f"{output_name}_max"] = stats["max"]
        if source_name in {"mPQ", "type_accuracy"}:
            row[f"{output_name}_std"] = stats["std"]
            row[f"{output_name}_cv"] = stats["cv"]

    if not slide_ids_available:
        row["notes"] = f"{missing_slide_ids} row(s) missing slide_id"
    elif len(df) == 1:
        row["notes"] = "single slide/region"
    elif len(df) == 2 and slide_dominated:
        row["notes"] = "two-slide run with one dominant slide"
    else:
        row["notes"] = ""
    return row, df.assign(run_label=row["run_label"], cohort=row["cohort"], run_key=run_key)


def infer_cohort(run_key):
    if run_key.startswith("klt_"):
        return "KLT"
    for cohort in TISSUE_COHORTS:
        if run_key.startswith(cohort.lower() + "_"):
            return cohort
    return ""


def collect_summaries(report_dir, metadata):
    paths = sorted(report_dir.glob("*/slide_summary.csv"))
    rows = []
    slide_frames = []
    for path in paths:
        row, slide_df = summarize_slide_file(path, metadata)
        rows.append(row)
        slide_frames.append(slide_df)

    summary = pd.DataFrame(rows)
    if summary.empty:
        return summary, pd.DataFrame()

    order_lookup = {
        Path(path).parent.name: meta.get("order", 10_000)
        for path, meta in metadata.items()
    }
    summary["_order"] = summary["run_key"].map(order_lookup).fillna(10_000)
    summary = summary.sort_values(["_order", "cohort", "run_label"]).drop(columns="_order")
    slides = pd.concat(slide_frames, ignore_index=True) if slide_frames else pd.DataFrame()
    return summary, slides


def fmt_num(value, digits=3):
    if pd.isna(value):
        return "NA"
    return f"{float(value):.{digits}f}"


def fmt_dom(slide, fraction):
    if not slide:
        slide = "missing ID"
    return f"{slide} ({fmt_num(fraction)})"


def escape_md(value):
    return str(value).replace("|", "\\|")


def md_table(rows, columns):
    lines = []
    header = [title for title, _ in columns]
    lines.append("| " + " | ".join(header) + " |")
    lines.append("| " + " | ".join(["---"] * len(columns)) + " |")
    for row in rows:
        cells = []
        for _, getter in columns:
            cells.append(escape_md(getter(row)))
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines)


def compact_rows(df):
    rows = []
    for _, row in df.iterrows():
        rows.append(
            {
                "Run": row["run_label"],
                "n": int(row["num_test_slides_or_regions"]),
                "Patch dominant": fmt_dom(
                    row["dominant_patch_slide_or_region"],
                    row["dominant_patch_fraction"],
                ),
                "Matched dominant": fmt_dom(
                    row["dominant_matched_slide_or_region"],
                    row["dominant_matched_fraction"],
                ),
                "mPQ min/mean/max": "/".join(
                    fmt_num(row[f"mPQ_{part}"]) for part in ("min", "mean", "max")
                ),
                "bPQ min/mean/max": "/".join(
                    fmt_num(row[f"bPQ_{part}"]) for part in ("min", "mean", "max")
                ),
                "F1 det min/mean/max": "/".join(
                    fmt_num(row[f"F1_detection_{part}"])
                    for part in ("min", "mean", "max")
                ),
                "Type acc min/mean/max": "/".join(
                    fmt_num(row[f"type_accuracy_{part}"])
                    for part in ("min", "mean", "max")
                ),
                "Macro F1 min/mean/max": "/".join(
                    fmt_num(row[f"type_macro_f1_{part}"])
                    for part in ("min", "mean", "max")
                ),
                "mPQ CV": fmt_num(row["mPQ_cv"]),
                "Acc CV": fmt_num(row["type_accuracy_cv"]),
                "Dominated": "yes" if row["slide_dominated"] else "no",
            }
        )
    return rows


def write_report(summary, slides, summary_csv_path, output_path, barplot_path):
    klt = summary[summary["cohort"] == "KLT"]
    tissue = summary[summary["cohort"].isin(TISSUE_COHORTS)]
    dominated = summary[summary["slide_dominated"]].copy()
    missing_ids = summary[~summary["slide_ids_available"]]

    max_patch = summary.loc[summary["dominant_patch_fraction"].idxmax()]
    max_matched = summary.loc[summary["dominant_matched_fraction"].idxmax()]

    lines = [
        "# Slide-wise / region-wise dominance report",
        "",
        "## Executive summary",
        "",
        (
            f"- Aggregated {len(summary)} completed type-assignment runs from "
            f"{len(slides)} slide/region rows."
        ),
        (
            f"- Using the dominance rule `patch_fraction > {DOMINANCE_THRESHOLD:.2f}` "
            f"or `matched_fraction > {DOMINANCE_THRESHOLD:.2f}`, "
            f"{len(dominated)} run(s) are slide-dominated."
        ),
        (
            f"- Largest patch share: {max_patch['run_label']} is dominated by "
            f"{fmt_dom(max_patch['dominant_patch_slide_or_region'], max_patch['dominant_patch_fraction'])}."
        ),
        (
            f"- Largest matched-nuclei share: {max_matched['run_label']} is dominated by "
            f"{fmt_dom(max_matched['dominant_matched_slide_or_region'], max_matched['dominant_matched_fraction'])}."
        ),
    ]

    if missing_ids.empty:
        lines.append("- All available slide summaries include non-empty slide/region IDs.")
    else:
        labels = ", ".join(missing_ids["run_label"].astype(str))
        lines.append(f"- Missing slide/region IDs were found in: {labels}.")

    lines.extend(["", "## KLT runs", ""])
    lines.append(table_for(klt))

    lines.extend(["", "## Tissue-specific Liver/Kidney/Ovary runs", ""])
    lines.append(table_for(tissue))

    lines.extend(["", "## Dominance warnings", ""])
    if dominated.empty:
        lines.append(
            "No run exceeds the configured slide-dominance threshold. "
            "The tissue-specific runs still have only two test slides each, so the "
            "dominant slide/region fractions should be reported rather than hidden."
        )
    else:
        for _, row in dominated.iterrows():
            qualifier = (
                " This is a two-slide/region run."
                if int(row["num_test_slides_or_regions"]) == 2
                else ""
            )
            lines.append(
                "- "
                f"{row['run_label']}: dominant patch slide/region "
                f"{fmt_dom(row['dominant_patch_slide_or_region'], row['dominant_patch_fraction'])}; "
                f"dominant matched slide/region "
                f"{fmt_dom(row['dominant_matched_slide_or_region'], row['dominant_matched_fraction'])}."
                f"{qualifier}"
            )

    if not missing_ids.empty:
        lines.extend(["", "## Missing slide IDs", ""])
        for _, row in missing_ids.iterrows():
            lines.append(
                f"- {row['run_label']}: {int(row['missing_slide_id_rows'])} "
                "slide_summary row(s) lack a slide_id."
            )

    two_slide = tissue[tissue["num_test_slides_or_regions"] == 2]
    if not two_slide.empty:
        lines.extend(["", "## Two-slide tissue-specific context", ""])
        for _, row in two_slide.iterrows():
            lines.append(
                "- "
                f"{row['run_label']}: patch dominance "
                f"{fmt_dom(row['dominant_patch_slide_or_region'], row['dominant_patch_fraction'])}; "
                f"matched-nuclei dominance "
                f"{fmt_dom(row['dominant_matched_slide_or_region'], row['dominant_matched_fraction'])}."
            )

    lines.extend(
        [
            "",
            "## Recommendation for the paper",
            "",
            (
                "Suggested limitation sentence: "
                "\"Slide-wise analysis showed that no KLT run exceeded the pre-specified "
                "dominance threshold, whereas tissue-specific evaluations used only two "
                "test slides per tissue and the ovary runs were dominated by one slide; "
                "therefore aggregate tissue-specific metrics should be reported with "
                "per-slide dominance diagnostics rather than treated as broad population "
                "estimates.\""
            ),
            "",
            (
                "Suggested table addition: include the number of test slides/regions, "
                "the dominant patch fraction, the dominant matched-nuclei fraction, "
                "and the min/mean/max per-slide mPQ and type accuracy alongside each "
                "aggregate final-inference result."
            ),
            "",
            "## Files",
            "",
            f"- Summary CSV: `{summary_csv_path.as_posix()}`",
            f"- Tissue-specific barplot: `{barplot_path.as_posix()}`",
        ]
    )

    output_path.write_text("\n".join(lines) + "\n")


def table_for(df):
    columns = [
        ("Run", lambda row: row["Run"]),
        ("n", lambda row: row["n"]),
        ("Patch dominant", lambda row: row["Patch dominant"]),
        ("Matched dominant", lambda row: row["Matched dominant"]),
        ("mPQ min/mean/max", lambda row: row["mPQ min/mean/max"]),
        ("bPQ min/mean/max", lambda row: row["bPQ min/mean/max"]),
        ("F1 det min/mean/max", lambda row: row["F1 det min/mean/max"]),
        ("Type acc min/mean/max", lambda row: row["Type acc min/mean/max"]),
        ("Macro F1 min/mean/max", lambda row: row["Macro F1 min/mean/max"]),
        ("mPQ CV", lambda row: row["mPQ CV"]),
        ("Acc CV", lambda row: row["Acc CV"]),
        ("Dominated", lambda row: row["Dominated"]),
    ]
    return md_table(compact_rows(df), columns)


def make_barplot(slides, summary, output_path):
    tissue_runs = summary[summary["cohort"].isin(TISSUE_COHORTS)]["run_key"].tolist()
    plot_df = slides[slides["run_key"].isin(tissue_runs)].copy()
    if plot_df.empty:
        return

    plot_df["patch_fraction"] = pd.to_numeric(plot_df["patch_fraction"], errors="coerce")
    plot_df["matched_fraction"] = pd.to_numeric(plot_df["matched_fraction"], errors="coerce")
    plot_df["label"] = plot_df["run_label"] + "\n" + plot_df["slide_id"].astype(str)
    plot_df = plot_df.sort_values(["cohort", "run_label", "slide_id"])

    y = np.arange(len(plot_df))
    height = 0.36
    fig_height = max(6.0, 0.46 * len(plot_df) + 1.6)
    fig, ax = plt.subplots(figsize=(12, fig_height))
    ax.barh(
        y - height / 2,
        plot_df["patch_fraction"],
        height,
        label="Patch fraction",
        color="#4C78A8",
    )
    ax.barh(
        y + height / 2,
        plot_df["matched_fraction"],
        height,
        label="Matched-nuclei fraction",
        color="#F58518",
    )
    ax.axvline(DOMINANCE_THRESHOLD, color="#C43C39", linewidth=1.2, linestyle="--")
    ax.set_xlim(0, 1)
    ax.set_xlabel("Fraction of run total")
    ax.set_yticks(y)
    ax.set_yticklabels(plot_df["label"])
    ax.invert_yaxis()
    ax.set_title("Tissue-specific slide/region dominance")
    ax.grid(axis="x", alpha=0.25)
    ax.legend(loc="upper right")
    fig.tight_layout()
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def main():
    args = parse_args()
    metadata = read_type_f1(args.type_f1_summary)
    summary, slides = collect_summaries(args.report_dir, metadata)
    if summary.empty:
        raise SystemExit(f"No slide_summary.csv files found under {args.report_dir}")

    args.summary_csv.parent.mkdir(parents=True, exist_ok=True)
    summary.to_csv(args.summary_csv, index=False)
    make_barplot(slides, summary, args.barplot)
    write_report(summary, slides, args.summary_csv, args.report_md, args.barplot)

    print(f"Wrote {args.summary_csv}")
    print(f"Wrote {args.report_md}")
    print(f"Wrote {args.barplot}")


if __name__ == "__main__":
    main()
