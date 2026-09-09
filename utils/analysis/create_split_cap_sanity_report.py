#!/usr/bin/env python3
"""Create split/cap sanity tables for STHELAR 5-class paper datasets."""

import os
from pathlib import Path

import pandas as pd
import yaml


DATASET_ROOT = Path(os.environ.get("DATA_ROOT", "data/cellvit_ready")).expanduser()
STHELAR_ROOT = Path(os.environ.get("STHELAR_ROOT", "data/STHELAR_40x")).expanduser()
OVERVIEW = STHELAR_ROOT / "patches_overview_sthelar40x.parquet"
SPLIT_CHECK_DIR = Path("reports/split_checks")
OUT_DIR = Path("reports/paper_tables")
OUT_CSV = OUT_DIR / "split_cap_sanity_table.csv"
OUT_MD = OUT_DIR / "split_cap_sanity_report.md"
DOMINANCE_THRESHOLD = 0.70


DATASETS = [
    {
        "dataset": "KLT",
        "tissue": "Kidney/Liver/Tonsil",
        "split_check": SPLIT_CHECK_DIR / "klt_margin128_spatial_leakage_check.csv",
        "manifest": DATASET_ROOT
        / "sthelar40x_kidney_liver_tonsil_5class_spatial_margin128"
        / "split_manifest.yaml",
    },
]

for stem, tissue in [
    ("sthelar40x_breast_5class_spatial_margin128_cap50000", "Breast"),
    ("sthelar40x_colon_5class_spatial_margin128_cap50000", "Colon"),
    ("sthelar40x_kidney_5class_spatial_margin128", "Kidney"),
    ("sthelar40x_liver_5class_spatial_margin128", "Liver"),
    ("sthelar40x_lung_5class_spatial_margin128_cap50000", "Lung"),
    ("sthelar40x_ovary_5class_spatial_margin128", "Ovary"),
    ("sthelar40x_pancreatic_5class_spatial_margin128_cap50000", "Pancreatic"),
    ("sthelar40x_skin_5class_spatial_margin128_cap50000", "Skin"),
    ("sthelar40x_tonsil_5class_spatial_margin128_cap50000", "Tonsil"),
]:
    DATASETS.append(
        {
            "dataset": stem,
            "tissue": tissue,
            "split_check": SPLIT_CHECK_DIR / f"{stem}.csv",
            "manifest": DATASET_ROOT / stem / "split_manifest.yaml",
        }
    )


def read_manifest(path):
    with path.open() as handle:
        return yaml.safe_load(handle)


def load_raw_counts(slide_ids):
    overview = pd.read_parquet(OVERVIEW, columns=["slide_id"])
    overview["slide_id"] = overview["slide_id"].astype(str)
    counts = overview[overview["slide_id"].isin(slide_ids)].groupby("slide_id").size()
    return counts.to_dict()


def dataset_rows(spec, raw_counts):
    manifest = read_manifest(spec["manifest"])
    check = pd.read_csv(spec["split_check"])
    max_cap = manifest.get("max_patches_per_slide")
    selected_slides = [str(slide) for slide in manifest.get("selected_slides", [])]

    rows = []
    for _, row in check.iterrows():
        slide_id = str(row["slide_id"])
        train = int(row["train_count"])
        valid = int(row["valid_count"])
        test = int(row["test_count"])
        discarded = int(row["discarded_by_margin_count"])
        kept = train + valid + test
        after_cap = kept + discarded
        before_cap = int(raw_counts.get(slide_id, after_cap))

        rows.append(
            {
                "dataset": spec["dataset"],
                "tissue": spec["tissue"],
                "row_type": "slide",
                "slide_id": slide_id,
                "selected_slides": ",".join(selected_slides),
                "max_patches_per_slide": max_cap,
                "before_cap_patches": before_cap,
                "after_cap_patches": after_cap,
                "after_spatial_margin_filter_patches": kept,
                "train_count": train,
                "valid_count": valid,
                "test_count": test,
                "discarded_by_margin_count": discarded,
                "cap_removed_patches": before_cap - after_cap,
                "margin_removed_fraction_after_cap": discarded / after_cap
                if after_cap
                else 0.0,
                "split_check_pass": bool(row.get("pass", True)),
                "split_check_csv": spec["split_check"].as_posix(),
                "split_manifest": spec["manifest"].as_posix(),
            }
        )

    total = {
        "dataset": spec["dataset"],
        "tissue": spec["tissue"],
        "row_type": "total",
        "slide_id": "ALL",
        "selected_slides": ",".join(selected_slides),
        "max_patches_per_slide": max_cap,
        "before_cap_patches": sum(r["before_cap_patches"] for r in rows),
        "after_cap_patches": sum(r["after_cap_patches"] for r in rows),
        "after_spatial_margin_filter_patches": sum(
            r["after_spatial_margin_filter_patches"] for r in rows
        ),
        "train_count": sum(r["train_count"] for r in rows),
        "valid_count": sum(r["valid_count"] for r in rows),
        "test_count": sum(r["test_count"] for r in rows),
        "discarded_by_margin_count": sum(r["discarded_by_margin_count"] for r in rows),
        "cap_removed_patches": sum(r["cap_removed_patches"] for r in rows),
        "margin_removed_fraction_after_cap": 0.0,
        "split_check_pass": all(r["split_check_pass"] for r in rows),
        "split_check_csv": spec["split_check"].as_posix(),
        "split_manifest": spec["manifest"].as_posix(),
    }
    if total["after_cap_patches"]:
        total["margin_removed_fraction_after_cap"] = (
            total["discarded_by_margin_count"] / total["after_cap_patches"]
        )
    rows.append(total)

    slide_after_cap_total = sum(r["after_cap_patches"] for r in rows if r["row_type"] == "slide")
    slide_kept_total = sum(
        r["after_spatial_margin_filter_patches"] for r in rows if r["row_type"] == "slide"
    )
    for row in rows:
        if row["row_type"] == "slide":
            row["after_cap_fraction_within_dataset"] = (
                row["after_cap_patches"] / slide_after_cap_total
                if slide_after_cap_total
                else 0.0
            )
            row["kept_fraction_within_dataset"] = (
                row["after_spatial_margin_filter_patches"] / slide_kept_total
                if slide_kept_total
                else 0.0
            )
        else:
            row["after_cap_fraction_within_dataset"] = 1.0
            row["kept_fraction_within_dataset"] = 1.0

    max_after_cap = max(r["after_cap_fraction_within_dataset"] for r in rows if r["row_type"] == "slide")
    max_kept = max(r["kept_fraction_within_dataset"] for r in rows if r["row_type"] == "slide")
    for row in rows:
        row["dominant_after_cap_slide"] = (
            row["row_type"] == "slide"
            and row["after_cap_fraction_within_dataset"] == max_after_cap
        )
        row["dominant_kept_slide"] = (
            row["row_type"] == "slide" and row["kept_fraction_within_dataset"] == max_kept
        )
        row["slide_dominates_after_cap"] = (
            row["row_type"] == "slide"
            and row["after_cap_fraction_within_dataset"] > DOMINANCE_THRESHOLD
        )
        row["slide_dominates_kept"] = (
            row["row_type"] == "slide"
            and row["kept_fraction_within_dataset"] > DOMINANCE_THRESHOLD
        )

    return rows


def fmt_int(value):
    return f"{int(value):,}"


def fmt_pct(value):
    return f"{100 * float(value):.1f}%"


def markdown_table(df, columns):
    lines = []
    lines.append("| " + " | ".join(title for title, _ in columns) + " |")
    lines.append("| " + " | ".join(["---"] * len(columns)) + " |")
    for _, row in df.iterrows():
        cells = [str(getter(row)).replace("|", "\\|") for _, getter in columns]
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines)


def write_report(table):
    slide_rows = table[table["row_type"] == "slide"].copy()
    total_rows = table[table["row_type"] == "total"].copy()
    klt_slides = slide_rows[slide_rows["dataset"] == "KLT"].copy()
    klt_total = total_rows[total_rows["dataset"] == "KLT"].iloc[0]
    tissue_slides = slide_rows[slide_rows["dataset"] != "KLT"].copy()
    tissue_totals = total_rows[total_rows["dataset"] != "KLT"].copy()

    dominant_tissues = tissue_slides[
        tissue_slides["slide_dominates_after_cap"] | tissue_slides["slide_dominates_kept"]
    ].copy()
    top_tissue_slides = (
        tissue_slides.sort_values(["dataset", "kept_fraction_within_dataset"])
        .groupby("dataset", as_index=False)
        .tail(1)
        .sort_values("kept_fraction_within_dataset", ascending=False)
    )

    lines = [
        "# Split/cap sanity report",
        "",
        "## Executive summary",
        "",
        (
            f"- Audited {len(DATASETS)} 5-class split manifests: KLT plus "
            f"{len(DATASETS) - 1} tissue-specific datasets."
        ),
        (
            "- `before_cap_patches` is counted from "
            "`patches_overview_sthelar40x.parquet`; `after_cap_patches` is "
            "`train + valid + test + discarded_by_margin`; and "
            "`after_spatial_margin_filter_patches` is `train + valid + test`."
        ),
        (
            "- The cap limits maximum patch contribution per slide. It does not "
            "balance slides: slides below the cap keep their original counts, and "
            "slides above the cap are sampled down to the cap."
        ),
        (
            f"- KLT uses a 10,000 patch/slide cap and keeps "
            f"{fmt_int(klt_total['after_spatial_margin_filter_patches'])} patches "
            f"after the 128-pixel spatial margin "
            f"({fmt_int(klt_total['train_count'])} train, "
            f"{fmt_int(klt_total['valid_count'])} valid, "
            f"{fmt_int(klt_total['test_count'])} test)."
        ),
    ]

    if dominant_tissues.empty:
        lines.append(
            f"- No tissue-specific dataset has one slide above the "
            f"{fmt_pct(DOMINANCE_THRESHOLD)} dominance threshold after cap or after "
            "spatial margin filtering."
        )
    else:
        labels = ", ".join(
            f"{row.tissue}:{row.slide_id} "
            f"({fmt_pct(row.kept_fraction_within_dataset)} kept)"
            for row in dominant_tissues.itertuples()
        )
        lines.append(
            f"- Tissue-specific datasets with one slide above the "
            f"{fmt_pct(DOMINANCE_THRESHOLD)} kept-patch threshold: {labels}."
        )

    lines.extend(["", "## KLT per-slide counts", ""])
    klt_columns = [
        ("Slide", lambda r: r["slide_id"]),
        ("Before cap", lambda r: fmt_int(r["before_cap_patches"])),
        ("After cap", lambda r: fmt_int(r["after_cap_patches"])),
        ("After margin", lambda r: fmt_int(r["after_spatial_margin_filter_patches"])),
        ("Train", lambda r: fmt_int(r["train_count"])),
        ("Valid", lambda r: fmt_int(r["valid_count"])),
        ("Test", lambda r: fmt_int(r["test_count"])),
        ("Margin discard", lambda r: fmt_int(r["discarded_by_margin_count"])),
        ("Kept share", lambda r: fmt_pct(r["kept_fraction_within_dataset"])),
    ]
    lines.append(markdown_table(klt_slides, klt_columns))

    lines.extend(["", "## KLT split totals", ""])
    lines.append(
        markdown_table(
            pd.DataFrame([klt_total]),
            [
                ("Before cap", lambda r: fmt_int(r["before_cap_patches"])),
                ("After cap", lambda r: fmt_int(r["after_cap_patches"])),
                ("After margin", lambda r: fmt_int(r["after_spatial_margin_filter_patches"])),
                ("Train", lambda r: fmt_int(r["train_count"])),
                ("Valid", lambda r: fmt_int(r["valid_count"])),
                ("Test", lambda r: fmt_int(r["test_count"])),
                ("Margin discard", lambda r: fmt_int(r["discarded_by_margin_count"])),
            ],
        )
    )

    lines.extend(["", "## Tissue-specific dominance", ""])
    lines.append(
        markdown_table(
            top_tissue_slides,
            [
                ("Tissue", lambda r: r["tissue"]),
                ("Dominant slide", lambda r: r["slide_id"]),
                ("Before cap", lambda r: fmt_int(r["before_cap_patches"])),
                ("After cap share", lambda r: fmt_pct(r["after_cap_fraction_within_dataset"])),
                ("Kept share", lambda r: fmt_pct(r["kept_fraction_within_dataset"])),
                ("Dominates?", lambda r: "yes" if r["slide_dominates_kept"] else "no"),
            ],
        )
    )

    lines.extend(["", "## Tissue-specific split totals", ""])
    lines.append(
        markdown_table(
            tissue_totals.sort_values("tissue"),
            [
                ("Tissue", lambda r: r["tissue"]),
                ("Before cap", lambda r: fmt_int(r["before_cap_patches"])),
                ("After cap", lambda r: fmt_int(r["after_cap_patches"])),
                ("After margin", lambda r: fmt_int(r["after_spatial_margin_filter_patches"])),
                ("Train", lambda r: fmt_int(r["train_count"])),
                ("Valid", lambda r: fmt_int(r["valid_count"])),
                ("Test", lambda r: fmt_int(r["test_count"])),
            ],
        )
    )

    lines.extend(
        [
            "",
            "## Notes for the paper",
            "",
            (
                "- The spatial split is within-slide for these datasets: every selected "
                "slide contributes train, validation, and test patches separated along "
                "the split axis with a 128-pixel boundary margin."
            ),
            (
                "- For capped datasets, the cap should be described as a maximum "
                "per-slide contribution, not as class balancing or slide balancing."
            ),
            (
                "- The CSV contains one row per slide plus an `ALL` total row per "
                "dataset for direct table construction."
            ),
            "",
            "## Files",
            "",
            f"- CSV: `{OUT_CSV.as_posix()}`",
        ]
    )

    OUT_MD.write_text("\n".join(lines) + "\n")


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    all_slide_ids = []
    for spec in DATASETS:
        manifest = read_manifest(spec["manifest"])
        all_slide_ids.extend(str(slide) for slide in manifest.get("selected_slides", []))
    raw_counts = load_raw_counts(sorted(set(all_slide_ids)))

    rows = []
    for spec in DATASETS:
        rows.extend(dataset_rows(spec, raw_counts))

    table = pd.DataFrame(rows)
    numeric = table.select_dtypes(include="number").columns
    table[numeric] = table[numeric].round(6)
    table.to_csv(OUT_CSV, index=False)
    write_report(table)

    print(f"Wrote {OUT_CSV}")
    print(f"Wrote {OUT_MD}")


if __name__ == "__main__":
    main()
