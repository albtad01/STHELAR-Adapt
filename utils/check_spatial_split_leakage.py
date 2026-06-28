#!/usr/bin/env python
"""Validate STHELAR spatial split leakage boundaries.

This diagnostic is read-only with respect to the dataset. It reconstructs the
pre-split, capped patch table from the raw STHELAR overview and compares the
expected spatial train/valid/test/discard assignment against the dataset's
``patch_info_with_split.csv``.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import pandas as pd
import yaml


SPLITS = ("train", "valid", "test")


def _read_yaml(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f)
    if not isinstance(data, dict):
        raise ValueError(f"Expected YAML mapping in {path}")
    return data


def _load_overview(
    overview_path: Path,
    slide_ids: list[str],
    max_patches_per_slide: int | None,
    random_seed: int,
) -> pd.DataFrame:
    cols = ["slide_id", "file_name", "xmin", "ymin", "xmax", "ymax"]
    df = pd.read_parquet(overview_path, columns=cols)
    df["slide_id"] = df["slide_id"].astype(str)
    df["file_name"] = df["file_name"].astype(str)
    df = df[df["slide_id"].isin(set(slide_ids))].copy()
    if df.empty:
        raise ValueError(f"No selected slides found in overview: {slide_ids}")

    df = df.sort_values(["slide_id", "file_name"]).reset_index(drop=True)

    parts: list[pd.DataFrame] = []
    for slide_id, df_slide in df.groupby("slide_id", sort=True):
        if max_patches_per_slide is not None and len(df_slide) > max_patches_per_slide:
            df_slide = df_slide.sample(n=max_patches_per_slide, random_state=random_seed)
        parts.append(df_slide)

    capped = pd.concat(parts, axis=0, ignore_index=True)
    return capped.sort_values(["slide_id", "file_name"]).reset_index(drop=True)


def _assign_expected_spatial_split(
    df: pd.DataFrame,
    axis: str,
    train_frac: float,
    valid_frac: float,
    boundary_margin: float,
) -> pd.DataFrame:
    out = df.copy()
    out["x_center"] = 0.5 * (out["xmin"] + out["xmax"])
    out["y_center"] = 0.5 * (out["ymin"] + out["ymax"])
    coord_col = "x_center" if axis == "x" else "y_center"

    expected_parts: list[pd.DataFrame] = []
    for slide_id, slide_df in out.groupby("slide_id", sort=True):
        slide_df = slide_df.copy()
        coord = slide_df[coord_col].astype(float)
        cmin = float(coord.min())
        cmax = float(coord.max())
        span = cmax - cmin
        if span <= 0:
            raise ValueError(f"Degenerate coordinate span for {slide_id}: {cmin}..{cmax}")

        b1 = cmin + train_frac * span
        b2 = cmin + (train_frac + valid_frac) * span

        slide_df["expected_split"] = "discard"
        slide_df.loc[coord <= (b1 - boundary_margin), "expected_split"] = "train"
        slide_df.loc[
            (coord >= (b1 + boundary_margin)) & (coord <= (b2 - boundary_margin)),
            "expected_split",
        ] = "valid"
        slide_df.loc[coord >= (b2 + boundary_margin), "expected_split"] = "test"

        slide_df["coord_min"] = cmin
        slide_df["coord_max"] = cmax
        slide_df["boundary_train_valid"] = b1
        slide_df["boundary_valid_test"] = b2
        expected_parts.append(slide_df)

    return pd.concat(expected_parts, axis=0, ignore_index=True)


def _coord_interval(df: pd.DataFrame, split: str, coord_col: str) -> tuple[float | None, float | None]:
    vals = df.loc[df["split"] == split, coord_col]
    if vals.empty:
        return None, None
    return float(vals.min()), float(vals.max())


def _make_plot(
    slide_df: pd.DataFrame,
    slide_id: str,
    coord_col: str,
    b1: float,
    b2: float,
    margin: float,
    output_path: Path,
) -> None:
    colors = {
        "train": "#4C78A8",
        "valid": "#F58518",
        "test": "#54A24B",
        "discard": "#B8B8B8",
    }

    fig, ax = plt.subplots(figsize=(10, 3.2))
    bins = min(80, max(10, int(slide_df[coord_col].nunique() // 2)))
    for split in ("discard", "train", "valid", "test"):
        vals = slide_df.loc[slide_df["expected_split"] == split, coord_col]
        if not vals.empty:
            ax.hist(vals, bins=bins, alpha=0.55, label=f"{split} ({len(vals)})", color=colors[split])

    ax.axvspan(b1 - margin, b1 + margin, color="#D62728", alpha=0.16, label="margin band")
    ax.axvspan(b2 - margin, b2 + margin, color="#D62728", alpha=0.16)
    ax.axvline(b1, color="#D62728", linestyle="--", linewidth=1.2)
    ax.axvline(b2, color="#D62728", linestyle="--", linewidth=1.2)
    ax.set_title(f"{slide_id}: {coord_col} distribution by expected split")
    ax.set_xlabel(coord_col)
    ax.set_ylabel("patches")
    ax.legend(loc="upper right", fontsize=8, ncol=2)
    fig.tight_layout()
    fig.savefig(output_path, dpi=160)
    plt.close(fig)


def run_check(
    dataset: Path,
    output_dir: Path,
    report_prefix: str = "klt_margin128_spatial_leakage_check",
    config_path: Path | None = None,
) -> tuple[bool, Path, list[Path], pd.DataFrame]:
    manifest_path = dataset / "split_manifest.yaml"
    patch_info_path = dataset / "patch_info_with_split.csv"
    if not manifest_path.exists():
        raise FileNotFoundError(manifest_path)
    if not patch_info_path.exists():
        raise FileNotFoundError(patch_info_path)

    manifest = _read_yaml(manifest_path)
    config = _read_yaml(config_path) if config_path is not None and config_path.exists() else {}
    strategy = manifest.get("actual_strategy", manifest.get("requested_strategy"))
    if strategy != "spatial":
        raise ValueError(f"Expected spatial split, found {strategy!r}")

    axis = str(manifest.get("split_axis", "x"))
    coord_col = "x_center" if axis == "x" else "y_center"
    boundary_margin = float(manifest.get("boundary_margin", 128))
    train_frac = float(manifest.get("train_frac", 0.70))
    valid_frac = float(manifest.get("valid_frac", 0.15))
    slide_ids = [str(s) for s in manifest.get("selected_slides", [])]
    config_slide_ids = [str(s) for s in config.get("slide_ids", [])]
    selected_slides_match_config = (
        not config_slide_ids or sorted(config_slide_ids) == sorted(slide_ids)
    )
    overview_path = Path(str(manifest["overview_path"]))
    max_patches_per_slide = manifest.get("max_patches_per_slide")
    random_seed = int(manifest.get("random_seed", 42))

    actual = pd.read_csv(patch_info_path)
    actual["slide_id"] = actual["slide_id"].astype(str)
    actual["file_name"] = actual["file_name"].astype(str)
    required_actual = {"slide_id", "file_name", "split", "xmin", "ymin", "xmax", "ymax", coord_col}
    missing = sorted(required_actual - set(actual.columns))
    if missing:
        raise ValueError(f"{patch_info_path} missing required columns: {missing}")

    fold_files_exist = {
        split: (dataset / f"cell_count_{split}.csv").exists()
        for split in SPLITS
    }
    global_split_counts = {
        split: int((actual["split"] == split).sum())
        for split in SPLITS
    }
    folds_nonempty = all(global_split_counts[split] > 0 for split in SPLITS)

    capped = _load_overview(
        overview_path=overview_path,
        slide_ids=slide_ids,
        max_patches_per_slide=max_patches_per_slide,
        random_seed=random_seed,
    )
    expected = _assign_expected_spatial_split(
        capped,
        axis=axis,
        train_frac=train_frac,
        valid_frac=valid_frac,
        boundary_margin=boundary_margin,
    )

    merged = expected.merge(
        actual[["slide_id", "file_name", "split"]],
        on=["slide_id", "file_name"],
        how="left",
        validate="one_to_one",
    )

    output_dir.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, Any]] = []
    plot_paths: list[Path] = []

    for slide_id in slide_ids:
        s = merged[merged["slide_id"] == slide_id].copy()
        b1 = float(s["boundary_train_valid"].iloc[0])
        b2 = float(s["boundary_valid_test"].iloc[0])
        cmin = float(s["coord_min"].iloc[0])
        cmax = float(s["coord_max"].iloc[0])

        actual_present = s["split"].notna()
        mismatch = s.loc[actual_present & (s["split"] != s["expected_split"])]
        missing_kept = s.loc[s["expected_split"].isin(SPLITS) & s["split"].isna()]
        unexpected_kept = s.loc[(s["expected_split"] == "discard") & s["split"].notna()]

        split_counts = {
            split: int((s["split"] == split).sum())
            for split in SPLITS
        }
        expected_counts = {
            f"expected_{split}_count": int((s["expected_split"] == split).sum())
            for split in ("train", "valid", "test", "discard")
        }

        # Margin violations are checked against the exact converter rules.
        train_viols = s.loc[(s["split"] == "train") & (s[coord_col] > b1 - boundary_margin)]
        valid_viols = s.loc[
            (s["split"] == "valid")
            & ((s[coord_col] < b1 + boundary_margin) | (s[coord_col] > b2 - boundary_margin))
        ]
        test_viols = s.loc[(s["split"] == "test") & (s[coord_col] < b2 + boundary_margin)]

        train_min, train_max = _coord_interval(s, "train", coord_col)
        valid_min, valid_max = _coord_interval(s, "valid", coord_col)
        test_min, test_max = _coord_interval(s, "test", coord_col)

        interval_disjoint = True
        gap_train_valid = None
        gap_valid_test = None
        if train_max is not None and valid_min is not None:
            gap_train_valid = valid_min - train_max
            interval_disjoint = interval_disjoint and train_max < valid_min
        if valid_max is not None and test_min is not None:
            gap_valid_test = test_min - valid_max
            interval_disjoint = interval_disjoint and valid_max < test_min
        if train_max is not None and test_min is not None:
            interval_disjoint = interval_disjoint and train_max < test_min

        slide_pass = (
            len(mismatch) == 0
            and len(missing_kept) == 0
            and len(unexpected_kept) == 0
            and len(train_viols) == 0
            and len(valid_viols) == 0
            and len(test_viols) == 0
            and interval_disjoint
            and selected_slides_match_config
            and all(fold_files_exist.values())
            and all(split_counts[split] > 0 for split in SPLITS)
        )

        row = {
            "slide_id": slide_id,
            "selected_slides_match_config": selected_slides_match_config,
            "train_fold_exists": fold_files_exist["train"],
            "valid_fold_exists": fold_files_exist["valid"],
            "test_fold_exists": fold_files_exist["test"],
            "global_train_count": global_split_counts["train"],
            "global_valid_count": global_split_counts["valid"],
            "global_test_count": global_split_counts["test"],
            "global_folds_nonempty": folds_nonempty,
            "axis": axis,
            "boundary_margin": int(boundary_margin),
            "coord_min": cmin,
            "coord_max": cmax,
            "boundary_train_valid": b1,
            "boundary_valid_test": b2,
            "train_count": split_counts["train"],
            "valid_count": split_counts["valid"],
            "test_count": split_counts["test"],
            **expected_counts,
            "discarded_by_margin_count": expected_counts["expected_discard_count"],
            "train_coord_min": train_min,
            "train_coord_max": train_max,
            "valid_coord_min": valid_min,
            "valid_coord_max": valid_max,
            "test_coord_min": test_min,
            "test_coord_max": test_max,
            "gap_train_valid": gap_train_valid,
            "gap_valid_test": gap_valid_test,
            "min_boundary_gap": min(
                gap
                for gap in [gap_train_valid, gap_valid_test]
                if gap is not None
            )
            if any(gap is not None for gap in [gap_train_valid, gap_valid_test])
            else None,
            "intervals_disjoint": interval_disjoint,
            "spatial_leakage_aware": interval_disjoint
            and len(train_viols) == 0
            and len(valid_viols) == 0
            and len(test_viols) == 0,
            "assignment_mismatch_count": int(len(mismatch)),
            "missing_expected_kept_count": int(len(missing_kept)),
            "unexpected_kept_discard_count": int(len(unexpected_kept)),
            "train_margin_violation_count": int(len(train_viols)),
            "valid_margin_violation_count": int(len(valid_viols)),
            "test_margin_violation_count": int(len(test_viols)),
            "pass": slide_pass,
        }
        rows.append(row)

        plot_path = output_dir / f"{report_prefix}_{slide_id}.png"
        _make_plot(
            slide_df=s,
            slide_id=slide_id,
            coord_col=coord_col,
            b1=b1,
            b2=b2,
            margin=boundary_margin,
            output_path=plot_path,
        )
        plot_paths.append(plot_path)

    summary = pd.DataFrame(rows)
    csv_path = output_dir / f"{report_prefix}.csv"
    summary.to_csv(csv_path, index=False)
    return bool(summary["pass"].all()), csv_path, plot_paths, summary


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dataset",
        type=Path,
        default=Path(
            "/gpfs/workdir/taddeial/workspace/Datasets/cellvit_ready/"
            "sthelar40x_kidney_liver_tonsil_5class_spatial_margin128"
        ),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("reports/split_checks"),
    )
    parser.add_argument(
        "--config",
        type=Path,
        help="Optional preprocessing config; selected slide IDs are checked against the dataset manifest.",
    )
    parser.add_argument(
        "--report-prefix",
        default="klt_margin128_spatial_leakage_check",
        help="Prefix for the CSV and plot filenames.",
    )
    args = parser.parse_args()

    passed, csv_path, plot_paths, summary = run_check(
        args.dataset,
        args.output_dir,
        report_prefix=args.report_prefix,
        config_path=args.config,
    )

    cols = [
        "slide_id",
        "train_count",
        "valid_count",
        "test_count",
        "discarded_by_margin_count",
        "gap_train_valid",
        "gap_valid_test",
        "min_boundary_gap",
        "assignment_mismatch_count",
        "train_margin_violation_count",
        "valid_margin_violation_count",
        "test_margin_violation_count",
        "spatial_leakage_aware",
        "pass",
    ]
    print(summary[cols].to_string(index=False))
    if not bool(summary["selected_slides_match_config"].all()):
        print("\nWARNING: selected slides in manifest do not match the provided config.")
    if not bool(summary["global_folds_nonempty"].all()):
        print("\nWARNING: at least one global split is empty.")
    for _, row in summary.iterrows():
        if row["min_boundary_gap"] is not None and row["min_boundary_gap"] < row["boundary_margin"]:
            print(
                f"\nWARNING: {row['slide_id']} has min boundary gap "
                f"{row['min_boundary_gap']} < margin {row['boundary_margin']}."
            )
        if min(row["train_count"], row["valid_count"], row["test_count"]) < 10:
            print(
                f"\nWARNING: {row['slide_id']} has an extremely small split: "
                f"train={row['train_count']} valid={row['valid_count']} test={row['test_count']}. "
                "If the x-axis split is degenerate, consider a future y-axis split."
            )
    print(f"\nCSV: {csv_path}")
    print("Plots:")
    for path in plot_paths:
        print(f"  {path}")
    print(f"\nFINAL: {'PASS' if passed else 'FAIL'}")
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
