#!/usr/bin/env python3

import argparse
import re
from pathlib import Path

import pandas as pd
import matplotlib.pyplot as plt


def slugify(text: str) -> str:
    text = re.sub(r"[^A-Za-z0-9_.-]+", "_", text)
    text = re.sub(r"_+", "_", text)
    return text.strip("_")


def to_numeric(df: pd.DataFrame, cols):
    for col in cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    return df


def plot_loss(run_df: pd.DataFrame, run_name: str, outdir: Path, show: bool):
    cols = [c for c in ["train_loss", "val_loss"] if c in run_df.columns]
    cols = [c for c in cols if run_df[c].notna().any()]

    if not cols:
        print(f"[skip] No train/val loss columns available for {run_name}")
        return None

    fig, ax = plt.subplots(figsize=(8, 5))

    for col in cols:
        ax.plot(run_df["epoch"], run_df[col], marker="o", label=col)

    ax.set_title(f"Loss curves\n{run_name}")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss")
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()

    outpath = outdir / f"{slugify(run_name)}__loss.png"
    fig.savefig(outpath, dpi=200)
    print(f"[saved] {outpath}")

    if show:
        plt.show()
    else:
        plt.close(fig)

    return outpath


def plot_validation_metrics(run_df: pd.DataFrame, run_name: str, outdir: Path, show: bool):
    metric_cols = [c for c in ["val_Dice", "val_bPQ", "val_mPQ"] if c in run_df.columns]
    metric_cols = [c for c in metric_cols if run_df[c].notna().any()]

    if not metric_cols:
        print(f"[skip] No validation metric columns available for {run_name}")
        return None

    fig, ax = plt.subplots(figsize=(8, 5))

    for col in metric_cols:
        ax.plot(run_df["epoch"], run_df[col], marker="o", label=col)

    ax.set_title(f"Validation metrics\n{run_name}")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Metric value")
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()

    outpath = outdir / f"{slugify(run_name)}__val_metrics.png"
    fig.savefig(outpath, dpi=200)
    print(f"[saved] {outpath}")

    if show:
        plt.show()
    else:
        plt.close(fig)

    return outpath


def plot_epoch_time(run_df: pd.DataFrame, run_name: str, outdir: Path, show: bool):
    if "epoch_wall_minutes" not in run_df.columns or not run_df["epoch_wall_minutes"].notna().any():
        print(f"[skip] No epoch_wall_minutes available for {run_name}")
        return None

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(run_df["epoch"], run_df["epoch_wall_minutes"], marker="o", label="epoch_wall_minutes")

    ax.set_title(f"Epoch wall time\n{run_name}")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Minutes")
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()

    outpath = outdir / f"{slugify(run_name)}__epoch_time.png"
    fig.savefig(outpath, dpi=200)
    print(f"[saved] {outpath}")

    if show:
        plt.show()
    else:
        plt.close(fig)

    return outpath


def main():
    parser = argparse.ArgumentParser(
        description="Plot epoch-wise training/validation curves from reports/epoch_metrics.csv."
    )
    parser.add_argument(
        "--epochs-csv",
        default="reports/epoch_metrics.csv",
        help="Path to epoch_metrics.csv.",
    )

    selector = parser.add_mutually_exclusive_group(required=True)
    selector.add_argument(
        "--run-name",
        help="Exact run_name to plot.",
    )
    selector.add_argument(
        "--contains",
        help="Substring used to select one or more run_name values.",
    )

    parser.add_argument(
        "--outdir",
        default="reports/figures/training_curves",
        help="Directory where PNG figures will be saved.",
    )
    parser.add_argument(
        "--show",
        action="store_true",
        help="Show figures interactively in addition to saving them.",
    )
    parser.add_argument(
        "--time",
        action="store_true",
        help="Also plot epoch wall time.",
    )

    args = parser.parse_args()

    epochs_csv = Path(args.epochs_csv)
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    if not epochs_csv.is_file():
        raise FileNotFoundError(f"epoch_metrics.csv not found: {epochs_csv}")

    df = pd.read_csv(epochs_csv)

    required = {"run_name", "epoch"}
    missing = required.difference(df.columns)
    if missing:
        raise ValueError(f"Missing required columns in {epochs_csv}: {sorted(missing)}")

    numeric_cols = [
        "epoch",
        "max_epochs",
        "train_loss",
        "train_Dice",
        "train_Jaccard",
        "val_loss",
        "val_Dice",
        "val_Jaccard",
        "val_bPQ",
        "val_mPQ",
        "old_lr",
        "new_lr",
        "epoch_wall_minutes",
    ]
    df = to_numeric(df, numeric_cols)

    if args.run_name:
        selected = df[df["run_name"] == args.run_name].copy()
        if selected.empty:
            matches = sorted(
                name for name in df["run_name"].dropna().unique()
                if args.run_name.lower() in name.lower()
            )
            print(f"No exact match for run_name: {args.run_name}")
            if matches:
                print("\nPossible matches:")
                for name in matches:
                    print(f"  - {name}")
            raise SystemExit(1)
        run_names = [args.run_name]

    else:
        selected = df[df["run_name"].astype(str).str.contains(args.contains, case=False, na=False)].copy()
        if selected.empty:
            print(f"No runs found containing: {args.contains}")
            raise SystemExit(1)
        run_names = sorted(selected["run_name"].dropna().unique())

    print(f"Selected {len(run_names)} run(s):")
    for name in run_names:
        print(f"  - {name}")

    for run_name in run_names:
        run_df = selected[selected["run_name"] == run_name].copy()
        run_df = run_df.sort_values("epoch")

        print("\n" + "=" * 80)
        print(run_name)
        print(run_df[
            [
                c for c in [
                    "epoch",
                    "train_loss",
                    "val_loss",
                    "val_Dice",
                    "val_bPQ",
                    "val_mPQ",
                    "old_lr",
                    "new_lr",
                    "epoch_wall_minutes",
                    "new_best",
                ]
                if c in run_df.columns
            ]
        ].to_string(index=False))
        print("=" * 80)

        plot_loss(run_df, run_name, outdir, args.show)
        plot_validation_metrics(run_df, run_name, outdir, args.show)

        if args.time:
            plot_epoch_time(run_df, run_name, outdir, args.show)


if __name__ == "__main__":
    main()