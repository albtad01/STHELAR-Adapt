#!/usr/bin/env python3
"""Create a compact Fisher diagnostic figure for PEFT placement evidence.

This utility is intentionally post-hoc: it reads the CSV files produced by
``utils/fisher_drift_diagnostic.py`` and does not recompute Fisher tensors.

The figure separates two complementary signals:

* total Fisher relevance, which is useful for large encoder regions such as
  MLP and attention QKV;
* parameter efficiency, which is useful for small decoder regions such as
  final heads.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


DEFAULT_GROUPS = [
    "final_heads",
    "encoder_attention_qkv",
    "encoder_mlp",
    "encoder_attention_proj",
    "decoder_nt",
    "decoder_conv_like",
]

SHORT_LABELS = {
    "final_heads": "final\nheads",
    "encoder_attention_qkv": "attn\nQKV",
    "encoder_mlp": "encoder\nMLP",
    "encoder_attention_proj": "attn\nproj",
    "decoder_nt": "decoder\nNT",
    "decoder_conv_like": "decoder\nconv",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--fisher-dir",
        type=Path,
        default=Path("reports/fisher_drift/klt_fullft_seed43_blocks_b5"),
        help="Directory containing fisher_summary.csv and fisher_drift_summary.csv.",
    )
    parser.add_argument(
        "--output-prefix",
        type=Path,
        default=Path("figures/fisher_peft_placement_evidence_horizontal"),
        help="Output prefix; .pdf/.png/.csv are written.",
    )
    parser.add_argument(
        "--checkpoint",
        type=int,
        default=10,
        help="Checkpoint index for Fisher mass/mean metrics.",
    )
    parser.add_argument(
        "--drift-a",
        type=int,
        default=5,
        help="Start checkpoint for late JS drift.",
    )
    parser.add_argument(
        "--drift-b",
        type=int,
        default=10,
        help="End checkpoint for late JS drift.",
    )
    parser.add_argument(
        "--groups",
        nargs="+",
        default=DEFAULT_GROUPS,
        help="Parameter groups to display, in order.",
    )
    return parser.parse_args()


def checkpoint_number(series: pd.Series) -> pd.Series:
    return series.astype(str).str.extract(r"checkpoint_(\d+)\.pth")[0].astype(int)


def row_normalize(values: np.ndarray) -> np.ndarray:
    """Normalize each row independently for display color only."""
    out = np.zeros_like(values, dtype=float)
    for idx, row in enumerate(values):
        finite = np.isfinite(row)
        if not finite.any():
            continue
        lo = float(np.nanmin(row[finite]))
        hi = float(np.nanmax(row[finite]))
        if hi > lo:
            out[idx] = (row - lo) / (hi - lo)
        elif hi > 0:
            out[idx] = 1.0
    return out


def fmt(value: float, row_name: str) -> str:
    if "per parameter" in row_name:
        if value >= 1:
            return f"{value:.2f}"
        if value >= 0.1:
            return f"{value:.3f}"
        return f"{value:.3g}"
    if "mass" in row_name:
        return f"{value:.1f}"
    if "JS" in row_name:
        return f"{value:.1f}"
    return f"{value:.3g}"


def main() -> None:
    args = parse_args()
    fisher_csv = args.fisher_dir / "fisher_summary.csv"
    drift_csv = args.fisher_dir / "fisher_drift_summary.csv"
    if not fisher_csv.exists():
        raise FileNotFoundError(fisher_csv)
    if not drift_csv.exists():
        raise FileNotFoundError(drift_csv)

    fisher = pd.read_csv(fisher_csv)
    drift = pd.read_csv(drift_csv)
    fisher["ckpt"] = checkpoint_number(fisher["checkpoint"])
    drift["ckpt_a"] = checkpoint_number(drift["checkpoint_a"])
    drift["ckpt_b"] = checkpoint_number(drift["checkpoint_b"])

    ckpt = fisher[fisher["ckpt"] == args.checkpoint].set_index("group")
    late = drift[
        (drift["ckpt_a"] == args.drift_a) & (drift["ckpt_b"] == args.drift_b)
    ].set_index("group")

    missing = [g for g in args.groups if g not in ckpt.index or g not in late.index]
    if missing:
        raise ValueError(f"Missing groups in Fisher outputs: {missing}")

    values = pd.DataFrame(index=args.groups)
    values["param_count"] = ckpt.loc[args.groups, "param_count"].astype(float)
    values["fisher_mass"] = ckpt.loc[args.groups, "fisher_mass"].astype(float)
    values["fisher_mass_percent"] = (
        100.0 * ckpt.loc[args.groups, "normalized_fisher_mass"].astype(float)
    )
    values["fisher_mean"] = ckpt.loc[args.groups, "fisher_mean"].astype(float)
    values["fisher_mean_x1e5"] = 1e5 * values["fisher_mean"]
    values["js_distance"] = late.loc[args.groups, "js_distance"].astype(float)
    values["js_distance_percent"] = 100.0 * values["js_distance"]
    values["js_per_param"] = values["js_distance"] / values["param_count"]
    values["js_per_param_x1e7"] = 1e7 * values["js_per_param"]

    matrix = np.vstack(
        [
            values["fisher_mass_percent"].to_numpy(),
            values["fisher_mean_x1e5"].to_numpy(),
            values["js_distance_percent"].to_numpy(),
            values["js_per_param_x1e7"].to_numpy(),
        ]
    )
    row_labels = [
        r"$100\,\tilde{M}_g$",
        r"$10^5\,\mu_g$",
        rf"$100\,d_{{JS}}$ {args.drift_a}$\to${args.drift_b}",
        r"$10^7\,d_{JS}/N_g$",
    ]
    row_descriptions = [
        "Fisher mass share (%)",
        "Fisher mass per parameter",
        "Late Fisher drift (%)",
        "Late drift per parameter",
    ]

    # Import matplotlib lazily so CSV extraction can still be used in minimal envs.
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib import patheffects

    display = row_normalize(matrix)

    fig, ax = plt.subplots(figsize=(7.35, 1.85))
    im = ax.imshow(display, cmap="YlGnBu", aspect="auto", vmin=0, vmax=1)

    ax.set_xticks(np.arange(len(args.groups)))
    ax.set_xticklabels([SHORT_LABELS.get(g, g) for g in args.groups], fontsize=9)
    ax.set_yticks(np.arange(len(row_labels)))
    ax.set_yticklabels(row_labels, fontsize=10)
    ax.tick_params(length=0)
    ax.set_xlabel("")
    ax.set_ylabel("")

    # Thin separators keep the figure readable at workshop-column scale.
    ax.set_xticks(np.arange(-0.5, len(args.groups), 1), minor=True)
    ax.set_yticks(np.arange(-0.5, len(row_labels), 1), minor=True)
    ax.grid(which="minor", color="white", linewidth=1.0)
    ax.tick_params(which="minor", bottom=False, left=False)
    for spine in ax.spines.values():
        spine.set_visible(False)

    for r_idx, row_name in enumerate(row_descriptions):
        for c_idx, value in enumerate(matrix[r_idx]):
            color = "white" if display[r_idx, c_idx] > 0.58 else "#1a1a1a"
            text = ax.text(
                c_idx,
                r_idx,
                fmt(float(value), row_name),
                ha="center",
                va="center",
                fontsize=8.6,
                color=color,
            )
            if color == "white":
                text.set_path_effects(
                    [patheffects.withStroke(linewidth=1.2, foreground="#123")]
                )

    # Subtle emphasis on the three intended PEFT targets.
    for c_idx in [0, 1, 2]:
        ax.add_patch(
            plt.Rectangle(
                (c_idx - 0.5, -0.5),
                1.0,
                len(row_labels),
                fill=False,
                edgecolor="#202020",
                linewidth=0.8,
            )
        )

    # Small colorbar only says colors are row-relative; the annotations carry units.
    cbar = fig.colorbar(im, ax=ax, fraction=0.025, pad=0.012)
    cbar.set_ticks([0, 1])
    cbar.set_ticklabels(["low", "high"])
    cbar.ax.tick_params(labelsize=7, length=0)
    cbar.outline.set_visible(False)
    cbar.set_label("within-row", fontsize=7, labelpad=-1)

    fig.tight_layout(pad=0.25)
    args.output_prefix.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output_prefix.with_suffix(".pdf"), bbox_inches="tight")
    fig.savefig(args.output_prefix.with_suffix(".png"), dpi=400, bbox_inches="tight")
    plt.close(fig)

    out = values.copy()
    out.insert(0, "group_label", [SHORT_LABELS.get(g, g).replace("\n", " ") for g in out.index])
    out.to_csv(args.output_prefix.with_suffix(".csv"))


if __name__ == "__main__":
    main()
