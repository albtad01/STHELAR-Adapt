#!/usr/bin/env python3
"""Build compact STHELAR qualitative grids for the workshop paper.

The script writes only derived artifacts under ``figures/`` and ``reports/``.
It never modifies run directories. Prediction maps are cached in
``reports/qualitative_grid_predictions``.
"""

from __future__ import annotations

import argparse
import io
import json
import math
import sys
import zipfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yaml
from PIL import Image

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from utils.visualize_sthelar_predictions import (  # noqa: E402
    PatchData,
    Prediction,
    _dense_npz,
    _instance_majority_types,
    instance_boundary,
    load_yaml,
    pair_instance_ids,
    predict_selected,
    resolve_zip_member,
)


TISSUES = [
    "Kidney",
    "Liver",
    "Tonsil",
    "Ovary",
    "Breast",
    "Colon",
    "Lung",
    "Pancreatic",
    "Skin",
]

PEFT_RUNS = {
    "Kidney": "run/sthelar40x_kidney_5class_spatial_margin128_lora_adaptformer_r8_a8_red16_decoder_heads_only_lr5e-5_e10_seed43_CLEAN/log/2026-06-30T232732_sthelar40x_kidney_5class_spatial_margin128_lora_adaptformer_r8_a8_red16_decoder_heads_only_lr5e-5_e10_seed43_CLEAN",
    "Liver": "run/sthelar40x_liver_5class_spatial_margin128_lora_adaptformer_r8_a8_red16_decoder_heads_only_lr5e-5_e10_seed43_CLEAN/log/2026-06-30T232558_sthelar40x_liver_5class_spatial_margin128_lora_adaptformer_r8_a8_red16_decoder_heads_only_lr5e-5_e10_seed43_CLEAN",
    "Tonsil": "run/sthelar40x_tonsil_5class_spatial_margin128_cap50000_lora_adaptformer_r8_a8_red16_decoder_heads_only_lr5e-5_e10_seed43_CLEAN/log/2026-06-30T232731_sthelar40x_tonsil_5class_spatial_margin128_cap50000_lora_adaptformer_r8_a8_red16_decoder_heads_only_lr5e-5_e10_seed43_CLEAN",
    "Ovary": "run/sthelar40x_ovary_5class_spatial_margin128_lora_adaptformer_r8_a8_red16_decoder_heads_only_lr5e-5_e10_seed43_CLEAN/log/2026-06-30T234638_sthelar40x_ovary_5class_spatial_margin128_lora_adaptformer_r8_a8_red16_decoder_heads_only_lr5e-5_e10_seed43_CLEAN",
    "Breast": "run/sthelar40x_breast_5class_spatial_margin128_cap50000_lora_adaptformer_r8_a8_red16_decoder_heads_only_lr5e-5_e10_seed42_CLEAN/log/2026-06-30T170944_sthelar40x_breast_5class_spatial_margin128_cap50000_lora_adaptformer_r8_a8_red16_decoder_heads_only_lr5e-5_e10_seed42_CLEAN",
    "Colon": "run/sthelar40x_colon_5class_spatial_margin128_cap50000_lora_adaptformer_r8_a8_red16_decoder_heads_only_lr5e-5_e10_seed43_CLEAN/log/2026-06-30T234437_sthelar40x_colon_5class_spatial_margin128_cap50000_lora_adaptformer_r8_a8_red16_decoder_heads_only_lr5e-5_e10_seed43_CLEAN",
    "Lung": "run/sthelar40x_lung_5class_spatial_margin128_cap50000_lora_adaptformer_r8_a8_red16_decoder_heads_only_lr5e-5_e10_seed43_CLEAN/log/2026-06-30T234638_sthelar40x_lung_5class_spatial_margin128_cap50000_lora_adaptformer_r8_a8_red16_decoder_heads_only_lr5e-5_e10_seed43_CLEAN",
    "Pancreatic": "run/sthelar40x_pancreatic_5class_spatial_margin128_cap50000_lora_adaptformer_r8_a8_red16_decoder_heads_only_lr5e-5_e10_seed43_CLEAN/log/2026-06-30T235608_sthelar40x_pancreatic_5class_spatial_margin128_cap50000_lora_adaptformer_r8_a8_red16_decoder_heads_only_lr5e-5_e10_seed43_CLEAN",
    "Skin": "run/sthelar40x_skin_5class_spatial_margin128_cap50000_lora_adaptformer_r8_a8_red16_decoder_heads_only_lr5e-5_e10_seed43_CLEAN/log/2026-07-01T071017_sthelar40x_skin_5class_spatial_margin128_cap50000_lora_adaptformer_r8_a8_red16_decoder_heads_only_lr5e-5_e10_seed43_CLEAN",
}

FROZEN_RUN = "run/sthelar40x_kidney_liver_tonsil_5class_spatial_margin128_freeze_e10_seed42_CLEAN/log/2026-06-24T195646_sthelar40x_kidney_liver_tonsil_5class_spatial_margin128_freeze_e10_seed42_CLEAN"

PALETTE = {
    1: ("Immune", "#0072B2"),
    2: ("Stromal", "#E69F00"),
    3: ("Epithelial", "#D55E00"),
    4: ("Melanocyte", "#CC79A7"),
    5: ("Other", "#7F7F7F"),
}


@dataclass
class Candidate:
    tissue: str
    patch: PatchData
    gt_count: int
    tissue_fraction: float
    center_count: int
    peft_json_mpq: float
    peft_json_bpq: float
    pre_score: float


@dataclass
class Selection:
    tissue: str
    patch: PatchData
    frozen: Prediction
    peft: Prediction
    metrics: dict[str, Any]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--shortlist", type=int, default=4)
    parser.add_argument("--min-gt-nuclei", type=int, default=20)
    parser.add_argument("--cache-dir", type=Path, default=Path("reports/qualitative_grid_predictions"))
    parser.add_argument("--metadata", type=Path, default=Path("reports/qualitative_grid_patch_selection.md"))
    parser.add_argument("--figure-dir", type=Path, default=Path("figures"))
    parser.add_argument("--dpi", type=int, default=300)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def rel(path: Path) -> Path:
    return path.resolve().relative_to(REPO_ROOT)


def run_paths(run_dir: str) -> tuple[Path, Path, Path]:
    base = REPO_ROOT / run_dir
    return base / "config.yaml", base / "checkpoints" / "model_best.pth", base / "inference_results.json"


def read_class_names(dataset_root: Path) -> dict[int, str]:
    config = load_yaml(dataset_root / "dataset_config.yaml")
    raw = config["nuclei_types"]
    if all(isinstance(value, int) for value in raw.values()):
        return {int(value): str(key) for key, value in raw.items()}
    return {int(key): str(value) for key, value in raw.items()}


def gt_count(instance_map: np.ndarray) -> int:
    return max(0, len(np.unique(instance_map)) - (1 if np.any(instance_map == 0) else 0))


def center_instance_count(instance_map: np.ndarray) -> int:
    h, w = instance_map.shape
    crop = instance_map[h // 4 : 3 * h // 4, w // 4 : 3 * w // 4]
    return gt_count(crop)


def tissue_fraction(image: np.ndarray) -> float:
    rgb = image.astype(np.float32) / 255.0
    white = (rgb[..., 0] > 0.86) & (rgb[..., 1] > 0.86) & (rgb[..., 2] > 0.86)
    return float(1.0 - white.mean())


def load_patch(dataset_root: Path, row: pd.Series) -> PatchData:
    with zipfile.ZipFile(dataset_root / "images.zip") as images, zipfile.ZipFile(
        dataset_root / "labels.zip"
    ) as labels:
        image_names = [name for name in images.namelist() if not Path(name).name.startswith("._")]
        label_names = [name for name in labels.namelist() if not Path(name).name.startswith("._")]
        image_member = resolve_zip_member(images, str(row["packed_file_name"]), image_names)
        label_name = str(row.get("packed_label_name", Path(str(row["packed_file_name"])).with_suffix(".npz").name))
        label_member = resolve_zip_member(labels, label_name, label_names)
        with images.open(image_member) as handle:
            image = np.asarray(Image.open(io.BytesIO(handle.read())).convert("RGB"))
        with labels.open(label_member) as handle:
            with np.load(io.BytesIO(handle.read()), allow_pickle=False) as data:
                instance = _dense_npz(data, "inst_map").astype(np.int32)
                type_map = _dense_npz(data, "type_map").astype(np.int16)
    return PatchData(row=row, image=image, gt_instance=instance, gt_type=type_map)


def metric_value(metrics: dict[str, Any], key: str) -> float:
    value = metrics.get(key, float("nan"))
    if isinstance(value, (int, float)) and not math.isnan(value):
        return float(value)
    return float("nan")


def candidate_rows(tissue: str, config_path: Path, results_path: Path, shortlist: int, min_gt: int) -> list[Candidate]:
    config = load_yaml(config_path)
    dataset_root = Path(config["data"]["dataset_path"]).expanduser().resolve()
    patch_info = pd.read_csv(dataset_root / "patch_info_with_split.csv")
    patch_lookup = {
        str(row["packed_file_name"]): row
        for _, row in patch_info.iterrows()
        if str(row.get("split", "")).lower() == "test"
    }
    with results_path.open() as handle:
        image_metrics = json.load(handle)["image_metrics"]
    ranked = []
    for patch_id, metrics in image_metrics.items():
        if patch_id not in patch_lookup:
            continue
        mpq = metric_value(metrics, "mPQ")
        bpq = metric_value(metrics, "bPQ")
        jaccard = metric_value(metrics, "Jaccard")
        if math.isnan(mpq):
            mpq = -1.0
        ranked.append((mpq, bpq if not math.isnan(bpq) else -1.0, jaccard if not math.isnan(jaccard) else -1.0, patch_id))
    ranked.sort(reverse=True)

    candidates: list[Candidate] = []
    for _, _, _, patch_id in ranked[: max(200, shortlist * 50)]:
        patch = load_patch(dataset_root, patch_lookup[patch_id])
        count = gt_count(patch.gt_instance)
        center_count = center_instance_count(patch.gt_instance)
        foreground = tissue_fraction(patch.image)
        if count < min_gt or center_count < 3 or foreground < 0.35:
            continue
        metrics = image_metrics[patch_id]
        mpq = max(0.0, metric_value(metrics, "mPQ"))
        bpq = max(0.0, metric_value(metrics, "bPQ"))
        score = 2.0 * mpq + 0.35 * bpq + 0.015 * min(count, 80) + 0.3 * foreground
        candidates.append(
            Candidate(
                tissue=tissue,
                patch=patch,
                gt_count=count,
                tissue_fraction=foreground,
                center_count=center_count,
                peft_json_mpq=mpq,
                peft_json_bpq=bpq,
                pre_score=score,
            )
        )
        if len(candidates) >= shortlist:
            break
    if not candidates:
        raise RuntimeError(f"No candidate patch found for {tissue}")
    return candidates


def type_accuracy(gt: PatchData, pred: Prediction, threshold: float = 0.30) -> tuple[float, int]:
    pairs, _, _ = pair_instance_ids(gt.gt_instance, pred.instance, threshold)
    true_types = _instance_majority_types(gt.gt_instance, gt.gt_type)
    pred_types = _instance_majority_types(pred.instance, pred.type_map)
    if not pairs:
        return 0.0, 0
    correct = 0
    for pred_id, (true_id, _) in pairs.items():
        correct += int(pred_types.get(pred_id, 0) == true_types.get(true_id, 0))
    return correct / len(pairs), len(pairs)


def render_type_map(instance: np.ndarray, type_map: np.ndarray, boundary_width: int = 2) -> np.ndarray:
    out = np.ones((*instance.shape, 3), dtype=np.float32)
    for cls_id, (_, color) in PALETTE.items():
        rgb = np.array(matplotlib.colors.to_rgb(color), dtype=np.float32)
        mask = (instance > 0) & (type_map == cls_id)
        out[mask] = 0.08 + 0.92 * rgb
    boundary = instance_boundary(instance, boundary_width)
    out[boundary] = np.array([0.08, 0.08, 0.08], dtype=np.float32)
    return np.clip(out, 0, 1)


def build_selection(args: argparse.Namespace) -> list[Selection]:
    all_candidates: dict[str, list[Candidate]] = {}
    for tissue in TISSUES:
        config_path, _, results_path = run_paths(PEFT_RUNS[tissue])
        all_candidates[tissue] = candidate_rows(
            tissue, config_path, results_path, args.shortlist, args.min_gt_nuclei
        )
        print(
            f"{tissue}: shortlisted "
            + ", ".join(candidate.patch.patch_id for candidate in all_candidates[tissue])
        )

    frozen_config, frozen_checkpoint, _ = run_paths(FROZEN_RUN)
    all_patches = [candidate.patch for candidates in all_candidates.values() for candidate in candidates]
    frozen = predict_selected(
        frozen_config,
        frozen_checkpoint,
        all_patches,
        args.cache_dir / "frozen",
        args.device,
        args.batch_size,
        force=False,
    )

    peft_predictions: dict[str, Prediction] = {}
    for tissue, candidates in all_candidates.items():
        config_path, checkpoint_path, _ = run_paths(PEFT_RUNS[tissue])
        predictions = predict_selected(
            config_path,
            checkpoint_path,
            [candidate.patch for candidate in candidates],
            args.cache_dir / "peft" / tissue.lower(),
            args.device,
            args.batch_size,
            force=False,
        )
        peft_predictions.update(predictions)

    selections: list[Selection] = []
    for tissue, candidates in all_candidates.items():
        ranked = []
        for candidate in candidates:
            patch_id = candidate.patch.patch_id
            frozen_acc, frozen_matched = type_accuracy(candidate.patch, frozen[patch_id])
            peft_acc, peft_matched = type_accuracy(candidate.patch, peft_predictions[patch_id])
            matched = min(frozen_matched, peft_matched)
            improvement = peft_acc - frozen_acc
            score = (
                3.0 * improvement
                + 1.2 * peft_acc
                + 0.025 * min(matched, 80)
                + 0.5 * candidate.peft_json_mpq
                + 0.15 * candidate.tissue_fraction
            )
            ranked.append(
                (
                    score,
                    Selection(
                        tissue=tissue,
                        patch=candidate.patch,
                        frozen=frozen[patch_id],
                        peft=peft_predictions[patch_id],
                        metrics={
                            "gt_nuclei": candidate.gt_count,
                            "center_nuclei": candidate.center_count,
                            "tissue_fraction": candidate.tissue_fraction,
                            "peft_json_mPQ": candidate.peft_json_mpq,
                            "peft_json_bPQ": candidate.peft_json_bpq,
                            "frozen_type_accuracy": frozen_acc,
                            "peft_type_accuracy": peft_acc,
                            "type_accuracy_improvement": improvement,
                            "frozen_matched_nuclei": frozen_matched,
                            "peft_matched_nuclei": peft_matched,
                            "selection_score": score,
                        },
                    ),
                )
            )
        ranked.sort(key=lambda item: item[0], reverse=True)
        selections.append(ranked[0][1])
    return selections


def save_grid(
    selections: list[Selection],
    tissues: list[str],
    output_base: Path,
    dpi: int,
    overwrite: bool,
) -> None:
    for suffix in ("pdf", "png"):
        path = output_base.with_suffix(f".{suffix}")
        if path.exists() and not overwrite:
            raise FileExistsError(f"Refusing to overwrite existing figure: {path}")

    selected_by_tissue = {selection.tissue: selection for selection in selections}
    ncols = len(tissues)
    fig_width = 7.05 if ncols == 6 else 10.7
    fig_height = 4.35 if ncols == 6 else 4.55
    fig, axes = plt.subplots(3, ncols, figsize=(fig_width, fig_height), squeeze=False)
    row_labels = ["Ground truth", "Frozen CellViT", "PEFT adapter"]
    for col, tissue in enumerate(tissues):
        selection = selected_by_tissue[tissue]
        panels = [
            render_type_map(selection.patch.gt_instance, selection.patch.gt_type),
            render_type_map(selection.frozen.instance, selection.frozen.type_map),
            render_type_map(selection.peft.instance, selection.peft.type_map),
        ]
        for row, panel in enumerate(panels):
            ax = axes[row, col]
            ax.imshow(panel, interpolation="nearest")
            ax.set_xticks([])
            ax.set_yticks([])
            for spine in ax.spines.values():
                spine.set_linewidth(0.35)
                spine.set_color("#D0D0D0")
            if row == 0:
                ax.set_title(tissue, fontsize=9 if ncols == 6 else 8, pad=4)
            if col == 0:
                ax.set_ylabel(row_labels[row], fontsize=9, labelpad=8)

    handles = [
        mpatches.Patch(facecolor=color, edgecolor="black", linewidth=0.4, label=name)
        for _, (name, color) in PALETTE.items()
    ]
    fig.legend(
        handles=handles,
        loc="lower center",
        ncol=5,
        frameon=False,
        fontsize=8,
        bbox_to_anchor=(0.5, 0.012),
        handlelength=1.2,
        columnspacing=1.1,
    )
    fig.subplots_adjust(
        left=0.080 if ncols == 6 else 0.055,
        right=0.995,
        top=0.925,
        bottom=0.115,
        wspace=0.035,
        hspace=0.055,
    )
    output_base.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_base.with_suffix(".pdf"), bbox_inches="tight", facecolor="white")
    fig.savefig(output_base.with_suffix(".png"), dpi=dpi, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def save_metadata(selections: list[Selection], main_tissues: list[str], output_path: Path, overwrite: bool) -> None:
    if output_path.exists() and not overwrite:
        raise FileExistsError(f"Refusing to overwrite existing metadata: {output_path}")
    lines = [
        "# Qualitative Grid Patch Selection",
        "",
        "Selection used 5-class STHELAR spatial test patches. Candidate patches were ranked by PEFT type-map quality, foreground fraction, ground-truth nuclei count, and post-inference matched-nuclei type-accuracy improvement over Frozen CellViT.",
        "",
        f"Main-paper tissues: {', '.join(main_tissues)}",
        "",
        "| Tissue | Patch ID | Slide | GT nuclei | Center nuclei | Tissue fraction | Frozen type acc. | PEFT type acc. | Improvement | Frozen matched | PEFT matched | PEFT mPQ JSON | PEFT bPQ JSON | Selection rationale |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|",
    ]
    for selection in selections:
        row = selection.patch.row
        metrics = selection.metrics
        slide = str(row.get("slide_id", Path(selection.patch.patch_id).stem.split("__")[0]))
        rationale = (
            "selected for high PEFT-vs-frozen matched type improvement "
            "with sufficient nuclei and foreground"
        )
        lines.append(
            "| {tissue} | `{patch}` | {slide} | {gt} | {center} | {fg:.3f} | "
            "{fa:.3f} | {pa:.3f} | {imp:.3f} | {fm} | {pm} | {mpq:.3f} | {bpq:.3f} | {rationale} |".format(
                tissue=selection.tissue,
                patch=selection.patch.patch_id,
                slide=slide,
                gt=int(metrics["gt_nuclei"]),
                center=int(metrics["center_nuclei"]),
                fg=float(metrics["tissue_fraction"]),
                fa=float(metrics["frozen_type_accuracy"]),
                pa=float(metrics["peft_type_accuracy"]),
                imp=float(metrics["type_accuracy_improvement"]),
                fm=int(metrics["frozen_matched_nuclei"]),
                pm=int(metrics["peft_matched_nuclei"]),
                mpq=float(metrics["peft_json_mPQ"]),
                bpq=float(metrics["peft_json_bPQ"]),
                rationale=rationale,
            )
        )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text("\n".join(lines) + "\n")


def main() -> None:
    args = parse_args()
    final_outputs = [
        args.figure_dir / "qualitative_grid_3x6_main.pdf",
        args.figure_dir / "qualitative_grid_3x6_main.png",
        args.figure_dir / "qualitative_grid_3x9_supplement.pdf",
        args.figure_dir / "qualitative_grid_3x9_supplement.png",
        args.metadata,
    ]
    existing = [path for path in final_outputs if path.exists()]
    if existing and not args.overwrite:
        raise FileExistsError(
            "Refusing to overwrite existing outputs: "
            + ", ".join(str(rel(path)) for path in existing)
        )

    selections = build_selection(args)
    selections_by_score = sorted(
        selections, key=lambda item: item.metrics["selection_score"], reverse=True
    )
    main_tissues = [selection.tissue for selection in selections_by_score[:6]]
    main_tissues = [tissue for tissue in TISSUES if tissue in set(main_tissues)]

    save_grid(
        selections,
        main_tissues,
        args.figure_dir / "qualitative_grid_3x6_main",
        args.dpi,
        args.overwrite,
    )
    save_grid(
        selections,
        TISSUES,
        args.figure_dir / "qualitative_grid_3x9_supplement",
        args.dpi,
        args.overwrite,
    )
    save_metadata(selections, main_tissues, args.metadata, args.overwrite)

    print("Selected patches:")
    for selection in selections:
        metrics = selection.metrics
        print(
            f"{selection.tissue}: {selection.patch.patch_id} "
            f"gt={metrics['gt_nuclei']} "
            f"frozen_acc={metrics['frozen_type_accuracy']:.3f} "
            f"peft_acc={metrics['peft_type_accuracy']:.3f} "
            f"delta={metrics['type_accuracy_improvement']:.3f}"
        )
    print("Wrote figures and metadata.")


if __name__ == "__main__":
    main()
