#!/usr/bin/env python3
"""Build the STHELAR 3x6 qualitative grid for the workshop paper.

This script is intentionally a thin wrapper around
``utils.visualize_sthelar_predictions``.  It reuses the existing data loading,
checkpoint loading, adapter insertion, prediction caching, palette, and boundary
utilities, then composes a compact multi-tissue figure.
"""

from __future__ import annotations

import argparse
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from visualize_sthelar_predictions import (
    PatchData,
    Prediction,
    _flatten_patch_ids,
    _identifier_set,
    _instance_majority_types,
    class_names_from_config,
    color_table,
    instance_boundary,
    load_patches,
    load_yaml,
    pair_instance_ids,
    predict_selected,
)


REPO_ROOT = Path(__file__).resolve().parents[1]

FROZEN_CONFIG = Path(
    "run/sthelar40x_kidney_liver_tonsil_5class_spatial_margin128_freeze_e10_seed42_CLEAN/"
    "log/2026-06-24T195646_sthelar40x_kidney_liver_tonsil_5class_spatial_margin128_freeze_e10_seed42_CLEAN/"
    "config.yaml"
)
FROZEN_CHECKPOINT = FROZEN_CONFIG.parent / "checkpoints/model_best.pth"
FROZEN_RESULTS = FROZEN_CONFIG.parent / "inference_results.json"

LINEAR_PROBING_CONFIG = Path(
    "run/sthelar40x_kidney_liver_tonsil_5class_spatial_margin128_klt_final_heads_only_frozen_encoder_decoder_e10_seed42/"
    "log/2026-07-02T153245_sthelar40x_kidney_liver_tonsil_5class_spatial_margin128_klt_final_heads_only_frozen_encoder_decoder_e10_seed42/"
    "config.yaml"
)
LINEAR_PROBING_CHECKPOINT = LINEAR_PROBING_CONFIG.parent / "checkpoints/model_best.pth"


@dataclass(frozen=True)
class TissueRun:
    name: str
    config: Path
    known_patch: str | None = None

    @property
    def checkpoint(self) -> Path:
        return self.config.parent / "checkpoints/model_best.pth"

    @property
    def results(self) -> Path:
        return self.config.parent / "inference_results.json"

    @property
    def run_label(self) -> str:
        return self.config.parent.name


TISSUES = [
    TissueRun(
        "Liver",
        Path(
            "run/sthelar40x_liver_5class_spatial_margin128_lora_adaptformer_r8_a8_red16_decoder_heads_only_lr5e-5_e10_seed42_CLEAN/"
            "log/2026-06-28T133237_sthelar40x_liver_5class_spatial_margin128_lora_adaptformer_r8_a8_red16_decoder_heads_only_lr5e-5_e10_seed42_CLEAN/"
            "config.yaml"
        ),
    ),
    TissueRun(
        "Ovary",
        Path(
            "run/sthelar40x_ovary_5class_spatial_margin128_lora_adaptformer_r8_a8_red16_decoder_heads_only_lr5e-5_e10_seed42_CLEAN/"
            "log/2026-06-28T203221_sthelar40x_ovary_5class_spatial_margin128_lora_adaptformer_r8_a8_red16_decoder_heads_only_lr5e-5_e10_seed42_CLEAN/"
            "config.yaml"
        ),
    ),
    TissueRun(
        "Breast",
        Path(
            "run/sthelar40x_breast_5class_spatial_margin128_cap50000_lora_adaptformer_r8_a8_red16_decoder_heads_only_lr5e-5_e10_seed42_CLEAN/"
            "log/2026-06-30T170944_sthelar40x_breast_5class_spatial_margin128_cap50000_lora_adaptformer_r8_a8_red16_decoder_heads_only_lr5e-5_e10_seed42_CLEAN/"
            "config.yaml"
        ),
    ),
    TissueRun(
        "Lung",
        Path(
            "run/sthelar40x_lung_5class_spatial_margin128_cap50000_lora_adaptformer_r8_a8_red16_decoder_heads_only_lr5e-5_e10_seed42_CLEAN/"
            "log/2026-07-01T025800_sthelar40x_lung_5class_spatial_margin128_cap50000_lora_adaptformer_r8_a8_red16_decoder_heads_only_lr5e-5_e10_seed42_CLEAN/"
            "config.yaml"
        ),
    ),
    TissueRun(
        "Kidney",
        Path(
            "run/sthelar40x_kidney_liver_tonsil_5class_spatial_margin128_lora_adaptformer_r8_a8_red16_decoder_heads_only_lr5e-5_e10_seed42_CLEAN/"
            "log/2026-06-25T113302_sthelar40x_kidney_liver_tonsil_5class_spatial_margin128_lora_adaptformer_r8_a8_red16_decoder_heads_only_lr5e-5_e10_seed42_CLEAN/"
            "config.yaml"
        ),
        known_patch="kidney_s1__x7680_y9216__kidney_s1_2248.png",
    ),
    TissueRun(
        "Tonsil",
        Path(
            "run/sthelar40x_tonsil_5class_spatial_margin128_cap50000_lora_adaptformer_r8_a8_red16_decoder_heads_only_lr5e-5_e10_seed43_CLEAN/"
            "log/2026-06-30T232731_sthelar40x_tonsil_5class_spatial_margin128_cap50000_lora_adaptformer_r8_a8_red16_decoder_heads_only_lr5e-5_e10_seed43_CLEAN/"
            "config.yaml"
        ),
    ),
]


SUPPLEMENT_TISSUES = [
    TissueRun(
        "Kidney",
        Path(
            "run/sthelar40x_kidney_liver_tonsil_5class_spatial_margin128_lora_adaptformer_r8_a8_red16_decoder_heads_only_lr5e-5_e10_seed42_CLEAN/"
            "log/2026-06-25T113302_sthelar40x_kidney_liver_tonsil_5class_spatial_margin128_lora_adaptformer_r8_a8_red16_decoder_heads_only_lr5e-5_e10_seed42_CLEAN/"
            "config.yaml"
        ),
    ),
    TissueRun(
        "Liver",
        Path(
            "run/sthelar40x_liver_5class_spatial_margin128_lora_adaptformer_r8_a8_red16_decoder_heads_only_lr5e-5_e10_seed42_CLEAN/"
            "log/2026-06-28T133237_sthelar40x_liver_5class_spatial_margin128_lora_adaptformer_r8_a8_red16_decoder_heads_only_lr5e-5_e10_seed42_CLEAN/"
            "config.yaml"
        ),
    ),
    TissueRun(
        "Tonsil",
        Path(
            "run/sthelar40x_tonsil_5class_spatial_margin128_cap50000_lora_adaptformer_r8_a8_red16_decoder_heads_only_lr5e-5_e10_seed43_CLEAN/"
            "log/2026-06-30T232731_sthelar40x_tonsil_5class_spatial_margin128_cap50000_lora_adaptformer_r8_a8_red16_decoder_heads_only_lr5e-5_e10_seed43_CLEAN/"
            "config.yaml"
        ),
    ),
    TissueRun(
        "Ovary",
        Path(
            "run/sthelar40x_ovary_5class_spatial_margin128_lora_adaptformer_r8_a8_red16_decoder_heads_only_lr5e-5_e10_seed42_CLEAN/"
            "log/2026-06-28T203221_sthelar40x_ovary_5class_spatial_margin128_lora_adaptformer_r8_a8_red16_decoder_heads_only_lr5e-5_e10_seed42_CLEAN/"
            "config.yaml"
        ),
    ),
    TissueRun(
        "Breast",
        Path(
            "run/sthelar40x_breast_5class_spatial_margin128_cap50000_lora_adaptformer_r8_a8_red16_decoder_heads_only_lr5e-5_e10_seed42_CLEAN/"
            "log/2026-06-30T170944_sthelar40x_breast_5class_spatial_margin128_cap50000_lora_adaptformer_r8_a8_red16_decoder_heads_only_lr5e-5_e10_seed42_CLEAN/"
            "config.yaml"
        ),
    ),
    TissueRun(
        "Colon",
        Path(
            "run/sthelar40x_colon_5class_spatial_margin128_cap50000_lora_adaptformer_r8_a8_red16_decoder_heads_only_lr5e-5_e10_seed42_CLEAN/"
            "log/2026-06-30T203052_sthelar40x_colon_5class_spatial_margin128_cap50000_lora_adaptformer_r8_a8_red16_decoder_heads_only_lr5e-5_e10_seed42_CLEAN/"
            "config.yaml"
        ),
    ),
    TissueRun(
        "Lung",
        Path(
            "run/sthelar40x_lung_5class_spatial_margin128_cap50000_lora_adaptformer_r8_a8_red16_decoder_heads_only_lr5e-5_e10_seed42_CLEAN/"
            "log/2026-07-01T025800_sthelar40x_lung_5class_spatial_margin128_cap50000_lora_adaptformer_r8_a8_red16_decoder_heads_only_lr5e-5_e10_seed42_CLEAN/"
            "config.yaml"
        ),
    ),
    TissueRun(
        "Pancreatic",
        Path(
            "run/sthelar40x_pancreatic_5class_spatial_margin128_cap50000_lora_adaptformer_r8_a8_red16_decoder_heads_only_lr5e-5_e10_seed43_CLEAN/"
            "log/2026-06-30T235608_sthelar40x_pancreatic_5class_spatial_margin128_cap50000_lora_adaptformer_r8_a8_red16_decoder_heads_only_lr5e-5_e10_seed43_CLEAN/"
            "config.yaml"
        ),
    ),
    TissueRun(
        "Skin",
        Path(
            "run/sthelar40x_skin_5class_spatial_margin128_cap50000_lora_adaptformer_r8_a8_red16_decoder_heads_only_lr5e-5_e10_seed42_CLEAN/"
            "log/2026-06-30T092331_sthelar40x_skin_5class_spatial_margin128_cap50000_lora_adaptformer_r8_a8_red16_decoder_heads_only_lr5e-5_e10_seed42_CLEAN/"
            "config.yaml"
        ),
    ),
]


@dataclass
class Candidate:
    tissue: TissueRun
    patch: PatchData
    peft_image_metrics: dict[str, Any]
    gt_nuclei: int
    tissue_fraction: float
    central_tissue_fraction: float


@dataclass
class SelectedPatch:
    candidate: Candidate
    frozen: Prediction
    peft: Prediction
    frozen_type_accuracy: float
    peft_type_accuracy: float
    improvement: float
    frozen_matched: int
    peft_matched: int


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Create STHELAR qualitative 3x6 grid.")
    parser.add_argument(
        "--grid",
        choices=("main-3x6", "supplement-3x9"),
        default="main-3x6",
    )
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--shortlist", type=int, default=4)
    parser.add_argument("--scan-limit", type=int, default=80)
    parser.add_argument("--bg-alpha", type=float, default=0.33)
    parser.add_argument("--mask-alpha", type=float, default=0.90)
    parser.add_argument("--boundary-width", type=int, default=3)
    parser.add_argument("--dpi", type=int, default=600)
    parser.add_argument(
        "--middle-row",
        choices=("frozen", "linear-probing"),
        default="frozen",
        help="Prediction variant to render in the second row.",
    )
    parser.add_argument(
        "--reuse-selection",
        type=Path,
        default=None,
        help="Reuse tissue/patch IDs from an existing selection markdown table.",
    )
    parser.add_argument("--force-inference", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def _resolve(path: Path) -> Path:
    return (REPO_ROOT / path).resolve()


def _metric(value: Any, default: float = 0.0) -> float:
    if isinstance(value, (int, float)) and not math.isnan(float(value)):
        return float(value)
    return default


def _load_image_metrics(path: Path) -> dict[str, dict[str, Any]]:
    import json

    with _resolve(path).open() as handle:
        return json.load(handle)["image_metrics"]


def _load_patch_info(dataset_root: Path) -> pd.DataFrame:
    return pd.read_csv(dataset_root / "patch_info_with_split.csv")


def _find_row(patch_info: pd.DataFrame, patch_id: str) -> pd.Series:
    hits = patch_info.apply(lambda row: patch_id in _identifier_set(row), axis=1)
    if not hits.any():
        raise ValueError(f"Patch ID not found in metadata: {patch_id}")
    return patch_info.loc[hits].iloc[0]


def _build_patch_lookup(patch_info: pd.DataFrame) -> dict[str, int]:
    lookup: dict[str, int] = {}
    for index, row in patch_info.iterrows():
        for column in ("packed_file_name", "packed_label_name", "file_name", "patch_id"):
            if column not in row or pd.isna(row[column]):
                continue
            value = str(row[column])
            lookup.setdefault(value, index)
            lookup.setdefault(Path(value).name, index)
            lookup.setdefault(Path(value).stem, index)
    return lookup


def _patch_ids_from_metrics(metrics: dict[str, dict[str, Any]]) -> list[str]:
    return sorted(
        metrics,
        key=lambda patch_id: (
            _metric(metrics[patch_id].get("mPQ")),
            _metric(metrics[patch_id].get("bPQ")),
            _metric(metrics[patch_id].get("Jaccard")),
        ),
        reverse=True,
    )


def _foreground_fraction(image: np.ndarray) -> float:
    rgb = image.astype(np.float32)
    not_white = np.mean(rgb, axis=2) < 235
    return float(np.mean(not_white))


def _central_foreground_fraction(image: np.ndarray) -> float:
    h, w = image.shape[:2]
    crop = image[h // 4 : 3 * h // 4, w // 4 : 3 * w // 4]
    return _foreground_fraction(crop)


def _gt_nuclei(patch: PatchData) -> int:
    ids = np.unique(patch.gt_instance)
    return int(np.sum(ids != 0))


def _load_single_patch(dataset_root: Path, patch_info: pd.DataFrame, patch_id: str) -> PatchData:
    row = _find_row(patch_info, patch_id)
    return load_patches(dataset_root, pd.DataFrame([row]))[0]


def _load_fixed_selection(path: Path) -> dict[str, str]:
    fixed: dict[str, str] = {}
    with path.open() as handle:
        for line in handle:
            if not line.startswith("| "):
                continue
            cells = [cell.strip() for cell in line.strip().strip("|").split("|")]
            if len(cells) < 2 or cells[0] in {"Tissue", "---"}:
                continue
            fixed[cells[0]] = cells[1]
    return fixed


def fixed_candidate(tissue: TissueRun, patch_id: str) -> Candidate:
    config = load_yaml(_resolve(tissue.config))
    dataset_root = Path(config["data"]["dataset_path"]).expanduser().resolve()
    patch_info = _load_patch_info(dataset_root)
    metrics = _load_image_metrics(tissue.results)
    patch = _load_single_patch(dataset_root, patch_info, patch_id)
    patch_metrics = metrics.get(patch.patch_id, metrics.get(Path(patch.patch_id).name, {}))
    return Candidate(
        tissue,
        patch,
        patch_metrics,
        _gt_nuclei(patch),
        _foreground_fraction(patch.image),
        _central_foreground_fraction(patch.image),
    )


def choose_shortlist(tissue: TissueRun, shortlist: int, scan_limit: int) -> list[Candidate]:
    config = load_yaml(_resolve(tissue.config))
    dataset_root = Path(config["data"]["dataset_path"]).expanduser().resolve()
    patch_info = _load_patch_info(dataset_root)
    patch_lookup = _build_patch_lookup(patch_info)
    metrics = _load_image_metrics(tissue.results)
    candidates: list[Candidate] = []
    tissue_prefix = tissue.name.lower()

    ordered_ids = _patch_ids_from_metrics(metrics)
    if tissue.known_patch:
        ordered_ids = [tissue.known_patch] + [item for item in ordered_ids if item != tissue.known_patch]

    rows: list[pd.Series] = []
    seen: set[str] = set()
    for patch_id in ordered_ids[: max(scan_limit, shortlist)]:
        if patch_id in seen:
            continue
        lookup_key = next(
            (key for key in (patch_id, Path(patch_id).name, Path(patch_id).stem) if key in patch_lookup),
            None,
        )
        if lookup_key is None:
            continue
        row = patch_info.loc[patch_lookup[lookup_key]]
        identifiers = {str(value).lower() for value in _identifier_set(row)}
        slide_id = str(row.get("slide_id", "")).lower()
        if slide_id and not slide_id.startswith(tissue_prefix):
            continue
        if not slide_id and not any(identifier.startswith(tissue_prefix) for identifier in identifiers):
            continue
        rows.append(row)
        seen.add(patch_id)
    patches = load_patches(dataset_root, pd.DataFrame(rows)) if rows else []

    for patch in patches:
        patch_id = patch.patch_id
        gt_n = _gt_nuclei(patch)
        tissue_fraction = _foreground_fraction(patch.image)
        central_fraction = _central_foreground_fraction(patch.image)
        patch_metrics = metrics.get(patch_id, metrics.get(Path(patch_id).name, {}))
        if tissue.known_patch and patch_id == tissue.known_patch:
            candidates.append(
                Candidate(tissue, patch, patch_metrics, gt_n, tissue_fraction, central_fraction)
            )
            continue
        if gt_n < 12 or tissue_fraction < 0.30 or central_fraction < 0.25:
            continue
        candidates.append(
            Candidate(tissue, patch, patch_metrics, gt_n, tissue_fraction, central_fraction)
        )
        if len(candidates) >= shortlist:
            break

    if not candidates:
        raise RuntimeError(f"No candidate patches found for {tissue.name}")
    print(
        f"{tissue.name}: shortlisted {len(candidates)} patch(es) from {len(rows)} scanned candidates",
        flush=True,
    )
    return candidates[:shortlist]


def matched_type_accuracy(gt: PatchData, pred: Prediction) -> tuple[float, int]:
    pred_to_true, _, _ = pair_instance_ids(gt.gt_instance, pred.instance, threshold=0.30)
    true_types = _instance_majority_types(gt.gt_instance, gt.gt_type)
    pred_types = _instance_majority_types(pred.instance, pred.type_map)
    matched = 0
    correct = 0
    for pred_id, (true_id, _) in pred_to_true.items():
        true_type = true_types.get(true_id, 0)
        pred_type = pred_types.get(pred_id, 0)
        if true_type == 0:
            continue
        matched += 1
        correct += int(pred_type == true_type)
    if matched == 0:
        return 0.0, 0
    return correct / matched, matched


def select_final(
    tissues: list[TissueRun],
    candidates_by_tissue: dict[str, list[Candidate]],
    frozen_predictions: dict[str, Prediction],
    peft_predictions_by_tissue: dict[str, dict[str, Prediction]],
) -> list[SelectedPatch]:
    selected: list[SelectedPatch] = []
    for tissue in tissues:
        scored: list[tuple[float, SelectedPatch]] = []
        for candidate in candidates_by_tissue[tissue.name]:
            patch_id = candidate.patch.patch_id
            frozen = frozen_predictions[patch_id]
            peft = peft_predictions_by_tissue[tissue.name][patch_id]
            frozen_acc, frozen_matched = matched_type_accuracy(candidate.patch, frozen)
            peft_acc, peft_matched = matched_type_accuracy(candidate.patch, peft)
            improvement = peft_acc - frozen_acc
            peft_mpq = _metric(candidate.peft_image_metrics.get("mPQ"))
            peft_bpq = _metric(candidate.peft_image_metrics.get("bPQ"))
            matched_bonus = min(peft_matched, 35) / 35
            nuclei_bonus = min(candidate.gt_nuclei, 45) / 45
            score = 3.0 * improvement + peft_acc + 0.35 * matched_bonus + 0.25 * nuclei_bonus + 0.25 * peft_mpq + 0.10 * peft_bpq
            if tissue.known_patch and patch_id == tissue.known_patch:
                score += 1.0
            scored.append(
                (
                    score,
                    SelectedPatch(
                        candidate,
                        frozen,
                        peft,
                        frozen_acc,
                        peft_acc,
                        improvement,
                        frozen_matched,
                        peft_matched,
                    ),
                )
            )
        scored.sort(key=lambda item: item[0], reverse=True)
        selected.append(scored[0][1])
    return selected


def render_type_overlay(
    image: np.ndarray,
    instance_map: np.ndarray,
    type_map: np.ndarray,
    colors: np.ndarray,
    bg_alpha: float,
    mask_alpha: float,
    boundary_width: int,
) -> np.ndarray:
    base = image.astype(np.float32) / 255.0
    rgb = (1.0 - bg_alpha) * np.ones_like(base) + bg_alpha * base
    foreground = instance_map > 0
    safe_types = np.clip(type_map, 0, len(colors) - 1)
    rgb[foreground] = (1.0 - mask_alpha) * rgb[foreground] + mask_alpha * colors[safe_types[foreground]]
    boundary = instance_boundary(instance_map, boundary_width)
    rgb[boundary] = np.array([0.04, 0.04, 0.04])
    return np.clip(rgb, 0, 1)


def save_grid(
    selected: list[SelectedPatch],
    output_stem: Path,
    class_names: dict[int, str],
    bg_alpha: float,
    mask_alpha: float,
    boundary_width: int,
    dpi: int,
    middle_row_label: str = "Frozen CellViT",
) -> None:
    colors = color_table(class_names)
    rows = ["Ground truth", middle_row_label, "PEFT adapter"]
    ncols = len(selected)
    fig, axes = plt.subplots(
        3,
        ncols,
        figsize=(1.18 * ncols + 0.55, 4.05),
        squeeze=False,
        constrained_layout=False,
    )
    for col, item in enumerate(selected):
        patch = item.candidate.patch
        panels = [
            (patch.gt_instance, patch.gt_type),
            (item.frozen.instance, item.frozen.type_map),
            (item.peft.instance, item.peft.type_map),
        ]
        for row, (instance_map, type_map) in enumerate(panels):
            axes[row, col].imshow(
                render_type_overlay(
                    patch.image,
                    instance_map,
                    type_map,
                    colors,
                    bg_alpha,
                    mask_alpha,
                    boundary_width,
                )
            )
            axes[row, col].axis("off")
            if row == 0:
                axes[row, col].set_title(item.candidate.tissue.name, fontsize=9, pad=4)

    handles = [
        mpatches.Patch(facecolor=colors[index], edgecolor="black", linewidth=0.6, label=name)
        for index, name in class_names.items()
        if index != 0
    ]
    fig.legend(
        handles=handles,
        loc="lower center",
        ncol=len(handles),
        frameon=False,
        fontsize=9,
        bbox_to_anchor=(0.5, 0.018),
        columnspacing=1.2,
        handlelength=1.3,
    )
    fig.subplots_adjust(left=0.10, right=0.995, top=0.91, bottom=0.14, wspace=0.025, hspace=0.035)
    for row, label in enumerate(rows):
        bbox = axes[row, 0].get_position()
        fig.text(
            0.037,
            0.5 * (bbox.y0 + bbox.y1),
            label,
            rotation=90,
            va="center",
            ha="center",
            fontsize=9,
        )
    output_stem.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_stem.with_suffix(".pdf"), bbox_inches="tight", facecolor="white")
    fig.savefig(output_stem.with_suffix(".png"), dpi=dpi, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def _short_metric(metrics: dict[str, Any], key: str) -> str:
    if key not in metrics:
        return "n/a"
    value = metrics[key]
    if isinstance(value, (int, float)) and not math.isnan(float(value)):
        return f"{float(value):.3f}"
    return "n/a"


def write_metadata(
    selected: list[SelectedPatch],
    output_path: Path,
    middle_row_label: str = "Frozen CellViT",
) -> None:
    middle_metric_label = f"{middle_row_label} type acc."
    middle_matched_label = f"{middle_row_label} matched"
    lines = [
        f"# {output_path.stem.replace('_', ' ').title()} Patch Selection",
        "",
        f"Selection used LoRA+AdaptFormer final-head PEFT runs and {middle_row_label} as the middle-row comparison.",
        "Paths are intentionally omitted from this report to keep it anonymized.",
        "",
        f"| Tissue | Patch ID | Slide ID | GT nuclei | Tissue frac. | {middle_metric_label} | PEFT type acc. | Improvement | {middle_matched_label} | PEFT matched | PEFT bPQ | PEFT mPQ | Reason |",
        "|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|",
    ]
    for item in selected:
        patch = item.candidate.patch
        row = patch.row
        patch_id = Path(patch.patch_id).name
        slide_id = str(row.get("slide_id", patch_id.split("__")[0]))
        reason = (
            f"central tissue with visible nuclei; PEFT type map closer to GT than {middle_row_label}"
            if item.improvement > 0
            else "central tissue with visible nuclei; selected as clearest available candidate"
        )
        if item.candidate.tissue.known_patch and patch_id == item.candidate.tissue.known_patch:
            reason = "known didactic kidney example: frozen predicts Other while PEFT recovers Immune labels"
        lines.append(
            "| "
            + " | ".join(
                [
                    item.candidate.tissue.name,
                    patch_id,
                    slide_id,
                    str(item.candidate.gt_nuclei),
                    f"{item.candidate.tissue_fraction:.3f}",
                    f"{item.frozen_type_accuracy:.3f}",
                    f"{item.peft_type_accuracy:.3f}",
                    f"{item.improvement:.3f}",
                    str(item.frozen_matched),
                    str(item.peft_matched),
                    _short_metric(item.candidate.peft_image_metrics, "bPQ"),
                    _short_metric(item.candidate.peft_image_metrics, "mPQ"),
                    reason,
                ]
            )
            + " |"
        )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text("\n".join(lines) + "\n")


def ensure_outputs_do_not_exist(paths: list[Path], overwrite: bool) -> None:
    existing = [path for path in paths if path.exists()]
    if existing and not overwrite:
        raise FileExistsError(
            "Refusing to overwrite existing output(s): "
            + ", ".join(str(path.relative_to(REPO_ROOT)) for path in existing)
        )


def main() -> None:
    args = parse_args()
    if args.grid == "supplement-3x9":
        tissues = SUPPLEMENT_TISSUES
        output_stem = REPO_ROOT / "figures/qualitative_grid_3x9_supplement"
        metadata_path = REPO_ROOT / "reports/qualitative/qualitative_grid_3x9_selection.md"
        cache_root = REPO_ROOT / "reports/qualitative/grid_3x9_predictions"
        output_paths = [
            output_stem.with_suffix(".pdf"),
            output_stem.with_suffix(".png"),
            metadata_path,
        ]
    else:
        tissues = TISSUES
        output_stem = REPO_ROOT / "figures/qualitative_grid_3x6_main"
        metadata_path = REPO_ROOT / "reports/qualitative/qualitative_grid_3x6_selection.md"
        cache_root = REPO_ROOT / "reports/qualitative/grid_3x6_predictions"
        output_paths = [
            output_stem.with_suffix(".pdf"),
            output_stem.with_suffix(".png"),
            REPO_ROOT / "figures/qualitative_grid_3x3_main.pdf",
            REPO_ROOT / "figures/qualitative_grid_3x3_main.png",
            metadata_path,
        ]
    ensure_outputs_do_not_exist(output_paths, args.overwrite)

    if args.reuse_selection is not None:
        fixed_selection = _load_fixed_selection(args.reuse_selection)
        candidates_by_tissue = {
            tissue.name: [fixed_candidate(tissue, fixed_selection[tissue.name])]
            for tissue in tissues
        }
    else:
        candidates_by_tissue = {
            tissue.name: choose_shortlist(tissue, args.shortlist, args.scan_limit)
            for tissue in tissues
        }
    all_candidates = [
        candidate.patch
        for candidates in candidates_by_tissue.values()
        for candidate in candidates
    ]
    if args.middle_row == "linear-probing":
        middle_config = LINEAR_PROBING_CONFIG
        middle_checkpoint = LINEAR_PROBING_CHECKPOINT
        middle_cache_name = "linear_probing"
        middle_row_label = "Linear Probing"
    else:
        middle_config = FROZEN_CONFIG
        middle_checkpoint = FROZEN_CHECKPOINT
        middle_cache_name = "frozen"
        middle_row_label = "Frozen CellViT"

    middle_predictions = predict_selected(
        _resolve(middle_config),
        _resolve(middle_checkpoint),
        all_candidates,
        cache_root / middle_cache_name,
        args.device,
        args.batch_size,
        args.force_inference,
    )
    peft_predictions_by_tissue: dict[str, dict[str, Prediction]] = {}
    for tissue in tissues:
        peft_predictions_by_tissue[tissue.name] = predict_selected(
            _resolve(tissue.config),
            _resolve(tissue.checkpoint),
            [candidate.patch for candidate in candidates_by_tissue[tissue.name]],
            cache_root / tissue.name.lower() / "peft",
            args.device,
            args.batch_size,
            args.force_inference,
        )

    selected = select_final(tissues, candidates_by_tissue, middle_predictions, peft_predictions_by_tissue)
    class_names = class_names_from_config(
        Path(load_yaml(_resolve(tissues[0].config))["data"]["dataset_path"]).expanduser().resolve()
    )
    save_grid(
        selected,
        output_stem,
        class_names,
        args.bg_alpha,
        args.mask_alpha,
        args.boundary_width,
        args.dpi,
        middle_row_label,
    )
    if args.grid == "main-3x6":
        top_three = sorted(selected, key=lambda item: (item.improvement, item.peft_type_accuracy, item.peft_matched), reverse=True)[:3]
        save_grid(
            top_three,
            REPO_ROOT / "figures/qualitative_grid_3x3_main",
            class_names,
            args.bg_alpha,
            args.mask_alpha,
            args.boundary_width,
            args.dpi,
            middle_row_label,
        )
    write_metadata(selected, metadata_path, middle_row_label)

    print("Selected patches:")
    for item in selected:
        print(
            f"{item.candidate.tissue.name}: {Path(item.candidate.patch.patch_id).name} "
            f"GT={item.candidate.gt_nuclei} middle_acc={item.frozen_type_accuracy:.3f} "
            f"peft_acc={item.peft_type_accuracy:.3f} improvement={item.improvement:.3f}"
        )


if __name__ == "__main__":
    main()
