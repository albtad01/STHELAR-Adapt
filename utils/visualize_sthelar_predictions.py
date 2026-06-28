#!/usr/bin/env python3
"""Create publication-quality qualitative panels for CellViT on STHELAR patches.

The utility reads patches directly from the CellViT-ready ``images.zip`` and
``labels.zip`` archives.  This avoids constructing/caching the complete dataset.
Predictions are made only for the selected patches and can be cached as compact
NPZ files for later rendering.

Examples
--------
python utils/visualize_sthelar_predictions.py \
    --config RUN/config.yaml --checkpoint RUN/checkpoints/model_best.pth \
    --dataset-split test --num-samples 4 --min-qc 0.60 \
    --output-dir reports/qualitative/liver_5class

python utils/visualize_sthelar_predictions.py \
    --config PEFT/config.yaml --checkpoint PEFT/checkpoints/model_best.pth \
    --baseline-config FROZEN/config.yaml \
    --baseline-checkpoint FROZEN/checkpoints/model_best.pth \
    --dataset-split test --patch-ids PATCH_A PATCH_B \
    --output-dir reports/qualitative/frozen_vs_peft
"""

from __future__ import annotations

import argparse
import csv
import gc
import io
import json
import sys
import zipfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

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


# Stable, color-blind-conscious colors shared by 5- and 9-class figures.
# STHELAR calls these modes 5/9-class while model output includes Background,
# hence six and ten entries respectively.
PALETTE_5 = {
    "Background": "#F3F3F3",
    "Immune": "#0072B2",
    "Stromal": "#E69F00",
    "Epithelial": "#D55E00",
    "Melanocyte": "#CC79A7",
    "Other": "#7F7F7F",
}
PALETTE_9 = {
    "Background": "#F3F3F3",
    "Epithelial": "#D55E00",
    "Blood_vessel": "#56B4E9",
    "Fibroblast_Myofibroblast": "#E69F00",
    "Myeloid": "#009E73",
    "B_Plasma": "#0072B2",
    "T_NK": "#6A3D9A",
    "Melanocyte": "#CC79A7",
    "Specialized": "#F0E442",
    "Other": "#7F7F7F",
}


@dataclass
class PatchData:
    row: pd.Series
    image: np.ndarray
    gt_instance: np.ndarray
    gt_type: np.ndarray

    @property
    def patch_id(self) -> str:
        return str(self.row["packed_file_name"])


@dataclass
class Prediction:
    instance: np.ndarray
    type_map: np.ndarray


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Render STHELAR H&E/GT/CellViT qualitative panels."
    )
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--baseline-config", type=Path, default=None)
    parser.add_argument("--baseline-checkpoint", type=Path, default=None)
    parser.add_argument(
        "--dataset-split", choices=("train", "valid", "test"), default="test"
    )
    parser.add_argument("--num-samples", type=int, default=4)
    parser.add_argument(
        "--patch-ids",
        nargs="+",
        default=None,
        help="Packed names, original names, or filename stems; comma-separated values work too.",
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--predictions-dir",
        type=Path,
        default=None,
        help="Read/write compact per-patch prediction NPZ files (default: OUTPUT/predictions).",
    )
    parser.add_argument(
        "--baseline-predictions-dir",
        type=Path,
        default=None,
        help="Read/write baseline NPZ files (default: OUTPUT/baseline_predictions).",
    )
    parser.add_argument("--qc-metadata", type=Path, default=None)
    parser.add_argument("--qc-metric", default="Jaccard")
    parser.add_argument("--min-qc", type=float, default=None)
    parser.add_argument(
        "--selection",
        choices=("best", "diverse", "random"),
        default="diverse",
        help="Selection after split/QC filtering. 'diverse' samples across the QC range.",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default=None, help="cuda:0, cpu, or mps")
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--alpha", type=float, default=0.52)
    parser.add_argument("--boundary-width", type=int, default=2)
    parser.add_argument("--error-overlay", action="store_true")
    parser.add_argument("--dpi", type=int, default=300)
    parser.add_argument(
        "--force-inference", action="store_true", help="Ignore cached prediction NPZ files."
    )
    return parser.parse_args()


def load_yaml(path: Path) -> dict[str, Any]:
    path = path.expanduser().resolve()
    if not path.is_file():
        raise FileNotFoundError(path)
    with path.open() as handle:
        value = yaml.safe_load(handle)
    if not isinstance(value, dict):
        raise ValueError(f"Expected a YAML mapping in {path}")
    return value


def class_names_from_config(dataset_root: Path) -> dict[int, str]:
    config = load_yaml(dataset_root / "dataset_config.yaml")
    raw = config.get("nuclei_types")
    if not isinstance(raw, dict):
        raise ValueError("dataset_config.yaml has no nuclei_types mapping")
    if all(isinstance(value, int) for value in raw.values()):
        names = {int(value): str(key) for key, value in raw.items()}
    else:
        names = {int(key): str(value) for key, value in raw.items()}
    expected = set(range(max(names) + 1))
    if set(names) != expected:
        raise ValueError(f"Nuclei type IDs must be contiguous; found {sorted(names)}")
    return dict(sorted(names.items()))


def color_table(class_names: dict[int, str]) -> np.ndarray:
    if len(class_names) == 6:
        palette = PALETTE_5
    elif len(class_names) == 10:
        palette = PALETTE_9
    else:
        raise ValueError(
            f"Expected STHELAR 5-class (6 outputs) or 9-class (10 outputs), got {len(class_names)}"
        )
    missing = [name for name in class_names.values() if name not in palette]
    if missing:
        raise ValueError(f"No publication palette color for classes: {missing}")
    return np.asarray(
        [matplotlib.colors.to_rgb(palette[class_names[idx]]) for idx in class_names],
        dtype=np.float32,
    )


def _flatten_patch_ids(values: Iterable[str] | None) -> list[str]:
    if not values:
        return []
    return [part.strip() for value in values for part in value.split(",") if part.strip()]


def _identifier_set(row: pd.Series) -> set[str]:
    identifiers: set[str] = set()
    for column in ("packed_file_name", "packed_label_name", "file_name", "patch_id"):
        if column in row and pd.notna(row[column]):
            value = str(row[column])
            identifiers.update((value, Path(value).name, Path(value).stem))
    return identifiers


def _load_image_metrics(config_path: Path, checkpoint_path: Path) -> dict[str, Any]:
    candidates = [
        config_path.resolve().parent / "inference_results.json",
        checkpoint_path.resolve().parent.parent / "inference_results.json",
    ]
    for path in candidates:
        if path.is_file():
            with path.open() as handle:
                return json.load(handle).get("image_metrics", {})
    return {}


def select_rows(
    patch_info: pd.DataFrame,
    split: str,
    patch_ids: list[str],
    num_samples: int,
    qc_metric: str,
    min_qc: float | None,
    selection: str,
    seed: int,
) -> pd.DataFrame:
    if "split" not in patch_info:
        raise ValueError("Patch metadata has no 'split' column")
    frame = patch_info[patch_info["split"].astype(str).str.lower() == split].copy()
    if patch_ids:
        selected = []
        for patch_id in patch_ids:
            hits = frame.apply(lambda row: patch_id in _identifier_set(row), axis=1)
            if not hits.any():
                raise ValueError(f"Patch ID not found in split={split}: {patch_id}")
            selected.append(frame.loc[hits].iloc[0])
        return pd.DataFrame(selected).reset_index(drop=True)

    metric_column = next(
        (column for column in frame.columns if column.lower() == qc_metric.lower()), None
    )
    if min_qc is not None:
        if metric_column is None:
            raise ValueError(
                f"--min-qc was given but QC metric '{qc_metric}' is absent from metadata"
            )
        frame = frame[pd.to_numeric(frame[metric_column], errors="coerce") >= min_qc]
    if frame.empty:
        raise ValueError("No patches remain after split/QC filtering")

    n = min(num_samples, len(frame))
    if selection == "random" or metric_column is None:
        return frame.sample(n=n, random_state=seed).reset_index(drop=True)
    ranked = frame.assign(
        _qc=pd.to_numeric(frame[metric_column], errors="coerce").fillna(-np.inf)
    ).sort_values("_qc", ascending=False)
    if selection == "best" or n == 1:
        return ranked.head(n).drop(columns="_qc").reset_index(drop=True)
    # Spread samples over the retained quality distribution instead of showing
    # four near-duplicates from the very top.
    indices = np.linspace(0, len(ranked) - 1, n, dtype=int)
    return ranked.iloc[indices].drop(columns="_qc").reset_index(drop=True)


def add_tissue_metadata(patch_info: pd.DataFrame, dataset_root: Path) -> pd.DataFrame:
    """Attach PanNuke/STHELAR ``types.csv`` tissue labels when needed."""
    if "tissue" in patch_info.columns or "type" in patch_info.columns:
        return patch_info
    types_path = dataset_root / "types.csv"
    if not types_path.is_file():
        return patch_info
    types = pd.read_csv(types_path)
    if not {"img", "type"}.issubset(types.columns):
        return patch_info
    join_column = "packed_file_name" if "packed_file_name" in patch_info.columns else "file_name"
    if join_column not in patch_info.columns:
        return patch_info
    lookup = types[["img", "type"]].drop_duplicates("img").rename(
        columns={"img": join_column, "type": "tissue"}
    )
    return patch_info.merge(lookup, how="left", on=join_column)


def resolve_zip_member(
    archive: zipfile.ZipFile,
    requested: str,
    names: list[str] | None = None,
    by_basename: dict[str, list[str]] | None = None,
) -> str:
    if names is None:
        names = [name for name in archive.namelist() if not Path(name).name.startswith("._")]
    if requested in names:
        return requested
    if by_basename is None:
        by_basename = {}
        for name in names:
            by_basename.setdefault(Path(name).name, []).append(name)
    matches = by_basename.get(Path(requested).name, [])
    if len(matches) == 1:
        return matches[0]
    raise FileNotFoundError(f"Could not uniquely resolve {requested} inside {archive.filename}")


def _dense_npz(data: Any, key: str) -> np.ndarray:
    if key in data.files:
        return np.asarray(data[key])
    keys = [f"{key}_{suffix}" for suffix in ("data", "indices", "indptr", "shape")]
    if all(item in data.files for item in keys):
        # Decode CSR directly. Importing scipy for a handful of 256x256 patches
        # adds considerable startup time on compute nodes and is unnecessary.
        values = data[keys[0]]
        columns = data[keys[1]].astype(np.intp, copy=False)
        indptr = data[keys[2]].astype(np.intp, copy=False)
        shape = tuple(int(value) for value in data[keys[3]])
        dense = np.zeros(shape, dtype=values.dtype)
        row_ids = np.repeat(np.arange(shape[0]), np.diff(indptr))
        dense[row_ids, columns] = values
        return dense
    raise KeyError(f"No dense or CSR '{key}' in label; keys={data.files}")


def load_patches(dataset_root: Path, rows: pd.DataFrame) -> list[PatchData]:
    patches: list[PatchData] = []
    with zipfile.ZipFile(dataset_root / "images.zip") as images, zipfile.ZipFile(
        dataset_root / "labels.zip"
    ) as labels:
        image_names = [name for name in images.namelist() if not Path(name).name.startswith("._")]
        label_names = [name for name in labels.namelist() if not Path(name).name.startswith("._")]
        image_basenames: dict[str, list[str]] = {}
        label_basenames: dict[str, list[str]] = {}
        for name in image_names:
            image_basenames.setdefault(Path(name).name, []).append(name)
        for name in label_names:
            label_basenames.setdefault(Path(name).name, []).append(name)
        for _, row in rows.iterrows():
            image_name = str(row["packed_file_name"])
            label_name = str(
                row.get("packed_label_name", Path(image_name).with_suffix(".npz").name)
            )
            image_member = resolve_zip_member(
                images, image_name, image_names, image_basenames
            )
            label_member = resolve_zip_member(
                labels, label_name, label_names, label_basenames
            )
            with images.open(image_member) as handle:
                image = np.asarray(Image.open(io.BytesIO(handle.read())).convert("RGB"))
            with labels.open(label_member) as handle:
                with np.load(io.BytesIO(handle.read()), allow_pickle=False) as data:
                    instance = _dense_npz(data, "inst_map").astype(np.int32)
                    type_map = _dense_npz(data, "type_map").astype(np.int16)
            patches.append(PatchData(row, image, instance, type_map))
    return patches


def _infer_config_for_checkpoint(checkpoint: Path) -> Path | None:
    for candidate in (checkpoint.parent.parent / "config.yaml", checkpoint.parent / "config.yaml"):
        if candidate.is_file():
            return candidate
    return None


def _insert_adapters(model: Any, config: dict[str, Any]) -> None:
    from models.adapters.utils import (
        insert_adaptformer,
        insert_bottleneck,
        insert_decoder_conv_adapters,
        insert_lora2,
        insert_plora,
        insert_vera,
    )

    adapters = config.get("adapters", {})
    kind = adapters.get("adapter_type")

    def decoder_adapters() -> None:
        if str(adapters.get("decoder_train_scope", "all")).lower() == "conv_adapters":
            insert_decoder_conv_adapters(
                model,
                reduction=adapters.get("decoder_adapter_reduction", 16),
                activation=adapters.get("decoder_adapter_activation", "GELU"),
                alpha_init=adapters.get("decoder_adapter_alpha_init", 1.0),
                train_alpha=adapters.get("decoder_adapter_train_alpha", True),
            )

    def lora() -> None:
        options = adapters.get("lora", {})
        insert_lora2(
            model,
            rank=options.get("rank", 8),
            alpha=options.get("alpha", 8),
            targets=options.get("targets", ["q", "v"]),
            dropout=options.get("dropout", 0.0),
        )

    def adaptformer() -> None:
        options = adapters.get("adaptformer", {})
        insert_adaptformer(
            model, options.get("activation", "GELU"), options.get("reduction", 16)
        )

    def vera() -> None:
        options = adapters.get("vera", {})
        insert_vera(
            model,
            rank=options.get("rank", adapters.get("vera_rank", 8)),
            alpha=options.get("alpha", adapters.get("vera_alpha", 8)),
            targets=options.get("targets", adapters.get("vera_targets", ["q", "v"])),
            dropout=options.get("dropout", adapters.get("vera_dropout", 0.0)),
            shared_matrices=options.get(
                "shared_matrices", adapters.get("vera_shared_matrices", True)
            ),
            train_alpha=options.get("train_alpha", adapters.get("vera_train_alpha", True)),
            seed=options.get("seed", adapters.get("vera_seed", config.get("random_seed", 42))),
        )

    if kind in ("lora", "lora_ntonly"):
        lora()
        decoder_adapters()
    elif kind == "adaptformer":
        adaptformer()
        decoder_adapters()
    elif kind == "lora_adaptformer" or kind == "lora_adaptformer_ntonly":
        lora()
        adaptformer()
        decoder_adapters()
    elif kind == "vera":
        vera()
        decoder_adapters()
    elif kind == "vera_adaptformer":
        vera()
        adaptformer()
        decoder_adapters()
    elif kind == "plora":
        options = adapters.get("plora", {})
        insert_plora(model, options.get("rank", 8), options.get("alpha", 8))
    elif kind == "bottleneck":
        options = adapters.get("bottleneck", {})
        insert_bottleneck(
            model, options.get("activation", "GELU"), options.get("reduction", 16)
        )
    elif kind in (None, "ntonly"):
        return
    else:
        raise ValueError(f"Unsupported adapter_type: {kind}")


def _build_model(config: dict[str, Any], architecture: str) -> Any:
    from models.segmentation.cell_segmentation.cellvit import CellViT, CellViT256, CellViTSAM
    from models.segmentation.cell_segmentation.cellvit_shared import (
        CellViT256Shared,
        CellViTSAMShared,
        CellViTShared,
    )

    data = config["data"]
    model_conf = config["model"]
    classes = {
        "CellViT": CellViT,
        "CellViTShared": CellViTShared,
        "CellViT256": CellViT256,
        "CellViT256Shared": CellViT256Shared,
        "CellViTSAM": CellViTSAM,
        "CellViTSAMShared": CellViTSAMShared,
    }
    if architecture not in classes:
        raise ValueError(f"Unsupported checkpoint architecture: {architecture}")
    cls = classes[architecture]
    common = dict(
        num_nuclei_classes=data["num_nuclei_classes"],
        num_tissue_classes=data["num_tissue_classes"],
        regression_loss=model_conf.get("regression_loss", False),
    )
    if architecture in ("CellViT", "CellViTShared"):
        return cls(
            **common,
            embed_dim=model_conf["embed_dim"],
            input_channels=model_conf.get("input_channels", 3),
            depth=model_conf["depth"],
            num_heads=model_conf["num_heads"],
            extract_layers=model_conf["extract_layers"],
        )
    if architecture in ("CellViT256", "CellViT256Shared"):
        return cls(model256_path=None, **common)
    return cls(model_path=None, vit_structure=model_conf["backbone"], **common)


def _prediction_cache_path(directory: Path, patch_id: str) -> Path:
    return directory / f"{Path(patch_id).stem}.npz"


def _read_cached_predictions(
    directory: Path, patches: list[PatchData]
) -> dict[str, Prediction] | None:
    output: dict[str, Prediction] = {}
    for patch in patches:
        path = _prediction_cache_path(directory, patch.patch_id)
        if not path.is_file():
            return None
        with np.load(path, allow_pickle=False) as data:
            output[patch.patch_id] = Prediction(
                data["instance_map"].astype(np.int32), data["type_map"].astype(np.int16)
            )
    return output


def predict_selected(
    config_path: Path,
    checkpoint_path: Path,
    patches: list[PatchData],
    cache_dir: Path,
    device_name: str | None,
    batch_size: int,
    force: bool,
) -> dict[str, Prediction]:
    if not force:
        cached = _read_cached_predictions(cache_dir, patches)
        if cached is not None:
            print(f"Using {len(cached)} cached predictions from {cache_dir}")
            return cached

    import torch
    import torch.nn.functional as functional

    config = load_yaml(config_path)
    if device_name is None:
        device_name = "cuda:0" if torch.cuda.is_available() else "cpu"
    device = torch.device(device_name)
    checkpoint_path = checkpoint_path.expanduser().resolve()
    if not checkpoint_path.is_file():
        raise FileNotFoundError(checkpoint_path)
    print(f"Loading checkpoint on CPU: {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    model = _build_model(config, checkpoint["arch"])
    _insert_adapters(model, config)
    incompat = model.load_state_dict(checkpoint["model_state_dict"], strict=True)
    if incompat.missing_keys or incompat.unexpected_keys:
        raise RuntimeError(f"Checkpoint incompatibility: {incompat}")
    del checkpoint
    model.to(device).eval()

    normalize = config.get("transformations", {}).get("normalize", {})
    mean = torch.tensor(normalize.get("mean", [0.5] * 3), device=device).view(1, 3, 1, 1)
    std = torch.tensor(normalize.get("std", [0.5] * 3), device=device).view(1, 3, 1, 1)
    magnification = int(config.get("data", {}).get("magnification", 40))
    cache_dir.mkdir(parents=True, exist_ok=True)
    result: dict[str, Prediction] = {}

    with torch.inference_mode():
        for start in range(0, len(patches), max(1, batch_size)):
            batch = patches[start : start + max(1, batch_size)]
            images = np.stack([patch.image for patch in batch])
            tensor = torch.from_numpy(images).permute(0, 3, 1, 2).float().to(device) / 255.0
            tensor = (tensor - mean) / std
            use_amp = bool(config.get("training", {}).get("mixed_precision", False))
            amp_enabled = use_amp and device.type == "cuda"
            with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=amp_enabled):
                raw = model(tensor, retrieve_tokens=False)
            raw["nuclei_binary_map"] = functional.softmax(raw["nuclei_binary_map"], dim=1)
            raw["nuclei_type_map"] = functional.softmax(raw["nuclei_type_map"], dim=1)
            instance_maps, instance_types = model.calculate_instance_map(
                raw, magnification=magnification
            )
            instance_class_maps = model.generate_instance_nuclei_map(
                instance_maps, instance_types
            )
            type_maps = torch.argmax(instance_class_maps, dim=1).cpu().numpy()
            instances = instance_maps.cpu().numpy()
            for index, patch in enumerate(batch):
                prediction = Prediction(
                    instances[index].astype(np.int32), type_maps[index].astype(np.int16)
                )
                result[patch.patch_id] = prediction
                np.savez_compressed(
                    _prediction_cache_path(cache_dir, patch.patch_id),
                    instance_map=prediction.instance,
                    type_map=prediction.type_map,
                )
            del tensor, raw, instance_maps, instance_class_maps

    del model
    gc.collect()
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return result


def instance_boundary(instance_map: np.ndarray, width: int = 1) -> np.ndarray:
    boundary = np.zeros(instance_map.shape, dtype=bool)
    boundary[1:] |= instance_map[1:] != instance_map[:-1]
    boundary[:-1] |= instance_map[:-1] != instance_map[1:]
    boundary[:, 1:] |= instance_map[:, 1:] != instance_map[:, :-1]
    boundary[:, :-1] |= instance_map[:, :-1] != instance_map[:, 1:]
    boundary &= instance_map != 0
    for _ in range(max(0, width - 1)):
        grown = boundary.copy()
        grown[1:] |= boundary[:-1]
        grown[:-1] |= boundary[1:]
        grown[:, 1:] |= boundary[:, :-1]
        grown[:, :-1] |= boundary[:, 1:]
        boundary = grown & (instance_map != 0)
    return boundary


def make_overlay(
    image: np.ndarray,
    instance_map: np.ndarray,
    type_map: np.ndarray,
    colors: np.ndarray,
    alpha: float,
    boundary_width: int,
) -> np.ndarray:
    rgb = image.astype(np.float32) / 255.0
    foreground = instance_map > 0
    safe_types = np.clip(type_map, 0, len(colors) - 1)
    rgb[foreground] = (1 - alpha) * rgb[foreground] + alpha * colors[safe_types[foreground]]
    boundary = instance_boundary(instance_map, boundary_width)
    rgb[boundary] = 0.12 * rgb[boundary] + 0.88 * colors[safe_types[boundary]]
    return np.clip(rgb, 0, 1)


def make_error_overlay(
    image: np.ndarray, gt: PatchData, pred: Prediction, boundary_width: int
) -> np.ndarray:
    rgb = image.astype(np.float32) / 255.0
    gt_fg, pred_fg = gt.gt_instance > 0, pred.instance > 0
    missed = gt_fg & ~pred_fg
    extra = pred_fg & ~gt_fg
    wrong_type = gt_fg & pred_fg & (gt.gt_type != pred.type_map)
    for mask, color in (
        (missed, np.array([0.85, 0.10, 0.10])),
        (extra, np.array([0.10, 0.45, 0.95])),
        (wrong_type, np.array([0.95, 0.75, 0.05])),
    ):
        rgb[mask] = 0.35 * rgb[mask] + 0.65 * color
    rgb[instance_boundary(gt.gt_instance, boundary_width)] *= 0.25
    return np.clip(rgb, 0, 1)


def plot_panel(
    patches: list[PatchData],
    peft: dict[str, Prediction],
    baseline: dict[str, Prediction] | None,
    class_names: dict[int, str],
    output_dir: Path,
    alpha: float,
    boundary_width: int,
    error_overlay: bool,
    dpi: int,
) -> None:
    colors = color_table(class_names)
    columns = ["H&E", "Ground truth"]
    if baseline is not None:
        columns.append("Frozen CellViT")
    columns.append("PEFT prediction" if baseline is not None else "Prediction")
    if error_overlay:
        columns.append("Errors")
    nrows, ncols = len(patches), len(columns)
    fig, axes = plt.subplots(
        nrows,
        ncols,
        figsize=(3.35 * ncols, 3.1 * nrows),
        squeeze=False,
        constrained_layout=False,
    )
    for row_index, patch in enumerate(patches):
        prediction = peft[patch.patch_id]
        panels = [
            patch.image,
            make_overlay(
                patch.image, patch.gt_instance, patch.gt_type, colors, alpha, boundary_width
            ),
        ]
        if baseline is not None:
            base = baseline[patch.patch_id]
            panels.append(
                make_overlay(
                    patch.image, base.instance, base.type_map, colors, alpha, boundary_width
                )
            )
        panels.append(
            make_overlay(
                patch.image,
                prediction.instance,
                prediction.type_map,
                colors,
                alpha,
                boundary_width,
            )
        )
        if error_overlay:
            panels.append(make_error_overlay(patch.image, patch, prediction, boundary_width))
        for column_index, panel in enumerate(panels):
            axes[row_index, column_index].imshow(panel)
            axes[row_index, column_index].axis("off")
            if row_index == 0:
                axes[row_index, column_index].set_title(columns[column_index], fontsize=11)
        label = Path(patch.patch_id).stem
        axes[row_index, 0].text(
            0,
            -0.035,
            label,
            transform=axes[row_index, 0].transAxes,
            fontsize=7,
            va="top",
        )

    handles = [
        mpatches.Patch(facecolor=colors[index], edgecolor="black", label=name)
        for index, name in class_names.items()
        if index != 0
    ]
    if error_overlay:
        handles.extend(
            mpatches.Patch(facecolor=color, label=label)
            for label, color in (
                ("Missed nucleus", "#D91A1A"),
                ("False positive", "#1A73E8"),
                ("Type mismatch", "#F2BF0D"),
            )
        )
    legend_columns = min(5, len(handles))
    fig.legend(
        handles=handles,
        loc="lower center",
        ncol=legend_columns,
        frameon=False,
        fontsize=8,
        bbox_to_anchor=(0.5, 0.005),
    )
    bottom = 0.055 + 0.025 * max(0, (len(handles) - 1) // legend_columns)
    fig.subplots_adjust(left=0.015, right=0.995, top=0.96, bottom=bottom, wspace=0.025, hspace=0.12)
    output_dir.mkdir(parents=True, exist_ok=True)
    for suffix in ("png", "pdf"):
        path = output_dir / f"qualitative_panel.{suffix}"
        fig.savefig(path, dpi=dpi, bbox_inches="tight", facecolor="white")
        print(f"Saved {path}")
    plt.close(fig)


def save_selection_csv(
    patches: list[PatchData],
    image_metrics: dict[str, Any],
    output_path: Path,
) -> None:
    records = []
    for patch in patches:
        row = patch.row
        record = {
            "patch_id": patch.patch_id,
            "slide_id": row.get("slide_id", ""),
            "tissue": row.get("tissue", row.get("type", "")),
            "split": row.get("split", ""),
        }
        for metric in ("Dice", "Jaccard", "bPQ", "mPQ"):
            if metric in row and pd.notna(row[metric]):
                record[f"qc_{metric}"] = row[metric]
        metrics = image_metrics.get(patch.patch_id, image_metrics.get(Path(patch.patch_id).name, {}))
        for metric in ("Dice", "Jaccard", "bPQ", "mPQ"):
            if metric in metrics:
                record[f"model_{metric}"] = metrics[metric]
        if pd.isna(record["tissue"]):
            record["tissue"] = ""
        records.append(record)
    keys = list(dict.fromkeys(key for record in records for key in record))
    with output_path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=keys)
        writer.writeheader()
        writer.writerows(records)
    print(f"Saved {output_path}")


def main() -> None:
    args = parse_args()
    if args.num_samples < 1 or args.batch_size < 1:
        raise ValueError("--num-samples and --batch-size must be positive")
    config_path = args.config.expanduser().resolve()
    checkpoint_path = args.checkpoint.expanduser().resolve()
    config = load_yaml(config_path)
    dataset_root = Path(config["data"]["dataset_path"]).expanduser().resolve()
    for required in ("images.zip", "labels.zip", "patch_info_with_split.csv", "dataset_config.yaml"):
        if not (dataset_root / required).is_file():
            raise FileNotFoundError(dataset_root / required)
    class_names = class_names_from_config(dataset_root)
    configured_classes = int(config["data"]["num_nuclei_classes"])
    if configured_classes != len(class_names):
        raise ValueError(
            f"Config expects {configured_classes} outputs, dataset defines {len(class_names)}"
        )

    metadata_path = args.qc_metadata.expanduser().resolve() if args.qc_metadata else dataset_root / "patch_info_with_split.csv"
    patch_info = add_tissue_metadata(pd.read_csv(metadata_path), dataset_root)
    rows = select_rows(
        patch_info,
        args.dataset_split,
        _flatten_patch_ids(args.patch_ids),
        args.num_samples,
        args.qc_metric,
        args.min_qc,
        args.selection,
        args.seed,
    )
    patches = load_patches(dataset_root, rows)
    output_dir = args.output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    prediction_dir = (
        args.predictions_dir.expanduser().resolve()
        if args.predictions_dir
        else output_dir / "predictions"
    )
    peft = predict_selected(
        config_path,
        checkpoint_path,
        patches,
        prediction_dir,
        args.device,
        args.batch_size,
        args.force_inference,
    )

    baseline = None
    if args.baseline_checkpoint:
        baseline_checkpoint = args.baseline_checkpoint.expanduser().resolve()
        baseline_config = args.baseline_config
        if baseline_config is None:
            baseline_config = _infer_config_for_checkpoint(baseline_checkpoint)
        if baseline_config is None:
            raise ValueError(
                "Could not infer baseline config; pass --baseline-config explicitly"
            )
        baseline_dir = (
            args.baseline_predictions_dir.expanduser().resolve()
            if args.baseline_predictions_dir
            else output_dir / "baseline_predictions"
        )
        # Models are intentionally loaded sequentially: the PEFT model has
        # already been deleted before this (large SAM-H checkpoints are ~GBs).
        baseline = predict_selected(
            Path(baseline_config),
            baseline_checkpoint,
            patches,
            baseline_dir,
            args.device,
            args.batch_size,
            args.force_inference,
        )
    elif args.baseline_config is not None:
        raise ValueError("--baseline-config requires --baseline-checkpoint")

    plot_panel(
        patches,
        peft,
        baseline,
        class_names,
        output_dir,
        args.alpha,
        args.boundary_width,
        args.error_overlay,
        args.dpi,
    )
    metrics = _load_image_metrics(config_path, checkpoint_path)
    save_selection_csv(patches, metrics, output_dir / "selected_samples.csv")


if __name__ == "__main__":
    main()
