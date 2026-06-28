#!/usr/bin/env python3
"""Estimate grouped diagonal Fisher information and checkpoint drift for CellViT.

This utility is deliberately read-only with respect to model checkpoints: it never
constructs an optimizer, calls ``step()``, or writes model weights.
"""

from __future__ import annotations

import argparse
import csv
import gc
import json
import logging
import re
import sys
from pathlib import Path
from typing import Any, Iterable

import albumentations as A
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from base_ml.base_loss import retrieve_loss_fn
from cell_segmentation.datasets.dataset_coordinator import select_dataset
from utils.adapter_checkpoint import (
    build_model,
    checkpoint_state_dict,
    instantiate_model,
    insert_adapters_from_config,
    load_adapter_state,
    load_yaml,
    seed_everything,
)


LOGGER = logging.getLogger("fisher_drift")
COARSE_GROUPS = (
    "vera",
    "lora",
    "adaptformer",
    "decoder_conv_adapter",
    "final_heads",
    "classifier",
    "decoder_original_trainable",
    "other_trainable",
    "frozen_encoder_base",
)
FULLFT_BLOCK_GROUPS = (
    "encoder_early_blocks",
    "encoder_mid_blocks",
    "encoder_late_blocks",
    "encoder_attention_qkv",
    "encoder_attention_proj",
    "encoder_mlp",
    "decoder_np",
    "decoder_hv",
    "decoder_nt",
    "decoder_conv_like",
    "final_heads",
    "classifier",
    "other_trainable",
)
GROUPS = COARSE_GROUPS


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument(
        "--dataset-split", choices=("train", "valid"), default="train"
    )
    parser.add_argument("--num-batches", type=int, default=20)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--output-dir", type=Path, default=Path("reports/fisher_drift"))
    parser.add_argument(
        "--grouping",
        choices=("coarse", "fullft_blocks"),
        default="coarse",
        help="Parameter grouping scheme. 'coarse' preserves the original PEFT-oriented groups.",
    )
    parser.add_argument(
        "--print-groups",
        action="store_true",
        help=(
            "Dry inspection: build model topology, validate the checkpoint path, "
            "and print group sizes and example names without loading weights."
        ),
    )
    parser.add_argument(
        "--compare-checkpoints",
        nargs="+",
        type=Path,
        help="Additional checkpoints; Fisher is computed on the same ordered batches.",
    )
    parser.add_argument(
        "--include-frozen",
        action="store_true",
        help="Also enable gradients for the frozen, non-adapter encoder base (expensive).",
    )
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument(
        "--max-drift-samples",
        type=int,
        default=200_000,
        help="Maximum evenly sampled Fisher entries retained per group/checkpoint.",
    )
    return parser.parse_args()


def groups_for(grouping: str) -> tuple[str, ...]:
    if grouping == "coarse":
        return COARSE_GROUPS
    if grouping == "fullft_blocks":
        return FULLFT_BLOCK_GROUPS
    raise ValueError(f"Unknown grouping mode: {grouping}")


def resolve_path(path: Path, relative_to: Path = REPO_ROOT) -> Path:
    path = path.expanduser()
    if not path.is_absolute():
        path = relative_to / path
    return path.resolve()


def torch_load(path: Path) -> Any:
    try:
        return torch.load(str(path), map_location="cpu", weights_only=False)
    except TypeError:  # PyTorch < 2.0
        return torch.load(str(path), map_location="cpu")


def parameter_group_coarse(name: str, originally_trainable: bool) -> str | None:
    """Map a model parameter name to one mutually exclusive diagnostic group."""
    lower = name.lower()
    if "vera_" in lower or "vera_adapter" in lower:
        return "vera"
    if "decoder_adapter" in lower:
        return "decoder_conv_adapter"
    if "adaptformer" in lower or (
        ".mlp." in lower
        and any(token in lower for token in ("adapter_downsample", "adapter_upsample", "adapter_alpha"))
    ):
        return "adaptformer"
    if ".attn.qkv." in lower and "adapter_" in lower:
        return "lora"
    if lower.startswith("classifier_head") or ".classifier_head" in lower:
        return "classifier"
    if "decoder0_header" in lower or "final_head" in lower:
        return "final_heads"

    is_encoder = lower.startswith("encoder.") or ".encoder." in lower
    if is_encoder and not originally_trainable:
        return "frozen_encoder_base"
    if not originally_trainable:
        return None
    if "decoder" in lower:
        return "decoder_original_trainable"
    return "other_trainable"


def encoder_block_index(name: str) -> int | None:
    match = re.search(r"(?:^|\.)blocks\.(\d+)(?:\.|$)", name)
    return int(match.group(1)) if match else None


def encoder_block_stage(index: int, max_index: int | None) -> str:
    if max_index is None or max_index < 2:
        return "encoder_mid_blocks"
    depth = max_index + 1
    early_cut = depth / 3.0
    late_cut = 2.0 * depth / 3.0
    if index < early_cut:
        return "encoder_early_blocks"
    if index < late_cut:
        return "encoder_mid_blocks"
    return "encoder_late_blocks"


def is_encoder_parameter(lower: str) -> bool:
    return lower.startswith("encoder.") or ".encoder." in lower


def is_final_head_parameter(lower: str) -> bool:
    return "decoder0_header" in lower or "final_head" in lower


def parameter_group_fullft_blocks(
    name: str,
    originally_trainable: bool,
    max_encoder_block_index: int | None,
) -> str | None:
    """Full fine-tuning grouping with mutually exclusive CellViT/SAM regions.

    Attention and MLP groups are prioritized over early/mid/late block buckets.
    The block buckets therefore collect the remaining per-block parameters such
    as norms, relative-position terms, and adapter-independent block scalars.
    """
    lower = name.lower()

    if lower.startswith("classifier_head") or ".classifier_head" in lower:
        return "classifier"
    if is_final_head_parameter(lower):
        return "final_heads"

    if "nuclei_binary_map_decoder" in lower or "nuclei_binary_maps_decoder" in lower:
        return "decoder_np"
    if "hv_map_decoder" in lower or "hv_maps_decoder" in lower:
        return "decoder_hv"
    if "nuclei_type_maps_decoder" in lower or "nuclei_type_map_decoder" in lower:
        return "decoder_nt"
    if (
        lower.startswith(("decoder0.", "decoder1.", "decoder2.", "decoder3."))
        or lower.startswith("decoder.")
        or ".decoder0." in lower
        or ".decoder1." in lower
        or ".decoder2." in lower
        or ".decoder3." in lower
        or ".decoder." in lower
        or "decoder_adapter" in lower
    ):
        return "decoder_conv_like"

    if is_encoder_parameter(lower):
        if ".attn.qkv." in lower or ".attn.qkv_" in lower or "attention.qkv" in lower:
            return "encoder_attention_qkv"
        if ".attn.proj." in lower or ".attn.proj_" in lower or "attention.proj" in lower:
            return "encoder_attention_proj"
        if ".mlp." in lower or ".mlp_" in lower:
            return "encoder_mlp"
        block_index = encoder_block_index(lower)
        if block_index is not None:
            return encoder_block_stage(block_index, max_encoder_block_index)
        if originally_trainable:
            return "other_trainable"
        return None

    if not originally_trainable:
        return None
    return "other_trainable"


def parameter_group(
    name: str,
    originally_trainable: bool,
    grouping: str,
    max_encoder_block_index: int | None,
) -> str | None:
    if grouping == "coarse":
        return parameter_group_coarse(name, originally_trainable)
    if grouping == "fullft_blocks":
        return parameter_group_fullft_blocks(name, originally_trainable, max_encoder_block_index)
    raise ValueError(f"Unknown grouping mode: {grouping}")


def max_encoder_block_index(model: nn.Module) -> int | None:
    indexes = [
        index
        for name, _ in model.named_parameters()
        if is_encoder_parameter(name.lower())
        for index in [encoder_block_index(name.lower())]
        if index is not None
    ]
    return max(indexes) if indexes else None


def group_parameters(
    model: nn.Module, include_frozen: bool, grouping: str
) -> tuple[dict[str, list[tuple[str, nn.Parameter]]], dict[str, str], dict[str, bool]]:
    groups = groups_for(grouping)
    grouped: dict[str, list[tuple[str, nn.Parameter]]] = {group: [] for group in groups}
    statuses: dict[str, set[bool]] = {group: set() for group in groups}
    original_flags: dict[str, bool] = {}
    max_block = max_encoder_block_index(model) if grouping == "fullft_blocks" else None
    for name, parameter in model.named_parameters():
        original_flags[name] = parameter.requires_grad
        group = parameter_group(name, parameter.requires_grad, grouping, max_block)
        if group is None:
            continue
        grouped[group].append((name, parameter))
        statuses[group].add(parameter.requires_grad)
        if (
            include_frozen
            and not parameter.requires_grad
            and (group == "frozen_encoder_base" or grouping == "fullft_blocks")
        ):
            parameter.requires_grad_(True)

    status_labels = {}
    for group, values in statuses.items():
        if not values:
            status_labels[group] = "absent"
        elif values == {True}:
            status_labels[group] = "trainable"
        elif values == {False}:
            status_labels[group] = "frozen"
        else:
            status_labels[group] = "mixed"
    return grouped, status_labels, original_flags


def print_group_inventory(
    model: nn.Module,
    include_frozen: bool,
    grouping: str,
    example_count: int = 5,
) -> None:
    grouped, statuses, original_flags = group_parameters(model, include_frozen, grouping)
    groups = groups_for(grouping)
    print(f"grouping: {grouping}")
    print(
        f"{'group':<28} {'param_count':>15} {'tensors':>8} {'status':>10}  example_parameter_names"
    )
    print("-" * 120)
    for group in groups:
        params = grouped[group]
        param_count = sum(parameter.numel() for _, parameter in params)
        examples = ", ".join(name for name, _ in params[:example_count])
        if len(params) > example_count:
            examples += ", ..."
        print(
            f"{group:<28} {param_count:>15,d} {len(params):>8,d} {statuses[group]:>10}  {examples}"
        )
    for name, parameter in model.named_parameters():
        parameter.requires_grad_(original_flags[name])


def build_dataset(config: dict[str, Any], split: str):
    normalize = config.get("transformations", {}).get("normalize", {})
    transforms = A.Compose(
        [
            A.Normalize(
                mean=normalize.get("mean", (0.5, 0.5, 0.5)),
                std=normalize.get("std", (0.5, 0.5, 0.5)),
            )
        ]
    )
    coordinator_split = "train" if split == "train" else "validation"
    return select_dataset(
        dataset_name=config["data"].get("dataset", "pannuke"),
        split=coordinator_split,
        dataset_config=config["data"],
        transforms=transforms,
    )


def build_losses(config: dict[str, Any]) -> dict[str, dict[str, dict[str, Any]]]:
    settings = config.get("loss", {})
    defaults = {
        "nuclei_binary_map": {
            "bce": ("xentropy_loss", 1.0),
            "dice": ("dice_loss", 1.0),
        },
        "hv_map": {
            "mse": ("mse_loss_maps", 1.0),
            "msge": ("msge_loss_maps", 1.0),
        },
        "nuclei_type_map": {
            "bce": ("xentropy_loss", 1.0),
            "dice": ("dice_loss", 1.0),
        },
        "tissue_types": {"ce": (None, 1.0)},
    }
    if config.get("model", {}).get("regression_loss", False):
        defaults["regression_map"] = {"mse": ("mse_loss_maps", 1.0)}

    result: dict[str, dict[str, dict[str, Any]]] = {}
    for branch, branch_defaults in defaults.items():
        configured = settings.get(
            "regression_loss" if branch == "regression_map" else branch
        )
        result[branch] = {}
        if configured:
            for name, item in configured.items():
                result[branch][name] = {
                    "fn": retrieve_loss_fn(item["loss_fn"], **item.get("args", {})),
                    "weight": float(item["weight"]),
                    "state": item.get("state", "static"),
                }
        else:
            for name, (loss_name, weight) in branch_defaults.items():
                fn = nn.CrossEntropyLoss() if loss_name is None else retrieve_loss_fn(loss_name)
                result[branch][name] = {"fn": fn, "weight": weight, "state": "static"}
    return result


def prepare_targets(
    masks: dict[str, torch.Tensor],
    tissue_names: Iterable[str],
    tissue_mapping: dict[str, int],
    num_classes: int,
    device: torch.device,
) -> dict[str, torch.Tensor]:
    binary = masks["nuclei_binary_map"].long()
    nuclei = masks["nuclei_type_map"].long()
    if binary.ndim == 4 and binary.shape[1] == 1:
        binary = binary[:, 0]
    if nuclei.ndim == 4 and nuclei.shape[1] == 1:
        nuclei = nuclei[:, 0]
    targets = {
        "nuclei_binary_map": F.one_hot(binary, num_classes=2)
        .permute(0, 3, 1, 2)
        .float()
        .to(device),
        "nuclei_type_map": F.one_hot(nuclei, num_classes=num_classes)
        .permute(0, 3, 1, 2)
        .float()
        .to(device),
        "hv_map": masks["hv_map"].float().to(device),
        "tissue_types": torch.tensor(
            [tissue_mapping[name] for name in tissue_names], dtype=torch.long, device=device
        ),
    }
    if "regression_map" in masks:
        targets["regression_map"] = masks["regression_map"].float().to(device)
    return targets


def diagnostic_loss(
    outputs: dict[str, torch.Tensor],
    targets: dict[str, torch.Tensor],
    losses: dict[str, dict[str, dict[str, Any]]],
    device: torch.device,
) -> torch.Tensor:
    predictions = dict(outputs)
    predictions["nuclei_binary_map"] = F.softmax(outputs["nuclei_binary_map"], dim=1)
    predictions["nuclei_type_map"] = F.softmax(outputs["nuclei_type_map"], dim=1)
    total = torch.zeros((), dtype=torch.float32, device=device)
    for branch, branch_losses in losses.items():
        if branch not in predictions or branch not in targets:
            continue
        for name, setting in branch_losses.items():
            if name == "msge":
                value = setting["fn"](
                    input=predictions[branch],
                    target=targets[branch],
                    focus=targets["nuclei_binary_map"],
                    device=device,
                )
            else:
                value = setting["fn"](input=predictions[branch], target=targets[branch])
            weight = setting["weight"]
            if setting["state"] == "dynamic":
                weight = 1.0 / (value.detach() + 1e-8)
            total = total + weight * value
    return total


def load_checkpoint_into_model(model: nn.Module, checkpoint: Path) -> None:
    payload = torch_load(checkpoint)
    if isinstance(payload, dict) and payload.get("format") == "cellvit_adapter_checkpoint":
        load_adapter_state(model, payload)
    else:
        model.load_state_dict(checkpoint_state_dict(payload), strict=True)
    del payload


def build_topology_for_group_print(config: dict[str, Any]) -> tuple[nn.Module, list[str]]:
    """Build model topology and configured adapters without loading checkpoint tensors."""
    seed_everything(config.get("random_seed", 42))
    model = instantiate_model(config)
    inserted = insert_adapters_from_config(model, config)
    return model, inserted


def sample_group_entries(
    entries: list[tuple[str, torch.Tensor]], max_samples: int
) -> torch.Tensor:
    total = sum(tensor.numel() for _, tensor in entries)
    if total == 0 or max_samples <= 0:
        return torch.empty(0, dtype=torch.float64)
    count = min(total, max_samples)
    targets = torch.linspace(0, total - 1, steps=count, dtype=torch.float64).round().long()
    pieces = []
    offset = 0
    for _, tensor in entries:
        end = offset + tensor.numel()
        mask = (targets >= offset) & (targets < end)
        if mask.any():
            local = targets[mask] - offset
            pieces.append(tensor.reshape(-1)[local].double())
        offset = end
    return torch.cat(pieces) if pieces else torch.empty(0, dtype=torch.float64)


def estimate_fisher(
    model: nn.Module,
    dataloader: DataLoader,
    losses: dict[str, dict[str, dict[str, Any]]],
    dataset_config: dict[str, Any],
    num_classes: int,
    num_batches: int,
    device: torch.device,
    include_frozen: bool,
    max_drift_samples: int,
    grouping: str,
) -> tuple[list[dict[str, Any]], dict[str, torch.Tensor], int, float]:
    grouped, statuses, original_flags = group_parameters(model, include_frozen, grouping)
    groups = groups_for(grouping)
    measured_names = {
        name
        for group, values in grouped.items()
        for name, parameter in values
        if parameter.requires_grad
        or (group == "frozen_encoder_base" and include_frozen)
    }
    accumulators = {
        name: torch.zeros_like(parameter, device="cpu", dtype=torch.float32)
        for name, parameter in model.named_parameters()
        if name in measured_names
    }

    model.eval()
    batch_count = 0
    loss_sum = 0.0
    for batch_index, batch in enumerate(dataloader):
        if batch_index >= num_batches:
            break
        model.zero_grad(set_to_none=True)
        images, masks, tissue_names = batch[0], batch[1], batch[2]
        images = images.to(device, dtype=torch.float32, non_blocking=True)
        targets = prepare_targets(
            masks,
            tissue_names,
            dataset_config["tissue_types"],
            num_classes,
            device,
        )
        # Explicit FP32 keeps squared gradients comparable and avoids scaled grads.
        with torch.autocast(device_type=device.type, enabled=False):
            outputs = model(images)
            loss = diagnostic_loss(outputs, targets, losses, device)
        if not torch.isfinite(loss):
            raise FloatingPointError(f"Non-finite loss in diagnostic batch {batch_index}")
        loss.backward()
        for name, parameter in model.named_parameters():
            if name not in accumulators or parameter.grad is None:
                continue
            accumulators[name].add_(parameter.grad.detach().float().cpu().square())
        batch_count += 1
        loss_sum += float(loss.detach().cpu())
        LOGGER.info("batch %d/%d loss=%.6f", batch_count, num_batches, float(loss))
        del images, masks, targets, outputs, loss

    if batch_count == 0:
        raise RuntimeError("The selected dataset yielded no batches")
    for value in accumulators.values():
        value.div_(batch_count)

    rows = []
    samples = {}
    total_mass = sum(float(value.double().sum()) for value in accumulators.values())
    for group in groups:
        params = grouped[group]
        param_count = sum(parameter.numel() for _, parameter in params)
        fisher_entries = [(name, accumulators[name]) for name, _ in params if name in accumulators]
        measured = bool(fisher_entries)
        fisher_mass = sum(float(value.double().sum()) for _, value in fisher_entries)
        fisher_max = max((float(value.max()) for _, value in fisher_entries), default=0.0)
        rows.append(
            {
                "group": group,
                "param_count": param_count,
                "fisher_mass": fisher_mass,
                "fisher_mean": fisher_mass / param_count if param_count and measured else 0.0,
                "fisher_max": fisher_max,
                "normalized_fisher_mass": fisher_mass / total_mass if total_mass else 0.0,
                "status": statuses[group],
                "measured": measured,
            }
        )
        samples[group] = sample_group_entries(fisher_entries, max_drift_samples)

    for name, parameter in model.named_parameters():
        parameter.requires_grad_(original_flags[name])
    model.zero_grad(set_to_none=True)
    del accumulators
    gc.collect()
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return rows, samples, batch_count, loss_sum / batch_count


def js_distance(a: torch.Tensor, b: torch.Tensor) -> float:
    count = min(a.numel(), b.numel())
    if count == 0:
        return 0.0
    a = torch.log1p(a[:count].double().clamp_min(0))
    b = torch.log1p(b[:count].double().clamp_min(0))
    sum_a, sum_b = float(a.sum()), float(b.sum())
    if sum_a == 0.0 and sum_b == 0.0:
        return 0.0
    if sum_a == 0.0:
        a.fill_(1.0 / count)
    else:
        a.div_(sum_a)
    if sum_b == 0.0:
        b.fill_(1.0 / count)
    else:
        b.div_(sum_b)
    midpoint = 0.5 * (a + b)
    kl_a = torch.where(a > 0, a * torch.log(a / midpoint), 0).sum()
    kl_b = torch.where(b > 0, b * torch.log(b / midpoint), 0).sum()
    return float(torch.sqrt(0.5 * (kl_a + kl_b)).clamp_min(0))


def write_csv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str]) -> None:
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def plot_fisher(
    rows: list[dict[str, Any]], checkpoints: list[str], output: Path, groups: tuple[str, ...]
) -> None:
    import matplotlib.pyplot as plt

    width = 0.8 / len(checkpoints)
    x = np.arange(len(groups))
    fig, axis = plt.subplots(figsize=(max(11, len(groups) * 1.25), 6))
    for index, checkpoint in enumerate(checkpoints):
        lookup = {
            row["group"]: row["normalized_fisher_mass"]
            for row in rows
            if row["checkpoint"] == checkpoint
        }
        axis.bar(
            x - 0.4 + width / 2 + index * width,
            [lookup.get(group, 0.0) for group in groups],
            width,
            label=Path(checkpoint).name,
        )
    axis.set_xticks(x, groups, rotation=35, ha="right")
    axis.set_ylabel("Normalized Fisher mass")
    axis.set_title("Diagonal Fisher mass by module group")
    axis.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(output, dpi=180)
    plt.close(fig)


def plot_drift(drift_rows: list[dict[str, Any]], output: Path, groups: tuple[str, ...]) -> None:
    import matplotlib.pyplot as plt

    pairs = list(dict.fromkeys((row["checkpoint_a"], row["checkpoint_b"]) for row in drift_rows))
    values = np.zeros((len(groups), len(pairs)), dtype=float)
    for row in drift_rows:
        values[groups.index(row["group"]), pairs.index((row["checkpoint_a"], row["checkpoint_b"]))] = row["js_distance"]
    labels = [f"{Path(a).name} → {Path(b).name}" for a, b in pairs]
    fig, axis = plt.subplots(figsize=(max(8, len(pairs) * 2.2), 7))
    image = axis.imshow(values, aspect="auto", cmap="magma", vmin=0)
    axis.set_xticks(range(len(labels)), labels, rotation=30, ha="right")
    axis.set_yticks(range(len(groups)), groups)
    axis.set_title("Jensen–Shannon Fisher drift")
    fig.colorbar(image, ax=axis, label="JS distance")
    fig.tight_layout()
    fig.savefig(output, dpi=180)
    plt.close(fig)


def main() -> None:
    args = parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
    if args.num_batches < 1:
        raise ValueError("--num-batches must be at least 1")
    if args.max_drift_samples < 1:
        raise ValueError("--max-drift-samples must be at least 1")
    groups = groups_for(args.grouping)

    config_path = resolve_path(args.config)
    config = load_yaml(config_path)
    output_dir = resolve_path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_paths = [resolve_path(args.checkpoint)]
    checkpoint_paths.extend(resolve_path(path) for path in (args.compare_checkpoints or []))
    checkpoint_paths = list(dict.fromkeys(checkpoint_paths))
    for path in checkpoint_paths:
        if not path.is_file():
            raise FileNotFoundError(f"Checkpoint not found: {path}")

    device = torch.device("cpu" if args.print_groups else args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is not available")
    if args.print_groups:
        LOGGER.info("building model topology from %s", config_path)
        model, inserted = build_topology_for_group_print(config)
        LOGGER.info(
            "checkpoint path verified=%s; decoder conv adapters inserted=%d",
            checkpoint_paths[0],
            len(inserted),
        )
        print_group_inventory(
            model=model,
            include_frozen=args.include_frozen,
            grouping=args.grouping,
        )
        return

    base_checkpoint = resolve_path(Path(config["model"]["pretrained"]))
    LOGGER.info("building model from %s", config_path)
    model, load_info, inserted = build_model(config, base_checkpoint)
    LOGGER.info(
        "base tensors loaded=%d; decoder conv adapters inserted=%d",
        load_info["loaded_tensors"],
        len(inserted),
    )
    model.to(device)

    dataset_config = load_yaml(resolve_path(Path(config["data"]["dataset_path"])) / "dataset_config.yaml")
    dataset = build_dataset(config, args.dataset_split)
    dataloader = DataLoader(
        dataset,
        batch_size=config["training"]["batch_size"],
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
    )
    LOGGER.info(
        "dataset split=%s samples=%d batch_size=%d; deterministic normalization-only ordering",
        args.dataset_split,
        len(dataset),
        config["training"]["batch_size"],
    )
    if args.include_frozen:
        LOGGER.warning("including frozen encoder gradients; this is substantially more expensive")
    losses = build_losses(config)

    all_rows: list[dict[str, Any]] = []
    sample_sets: list[dict[str, torch.Tensor]] = []
    checkpoint_meta = []
    for checkpoint in checkpoint_paths:
        LOGGER.info("loading checkpoint %s", checkpoint)
        load_checkpoint_into_model(model, checkpoint)
        rows, samples, batches_used, mean_loss = estimate_fisher(
            model=model,
            dataloader=dataloader,
            losses=losses,
            dataset_config=dataset_config,
            num_classes=config["data"]["num_nuclei_classes"],
            num_batches=args.num_batches,
            device=device,
            include_frozen=args.include_frozen,
            max_drift_samples=args.max_drift_samples,
            grouping=args.grouping,
        )
        checkpoint_label = str(checkpoint)
        for row in rows:
            all_rows.append({"checkpoint": checkpoint_label, **row})
        sample_sets.append(samples)
        checkpoint_meta.append(
            {"checkpoint": checkpoint_label, "batches_used": batches_used, "mean_loss": mean_loss}
        )
        LOGGER.info("checkpoint complete: batches=%d mean_loss=%.6f", batches_used, mean_loss)

    summary_fields = [
        "checkpoint",
        "group",
        "param_count",
        "fisher_mass",
        "fisher_mean",
        "fisher_max",
        "normalized_fisher_mass",
        "status",
        "measured",
    ]
    write_csv(output_dir / "fisher_summary.csv", all_rows, summary_fields)

    drift_rows: list[dict[str, Any]] = []
    for index in range(len(checkpoint_paths) - 1):
        rows_a = {row["group"]: row for row in all_rows if row["checkpoint"] == str(checkpoint_paths[index])}
        rows_b = {row["group"]: row for row in all_rows if row["checkpoint"] == str(checkpoint_paths[index + 1])}
        for group in groups:
            drift_rows.append(
                {
                    "checkpoint_a": str(checkpoint_paths[index]),
                    "checkpoint_b": str(checkpoint_paths[index + 1]),
                    "group": group,
                    "param_count": min(rows_a[group]["param_count"], rows_b[group]["param_count"]),
                    "fisher_mass_a": rows_a[group]["fisher_mass"],
                    "fisher_mass_b": rows_b[group]["fisher_mass"],
                    "js_distance": js_distance(sample_sets[index][group], sample_sets[index + 1][group]),
                }
            )
    if drift_rows:
        write_csv(
            output_dir / "fisher_drift_summary.csv",
            drift_rows,
            [
                "checkpoint_a",
                "checkpoint_b",
                "group",
                "param_count",
                "fisher_mass_a",
                "fisher_mass_b",
                "js_distance",
            ],
        )

    json_payload = {
        "config": str(config_path),
        "dataset_split": args.dataset_split,
        "requested_num_batches": args.num_batches,
        "grouping": args.grouping,
        "include_frozen": args.include_frozen,
        "precision": "float32 forward/backward and Fisher accumulation",
        "checkpoint_runs": checkpoint_meta,
        "groups": all_rows,
        "drift": drift_rows,
    }
    with (output_dir / "fisher_summary.json").open("w") as handle:
        json.dump(json_payload, handle, indent=2)

    labels = [str(path) for path in checkpoint_paths]
    plot_fisher(all_rows, labels, output_dir / "normalized_fisher_mass.png", groups)
    if drift_rows:
        plot_drift(drift_rows, output_dir / "fisher_drift_heatmap.png", groups)
    LOGGER.info("wrote diagnostic outputs to %s", output_dir)


if __name__ == "__main__":
    main()
