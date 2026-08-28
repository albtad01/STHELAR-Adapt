"""Shared model reconstruction and adapter-checkpoint helpers."""

import json
import random
import subprocess
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import yaml

from models.adapters.lora_ntonly import set_lora_ntonly_trainable
from models.adapters.utils import (
    apply_peft_trainability,
    freeze_all,
    insert_adaptformer,
    insert_decoder_conv_adapters,
    insert_lora2,
    insert_vera,
    set_ntonly_trainable,
)
from models.segmentation.cell_segmentation.cellvit import CellViT256, CellViTSAM
from models.segmentation.cell_segmentation.cellvit_shared import (
    CellViT256Shared,
    CellViTSAMShared,
)
from models.segmentation.cell_segmentation.utils import Conv2DBlock


FORMAT_VERSION = 1


def seed_everything(seed):
    seed = int(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def load_yaml(path):
    with Path(path).open() as handle:
        return yaml.safe_load(handle)


def resolve_run_dir(path):
    path = Path(path).expanduser().resolve()
    if (path / "config.yaml").is_file() and (path / "checkpoints").is_dir():
        return path
    log_dir = path / "log"
    candidates = []
    if log_dir.is_dir():
        candidates = [
            candidate
            for candidate in log_dir.iterdir()
            if candidate.is_dir()
            and (candidate / "config.yaml").is_file()
            and (candidate / "checkpoints").is_dir()
        ]
    if not candidates:
        raise FileNotFoundError(
            f"Could not resolve a timestamped run directory from: {path}"
        )
    return sorted(candidates, key=lambda candidate: candidate.name)[-1]


def resolve_checkpoint_path(run_dir, checkpoint):
    checkpoint = Path(checkpoint)
    if checkpoint.is_absolute() and checkpoint.is_file():
        return checkpoint
    direct = (Path.cwd() / checkpoint).resolve()
    if direct.is_file():
        return direct
    under_run = Path(run_dir) / "checkpoints" / checkpoint
    if under_run.is_file():
        return under_run.resolve()
    raise FileNotFoundError(f"Checkpoint not found: {checkpoint}")


def checkpoint_state_dict(payload):
    if isinstance(payload, dict) and "model_state_dict" in payload:
        return payload["model_state_dict"]
    if isinstance(payload, dict) and all(
        isinstance(key, str) for key in payload.keys()
    ):
        return payload
    raise ValueError("Unsupported checkpoint format: no model_state_dict")


def instantiate_model(config):
    backbone = str(config["model"]["backbone"]).upper()
    if backbone == "VIT256":
        model_class = (
            CellViT256Shared
            if config["model"].get("shared_decoders", False)
            else CellViT256
        )
        return model_class(
            model256_path=config["model"].get("pretrained_encoder"),
            num_nuclei_classes=config["data"]["num_nuclei_classes"],
            num_tissue_classes=config["data"]["num_tissue_classes"],
            drop_rate=config["training"].get("drop_rate", 0),
            attn_drop_rate=config["training"].get("attn_drop_rate", 0),
            drop_path_rate=config["training"].get("drop_path_rate", 0),
            regression_loss=config["training"].get("regression_loss", False),
        )
    if backbone not in {"SAM-B", "SAM-L", "SAM-H"}:
        raise NotImplementedError(
            "Adapter-only export currently supports ViT256 and SAM CellViT backbones; "
            f"received {backbone}"
        )
    model_class = (
        CellViTSAMShared
        if config["model"].get("shared_decoders", False)
        else CellViTSAM
    )
    return model_class(
        model_path=config["model"].get("pretrained_encoder"),
        num_nuclei_classes=config["data"]["num_nuclei_classes"],
        num_tissue_classes=config["data"]["num_tissue_classes"],
        vit_structure=backbone,
        drop_rate=config["training"].get("drop_rate", 0),
        regression_loss=config["training"].get("regression_loss", False),
    )


def _reinitialize_transfer_heads(model):
    for block in model.nuclei_type_maps_decoder.decoder0_header:
        if isinstance(block, Conv2DBlock):
            for layer in block.block:
                if isinstance(layer, nn.Conv2d):
                    nn.init.kaiming_normal_(
                        layer.weight, mode="fan_out", nonlinearity="relu"
                    )
                    if layer.bias is not None:
                        nn.init.zeros_(layer.bias)
                elif isinstance(layer, nn.BatchNorm2d):
                    nn.init.ones_(layer.weight)
                    nn.init.zeros_(layer.bias)
        elif isinstance(block, nn.Conv2d):
            nn.init.kaiming_normal_(block.weight, mode="fan_out", nonlinearity="relu")
            if block.bias is not None:
                nn.init.zeros_(block.bias)
    if isinstance(model.classifier_head, nn.Linear):
        nn.init.xavier_normal_(model.classifier_head.weight)
        nn.init.zeros_(model.classifier_head.bias)


def load_base_checkpoint(model, checkpoint_path, reproduce_training_init=True):
    payload = torch.load(str(checkpoint_path), map_location="cpu")
    source = checkpoint_state_dict(payload)
    target = model.state_dict()
    compatible = {
        key: value
        for key, value in source.items()
        if key in target and target[key].shape == value.shape
    }
    skipped_shape = {
        key: {"checkpoint": list(value.shape), "model": list(target[key].shape)}
        for key, value in source.items()
        if key in target and target[key].shape != value.shape
    }
    missing_target = [key for key in target if key not in compatible]
    model.load_state_dict(compatible, strict=False)
    # This mirrors the current training code exactly. It matters for old runs in
    # which a classifier mismatch also triggered deterministic NT-head init.
    if reproduce_training_init and missing_target:
        _reinitialize_transfer_heads(model)
    del payload
    return {
        "loaded_tensors": len(compatible),
        "missing_target_keys": missing_target,
        "skipped_shape": skipped_shape,
    }


def insert_adapters_from_config(model, config):
    adapter_config = config.get("adapters", {})
    adapter_type = str(adapter_config.get("adapter_type", "freeze")).lower()
    decoder_scope = str(adapter_config.get("decoder_train_scope", "all")).lower()

    if adapter_type in {
        "lora",
        "lora_ntonly",
        "lora_adaptformer",
        "lora_adaptformer_ntonly",
    }:
        lora = adapter_config["lora"]
        insert_lora2(
            model,
            rank=lora["rank"],
            alpha=lora["alpha"],
            targets=lora.get("targets", ["q", "v"]),
            dropout=lora.get("dropout", 0.0),
        )

    if adapter_type in {
        "adaptformer",
        "lora_adaptformer",
        "lora_adaptformer_ntonly",
        "vera_adaptformer",
    }:
        adaptformer = adapter_config["adaptformer"]
        insert_adaptformer(
            model,
            adaptformer["activation"],
            adaptformer["reduction"],
        )

    if adapter_type in {"vera", "vera_adaptformer", "vera_ntonly", "vera_adaptformer_ntonly"}:
        vera = adapter_config.get("vera", {})
        insert_vera(
            model,
            rank=vera.get("rank", adapter_config.get("vera_rank", 8)),
            alpha=vera.get("alpha", adapter_config.get("vera_alpha", 8)),
            targets=vera.get(
                "targets", adapter_config.get("vera_targets", ["q", "v"])
            ),
            dropout=vera.get(
                "dropout", adapter_config.get("vera_dropout", 0.0)
            ),
            shared_matrices=vera.get(
                "shared_matrices",
                adapter_config.get("vera_shared_matrices", True),
            ),
            train_alpha=vera.get(
                "train_alpha", adapter_config.get("vera_train_alpha", True)
            ),
            seed=vera.get(
                "seed", adapter_config.get("vera_seed", config.get("random_seed", 42))
            ),
        )

    inserted_decoder_adapters = []
    if decoder_scope == "conv_adapters":
        inserted_decoder_adapters = insert_decoder_conv_adapters(
            model,
            reduction=adapter_config.get("decoder_adapter_reduction", 16),
            activation=adapter_config.get("decoder_adapter_activation", "GELU"),
            alpha_init=adapter_config.get("decoder_adapter_alpha_init", 1.0),
            train_alpha=adapter_config.get("decoder_adapter_train_alpha", True),
        )

    peft_modes = {
        "lora",
        "adaptformer",
        "lora_adaptformer",
        "vera",
        "vera_adaptformer",
    }
    if adapter_type in peft_modes:
        apply_peft_trainability(model, decoder_scope)
    elif adapter_type in {"lora_ntonly", "lora_adaptformer_ntonly"}:
        set_lora_ntonly_trainable(model)
    elif adapter_type == "ntonly":
        freeze_all(model)
        set_ntonly_trainable(model)
    elif adapter_type == "freeze":
        freeze_all(model)
    elif adapter_type in {"fullft", "all"}:
        for parameter in model.parameters():
            parameter.requires_grad = True
    else:
        raise NotImplementedError(
            f"Adapter-only reconstruction is not implemented for {adapter_type!r}"
        )
    return inserted_decoder_adapters


def build_model(config, base_checkpoint):
    seed_everything(config.get("random_seed", 42))
    model = instantiate_model(config)
    # The SAM transfer path deliberately reinitializes the NT/tissue transfer
    # heads when a base tensor has an incompatible shape.  The strict ViT256
    # campaign does not; it preserves the constructor initialization for the
    # one-way tissue head.  Mirror those two training paths exactly so frozen
    # parameters can be checked rather than silently included in an adapter.
    reproduce_training_init = str(config["model"]["backbone"]).upper() != "VIT256"
    load_info = load_base_checkpoint(
        model,
        base_checkpoint,
        reproduce_training_init=reproduce_training_init,
    )
    inserted = insert_adapters_from_config(model, config)
    return model, load_info, inserted


def git_commit(repo_root):
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            cwd=str(repo_root),
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def dataset_metadata(config):
    dataset_path = Path(config["data"]["dataset_path"])
    dataset_config = load_yaml(dataset_path / "dataset_config.yaml")
    split_manifest_path = dataset_path / "split_manifest.yaml"
    split_manifest = (
        load_yaml(split_manifest_path) if split_manifest_path.is_file() else {}
    )
    return dataset_config, split_manifest


def adapter_metadata(config, run_dir, checkpoint_path, base_checkpoint, model):
    adapter_config = config.get("adapters", {})
    dataset_config, split_manifest = dataset_metadata(config)
    total_params = sum(parameter.numel() for parameter in model.parameters())
    trainable_params = sum(
        parameter.numel() for parameter in model.parameters() if parameter.requires_grad
    )
    repo_root = Path(__file__).resolve().parents[1]
    return {
        "format_version": FORMAT_VERSION,
        "base_model": (
            "CellViT-256-x40"
            if str(config["model"]["backbone"]).upper() == "VIT256"
            else "CellViT-{}-x40".format(
                str(config["model"]["backbone"]).upper()
            )
        ),
        "base_checkpoint": config["model"].get("pretrained"),
        "base_checkpoint_resolved": str(base_checkpoint),
        "adapter_type": adapter_config.get("adapter_type"),
        "decoder_train_scope": adapter_config.get("decoder_train_scope", "all"),
        "lora": adapter_config.get("lora"),
        "adaptformer": adapter_config.get("adaptformer"),
        "vera": adapter_config.get("vera"),
        "decoder_adapter": {
            "reduction": adapter_config.get("decoder_adapter_reduction"),
            "activation": adapter_config.get("decoder_adapter_activation"),
            "alpha_init": adapter_config.get("decoder_adapter_alpha_init"),
            "train_alpha": adapter_config.get("decoder_adapter_train_alpha"),
        },
        "num_nuclei_classes": config["data"]["num_nuclei_classes"],
        "num_tissue_classes": config["data"]["num_tissue_classes"],
        "label_mapping": dataset_config.get("nuclei_types"),
        "nuclei_types": dataset_config.get("nuclei_types"),
        "tissue_types": dataset_config.get("tissue_types"),
        "label_mode": split_manifest.get("label_mode"),
        "tissue": split_manifest.get("tissue"),
        "split": split_manifest.get("actual_strategy"),
        "trainable_parameter_count": trainable_params,
        "trainable_params": trainable_params,
        "total_parameter_count": total_params,
        "trainable_ratio_percent": 100.0 * trainable_params / total_params,
        "trainable_ratio": 100.0 * trainable_params / total_params,
        "original_run_name": config.get("logging", {}).get(
            "log_comment", Path(run_dir).name
        ),
        "run_name": config.get("logging", {}).get(
            "log_comment", Path(run_dir).name
        ),
        "original_run_dir": str(run_dir),
        "original_config_path": str(Path(run_dir) / "config.yaml"),
        "config_path": str(Path(run_dir) / "config.yaml"),
        "source_checkpoint": str(checkpoint_path),
        "git_commit": git_commit(repo_root),
    }


def load_adapter_state(model, adapter_payload):
    target = model.state_dict()
    loaded = []
    for section in ("adapter_state_dict", "mutable_buffer_state_dict"):
        for key, value in adapter_payload.get(section, {}).items():
            if key not in target:
                raise KeyError(f"Adapter checkpoint key not found in model: {key}")
            if target[key].shape != value.shape:
                raise ValueError(
                    f"Shape mismatch for {key}: adapter={tuple(value.shape)} "
                    f"model={tuple(target[key].shape)}"
                )
            target[key].copy_(value)
            loaded.append(key)
    return loaded


def metadata_as_json(metadata):
    return json.dumps(metadata, indent=2, sort_keys=True)
