"""Small public API for loading STHELAR adapters on CellViT-SAM-H-x40."""

import json
from pathlib import Path

import torch

from utils.adapter_checkpoint import (
    build_model,
    checkpoint_state_dict,
    load_adapter_state,
    load_yaml,
)


SUPPORTED_BASE_MODELS = {"cellvit-sam-h-x40"}


def _resolve(path):
    path = Path(path).expanduser()
    if not path.is_absolute():
        path = Path.cwd() / path
    return path.resolve()


def get_adapter_metadata(adapter_checkpoint):
    adapter_checkpoint = _resolve(adapter_checkpoint)
    payload = torch.load(str(adapter_checkpoint), map_location="cpu")
    if payload.get("format") != "cellvit_adapter_checkpoint":
        raise ValueError(f"Not a CellViT adapter checkpoint: {adapter_checkpoint}")
    return payload["metadata"]


def load_cellvit_base(
    model_name,
    base_checkpoint,
    config_path,
    device="cpu",
    eval_mode=True,
):
    normalized_name = str(model_name).lower()
    if normalized_name not in SUPPORTED_BASE_MODELS:
        raise ValueError(
            f"Unsupported model_name={model_name!r}; supported={sorted(SUPPORTED_BASE_MODELS)}"
        )
    config_path = _resolve(config_path)
    base_checkpoint = _resolve(base_checkpoint)
    config = load_yaml(config_path)
    if str(config["model"]["backbone"]).upper() != "SAM-H":
        raise ValueError("cellvit-sam-h-x40 requires model.backbone: SAM-H")

    model, load_info, inserted = build_model(config, base_checkpoint)
    model.to(device)
    if eval_mode:
        model.eval()
    model._cellvit_adapter_config = config
    model._cellvit_adapter_config_path = str(config_path)
    model._cellvit_base_checkpoint = str(base_checkpoint)
    model._loaded_sthelar_adapter_metadata = None
    print(
        f"Loaded {normalized_name} base and adapter architecture: "
        f"base_tensors={load_info['loaded_tensors']} "
        f"decoder_conv_adapters={len(inserted)} device={device}"
    )
    return model


def load_sthelar_adapter(
    cellvit,
    adapter_checkpoint,
    print_metadata=True,
    allow_replace=False,
):
    if not hasattr(cellvit, "_cellvit_adapter_config"):
        raise ValueError(
            "Model was not created by load_cellvit_base; adapter architecture is unknown"
        )
    if (
        cellvit._loaded_sthelar_adapter_metadata is not None
        and not allow_replace
    ):
        raise RuntimeError(
            "A STHELAR adapter is already loaded. Rebuild the base model or pass "
            "allow_replace=True for a compatible adapter architecture."
        )

    adapter_checkpoint = _resolve(adapter_checkpoint)
    payload = torch.load(str(adapter_checkpoint), map_location="cpu")
    if payload.get("format") != "cellvit_adapter_checkpoint":
        raise ValueError(f"Not a CellViT adapter checkpoint: {adapter_checkpoint}")
    metadata = payload["metadata"]
    config = cellvit._cellvit_adapter_config
    expected_nuclei = int(config["data"]["num_nuclei_classes"])
    expected_tissues = int(config["data"]["num_tissue_classes"])
    if int(metadata["num_nuclei_classes"]) != expected_nuclei:
        raise ValueError(
            "Adapter num_nuclei_classes does not match the configured model: "
            f"{metadata['num_nuclei_classes']} != {expected_nuclei}"
        )
    if int(metadata["num_tissue_classes"]) != expected_tissues:
        raise ValueError(
            "Adapter num_tissue_classes does not match the configured model: "
            f"{metadata['num_tissue_classes']} != {expected_tissues}"
        )
    config_adapter_type = config.get("adapters", {}).get("adapter_type")
    config_decoder_scope = config.get("adapters", {}).get(
        "decoder_train_scope", "all"
    )
    if metadata.get("adapter_type") != config_adapter_type:
        raise ValueError(
            f"Adapter type mismatch: {metadata.get('adapter_type')} != {config_adapter_type}"
        )
    if metadata.get("decoder_train_scope", "all") != config_decoder_scope:
        raise ValueError(
            "Decoder scope mismatch: "
            f"{metadata.get('decoder_train_scope')} != {config_decoder_scope}"
        )

    loaded = load_adapter_state(cellvit, payload)
    cellvit._loaded_sthelar_adapter_metadata = metadata
    cellvit.eval()
    print(f"Loaded STHELAR adapter tensors: {len(loaded)} from {adapter_checkpoint}")
    if print_metadata:
        print(json.dumps(metadata, indent=2, sort_keys=True))
    return cellvit


def add_liver_adapter(cellvit, adapter_checkpoint, **kwargs):
    return load_sthelar_adapter(cellvit, adapter_checkpoint, **kwargs)


def add_tonsil_adapter(cellvit, adapter_checkpoint, **kwargs):
    return load_sthelar_adapter(cellvit, adapter_checkpoint, **kwargs)


def verify_adapter_equivalence(
    base_checkpoint,
    adapter_checkpoint,
    config_path,
    full_checkpoint=None,
):
    adapter_checkpoint = _resolve(adapter_checkpoint)
    payload = torch.load(str(adapter_checkpoint), map_location="cpu")
    metadata = payload["metadata"]
    full_checkpoint = _resolve(full_checkpoint or metadata["source_checkpoint"])
    config_path = _resolve(config_path)
    base_checkpoint = _resolve(base_checkpoint)
    config = load_yaml(config_path)

    adapter_model = load_cellvit_base(
        "cellvit-sam-h-x40",
        base_checkpoint,
        config_path,
        device="cpu",
    )
    load_sthelar_adapter(adapter_model, adapter_checkpoint, print_metadata=False)

    reference_model, _, _ = build_model(config, base_checkpoint)
    full_payload = torch.load(str(full_checkpoint), map_location="cpu")
    reference_model.load_state_dict(
        checkpoint_state_dict(full_payload), strict=True
    )
    mismatches = []
    for key, value in reference_model.state_dict().items():
        candidate = adapter_model.state_dict()[key]
        if not torch.equal(value, candidate):
            difference = (
                float((value - candidate).abs().max())
                if value.is_floating_point()
                else 1.0
            )
            mismatches.append((key, difference))
    if mismatches:
        raise AssertionError(
            f"Adapter equivalence failed for {len(mismatches)} tensors: {mismatches[:10]}"
        )
    print("Adapter equivalence: all model state tensors match exactly")
    return {"matched": True, "tensor_count": len(reference_model.state_dict())}
