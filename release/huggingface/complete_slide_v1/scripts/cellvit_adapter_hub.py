"""Small public API for loading STHELAR adapters on CellViT-SAM-H-x40."""

import json
import os
from pathlib import Path

import torch

from utils.adapter_checkpoint import (
    build_model,
    checkpoint_state_dict,
    load_adapter_state,
    load_yaml,
)


SUPPORTED_BASE_MODELS = {"cellvit-sam-h-x40"}
DEFAULT_BASE_CHECKPOINT = "models/pretrained/CellViT-SAM-H-x40.pth"

_REGISTERED_ADAPTER_ALIASES = {
    "ovary_5": {
        "local": "adapters/sthelar40x_ovary_5class_spatial_margin128_lora_adaptformer_r8_a8_red16_decoder_heads_only_lr5e-5_e10_seed42_CLEAN_adapter.pth",
        "repo_env": "STHELAR_OVARY_5_REPO",
        "repo_name": "cellvit-sthelar-ovary-5",
    },
}


def _default_sthelar_5class_heads_only_config(base_checkpoint):
    return {
        "random_seed": 42,
        "data": {
            "num_nuclei_classes": 6,
            "num_tissue_classes": 1,
        },
        "model": {
            "backbone": "SAM-H",
            "pretrained": str(base_checkpoint),
            "shared_decoders": False,
        },
        "training": {
            "drop_rate": 0,
            "regression_loss": False,
        },
        "adapters": {
            "adapter_type": "lora_adaptformer",
            "decoder_train_scope": "heads_only",
            "lora": {
                "rank": 8,
                "alpha": 8,
                "targets": ["q", "v"],
                "dropout": 0.0,
            },
            "adaptformer": {
                "activation": "GELU",
                "reduction": 16,
            },
        },
    }


def _resolve(path):
    path = Path(path).expanduser()
    if not path.is_absolute():
        path = Path.cwd() / path
    return path.resolve()


def _load_adapter_payload(path):
    path = Path(path).expanduser()
    if path.suffix == ".safetensors":
        try:
            from safetensors import safe_open
        except ImportError as exc:
            raise ImportError(
                "Loading .safetensors adapters requires the safetensors package"
            ) from exc

        adapter_state = {}
        mutable_buffers = {}
        metadata = {}
        with safe_open(str(path), framework="pt", device="cpu") as handle:
            raw_metadata = handle.metadata() or {}
            if raw_metadata.get("metadata_json"):
                metadata = json.loads(raw_metadata["metadata_json"])
            for key in handle.keys():
                tensor = handle.get_tensor(key)
                if key.startswith("adapter_state_dict."):
                    adapter_state[key.removeprefix("adapter_state_dict.")] = tensor
                elif key.startswith("mutable_buffer_state_dict."):
                    mutable_buffers[
                        key.removeprefix("mutable_buffer_state_dict.")
                    ] = tensor
        return {
            "format": "cellvit_adapter_checkpoint",
            "metadata": metadata,
            "adapter_state_dict": adapter_state,
            "mutable_buffer_state_dict": mutable_buffers,
        }
    return torch.load(str(path), map_location="cpu")


def _adapter_weight_from_package(package_dir):
    package_dir = _resolve(package_dir)
    adapter_config_path = package_dir / "adapter_config.json"
    legacy_config_path = package_dir / "adapter_config.yaml"
    if adapter_config_path.is_file() or legacy_config_path.is_file():
        if adapter_config_path.is_file():
            adapter_config = json.loads(adapter_config_path.read_text())
        else:
            adapter_config_path = legacy_config_path
            adapter_config = load_yaml(adapter_config_path)
        weight_name = adapter_config.get("adapter_weight_file", "adapter_model.pth")
        weight_path = Path(weight_name)
        if not weight_path.is_absolute():
            weight_path = package_dir / weight_path
        if weight_path.is_file():
            return weight_path.resolve()
        raise FileNotFoundError(f"Adapter weights not found: {weight_path}")

    candidates = [
        package_dir / "adapter_model.safetensors",
        package_dir / "adapter_model.pth",
    ]
    candidates.extend(sorted(package_dir.glob("*_adapter.pth")))
    candidates.extend(sorted(package_dir.glob("*.safetensors")))
    candidates.extend(sorted(package_dir.glob("*.pth")))
    existing = []
    seen = set()
    for candidate in candidates:
        if candidate.is_file() and candidate not in seen:
            existing.append(candidate)
            seen.add(candidate)
    if len(existing) == 1:
        return existing[0].resolve()
    if not existing:
        raise FileNotFoundError(
            f"No adapter weights found in {package_dir}; expected adapter_config.json, "
            "legacy adapter_config.yaml, "
            "adapter_model.safetensors, adapter_model.pth, or *_adapter.pth"
        )
    raise FileNotFoundError(
        f"Multiple possible adapter weights found in {package_dir}: "
        f"{[path.name for path in existing]}. Add adapter_config.yaml or pass a file."
    )


def _download_hf_adapter(repo_id, revision=None, cache_dir=None):
    try:
        from huggingface_hub import snapshot_download
    except ImportError as exc:
        raise ImportError(
            "Resolving Hugging Face adapter repos requires huggingface_hub. "
            "Install it or pass a local adapter path."
        ) from exc
    return Path(
        snapshot_download(repo_id=repo_id, revision=revision, cache_dir=cache_dir)
    ).resolve()


def _repo_id_from_alias(alias, alias_entry):
    specific = os.environ.get(alias_entry.get("repo_env", ""))
    if specific:
        return specific
    namespace = os.environ.get("STHELAR_HF_NAMESPACE")
    if namespace:
        return f"{namespace.rstrip('/')}/{alias_entry['repo_name']}"
    raise FileNotFoundError(
        f"Adapter alias {alias!r} was not found locally. Set "
        f"{alias_entry.get('repo_env', 'STHELAR_<ALIAS>_REPO')} to the HF repo id, "
        "or set STHELAR_HF_NAMESPACE so the repo id can be derived."
    )


def register_adapter_alias(alias, source):
    """Register or override a STHELAR adapter alias.

    `source` can be a local adapter `.pth`/`.safetensors`, a local HF-style
    adapter directory, or a Hugging Face repo id.
    """
    _REGISTERED_ADAPTER_ALIASES[str(alias)] = {"source": str(source)}


def resolve_adapter_checkpoint(adapter, prefer_hf=False, revision=None, cache_dir=None):
    """Resolve an adapter alias, local path, package directory, or HF repo id."""
    adapter = str(adapter)
    alias_entry = _REGISTERED_ADAPTER_ALIASES.get(adapter)
    if alias_entry is not None:
        if "source" in alias_entry:
            return resolve_adapter_checkpoint(
                alias_entry["source"],
                prefer_hf=prefer_hf,
                revision=revision,
                cache_dir=cache_dir,
            )
        local = alias_entry.get("local")
        if local and not prefer_hf:
            local_path = _resolve(local)
            if local_path.is_file():
                return local_path
            if local_path.is_dir():
                return _adapter_weight_from_package(local_path)
        adapter = _repo_id_from_alias(adapter, alias_entry)

    candidate = Path(adapter).expanduser()
    if candidate.exists():
        candidate = _resolve(candidate)
        if candidate.is_dir():
            return _adapter_weight_from_package(candidate)
        return candidate

    if "/" not in adapter:
        raise FileNotFoundError(
            f"Adapter {adapter!r} is neither a registered alias nor a local path"
        )
    package_dir = _download_hf_adapter(adapter, revision=revision, cache_dir=cache_dir)
    return _adapter_weight_from_package(package_dir)


def get_adapter_metadata(adapter_checkpoint, **resolve_kwargs):
    adapter_checkpoint = resolve_adapter_checkpoint(adapter_checkpoint, **resolve_kwargs)
    payload = _load_adapter_payload(adapter_checkpoint)
    if payload.get("format") != "cellvit_adapter_checkpoint":
        raise ValueError(f"Not a CellViT adapter checkpoint: {adapter_checkpoint}")
    return payload["metadata"]


def model(
    model_name,
    base_checkpoint=None,
    config_path=None,
    device="cpu",
    eval_mode=True,
):
    """Load a CellViT base model ready for STHELAR 5-class heads_only adapters.

    This intentionally defaults to the selected tissue-specific PEFT architecture:
    LoRA+AdaptFormer r8/a8/red16 with decoder `heads_only`.
    """
    base_checkpoint = _resolve(
        base_checkpoint
        or os.environ.get("CELLVIT_SAM_H_X40_CHECKPOINT", DEFAULT_BASE_CHECKPOINT)
    )
    if config_path is not None:
        return load_cellvit_base(
            model_name=model_name,
            base_checkpoint=base_checkpoint,
            config_path=config_path,
            device=device,
            eval_mode=eval_mode,
        )

    normalized_name = str(model_name).lower()
    if normalized_name not in SUPPORTED_BASE_MODELS:
        raise ValueError(
            f"Unsupported model_name={model_name!r}; supported={sorted(SUPPORTED_BASE_MODELS)}"
        )
    if not base_checkpoint.is_file():
        raise FileNotFoundError(
            f"Base checkpoint not found: {base_checkpoint}. Pass base_checkpoint=... "
            "or set CELLVIT_SAM_H_X40_CHECKPOINT."
        )
    config = _default_sthelar_5class_heads_only_config(base_checkpoint)
    cellvit, load_info, inserted = build_model(config, base_checkpoint)
    cellvit.to(device)
    if eval_mode:
        cellvit.eval()
    cellvit._cellvit_adapter_config = config
    cellvit._cellvit_adapter_config_path = None
    cellvit._cellvit_base_checkpoint = str(base_checkpoint)
    cellvit._loaded_sthelar_adapter_metadata = None
    print(
        f"Loaded {normalized_name} base with default STHELAR 5-class "
        f"LoRA+AdaptFormer heads_only architecture: "
        f"base_tensors={load_info['loaded_tensors']} "
        f"decoder_conv_adapters={len(inserted)} device={device}"
    )
    return cellvit


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
    prefer_hf=False,
    revision=None,
    cache_dir=None,
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

    adapter_checkpoint = resolve_adapter_checkpoint(
        adapter_checkpoint,
        prefer_hf=prefer_hf,
        revision=revision,
        cache_dir=cache_dir,
    )
    payload = _load_adapter_payload(adapter_checkpoint)
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


def add_ovary_adapter(cellvit, adapter_checkpoint="ovary_5", **kwargs):
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
