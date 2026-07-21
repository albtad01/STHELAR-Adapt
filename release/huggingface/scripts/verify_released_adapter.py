#!/usr/bin/env python3
"""Verify a released STHELAR-Adapt safetensors package on CPU."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
from pathlib import Path

import torch
from safetensors.torch import load_file

SEARCH_ROOTS = [Path.cwd(), *Path(__file__).resolve().parents]
REPO_ROOT = next(
    (root for root in SEARCH_ROOTS if (root / "utils" / "adapter_checkpoint.py").is_file()),
    Path.cwd(),
)
sys.path.insert(0, str(REPO_ROOT))

COMPONENT_PATTERNS = {
    "lora": ("adapter_q_down", "adapter_q_up", "adapter_v_down", "adapter_v_up"),
    "adaptformer": (".mlp.adapter_",),
    "np_head": ("nuclei_binary_map_decoder.decoder0_header",),
    "hv_head": ("hv_map_decoder.decoder0_header",),
    "nt_head": ("nuclei_type_maps_decoder.decoder0_header",),
}
EXPECTED_BASE_MODEL = "CellViT-SAM-H-x40"
REQUIRED_COMPONENTS = {"lora", "adaptformer", "np_head", "hv_head", "nt_head"}
PRIVATE_METADATA_PATTERNS = (
    re.compile(r"(?:^|[\\/])gpfs[\\/]", re.IGNORECASE),
    re.compile(r"(?:^|[\\/])home[\\/]", re.IGNORECASE),
    re.compile(r"(?:^|[\\/])Users[\\/]"),
    re.compile(r"(?:^|[\\/])Volumes[\\/]"),
    re.compile(r"taddeial", re.IGNORECASE),
    re.compile(r"ruche", re.IGNORECASE),
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def validate_mapping(config: dict) -> None:
    mapping = config.get("label_mapping") or config.get("nuclei_types")
    if not isinstance(mapping, dict) or not mapping:
        raise ValueError("Missing label_mapping/nuclei_types")
    normalized = {str(key): int(value) for key, value in mapping.items()}
    expected_names = {"Background", "Immune", "Stromal", "Epithelial", "Melanocyte", "Other"}
    if set(normalized) != expected_names:
        raise ValueError(f"Unexpected five-class mapping names: {sorted(normalized)}")
    if normalized["Background"] != 0 or sorted(normalized.values()) != list(range(6)):
        raise ValueError("Label ids must be contiguous 0..5 with Background=0")
    if int(config.get("num_nuclei_classes", -1)) != len(normalized):
        raise ValueError("num_nuclei_classes does not match label mapping")


def reject_private_metadata(value, location="metadata") -> None:
    if isinstance(value, dict):
        for key, item in value.items():
            reject_private_metadata(item, f"{location}.{key}")
    elif isinstance(value, list):
        for index, item in enumerate(value):
            reject_private_metadata(item, f"{location}[{index}]")
    elif isinstance(value, str):
        if Path(value).is_absolute() or re.match(r"^[A-Za-z]:[\\/]", value):
            raise ValueError(f"Absolute path in released metadata at {location}")
        if any(pattern.search(value) for pattern in PRIVATE_METADATA_PATTERNS):
            raise ValueError(f"Private path/username in released metadata at {location}")


def load_into_model(config: dict, tensors: dict[str, torch.Tensor], base: Path, smoke: bool) -> dict:
    from utils.adapter_checkpoint import build_model, load_adapter_state

    if not base.is_file():
        raise FileNotFoundError(base)
    training_config = {
        "random_seed": int(config.get("random_seed", 42)),
        "data": {
            "num_nuclei_classes": int(config["num_nuclei_classes"]),
            "num_tissue_classes": int(config["num_tissue_classes"]),
        },
        "model": {"backbone": "SAM-H", "shared_decoders": False},
        "training": {"drop_rate": 0, "regression_loss": False},
        "adapters": {
            "adapter_type": config["adapter_type"],
            "decoder_train_scope": config["decoder_train_scope"],
            "lora": config["lora"],
            "adaptformer": config["adaptformer"],
        },
    }
    model, load_info, _ = build_model(training_config, base)
    sections = {"adapter_state_dict": {}, "mutable_buffer_state_dict": {}}
    for flat_key, tensor in tensors.items():
        section, key = flat_key.split(".", 1)
        if section not in sections:
            raise ValueError(f"Unexpected safetensors section: {section}")
        sections[section][key] = tensor
    expected_parameter_keys = {
        name for name, parameter in model.named_parameters() if parameter.requires_grad
    }
    actual_parameter_keys = set(sections["adapter_state_dict"])
    missing_adapter_keys = sorted(expected_parameter_keys - actual_parameter_keys)
    unexpected_adapter_keys = sorted(actual_parameter_keys - expected_parameter_keys)
    declared_buffer_keys = set(config.get("mutable_buffer_names", []))
    actual_buffer_keys = set(sections["mutable_buffer_state_dict"])
    missing_buffer_keys = sorted(declared_buffer_keys - actual_buffer_keys)
    unexpected_buffer_keys = sorted(actual_buffer_keys - declared_buffer_keys)
    if missing_adapter_keys or unexpected_adapter_keys:
        raise RuntimeError(
            "Adapter parameter key mismatch: "
            f"missing={missing_adapter_keys[:10]}, unexpected={unexpected_adapter_keys[:10]}"
        )
    if missing_buffer_keys or unexpected_buffer_keys:
        raise RuntimeError(
            "Mutable-buffer key mismatch: "
            f"missing={missing_buffer_keys[:10]}, unexpected={unexpected_buffer_keys[:10]}"
        )
    loaded = load_adapter_state(model, sections)
    if len(loaded) != len(tensors):
        raise RuntimeError(f"Model state load count mismatch: {len(loaded)} != {len(tensors)}")
    output_shapes = {}
    if smoke:
        model.eval()
        with torch.inference_mode():
            output = model(torch.zeros(1, 3, 256, 256))
        if not isinstance(output, dict):
            raise RuntimeError(f"Forward smoke test returned {type(output).__name__}, expected dict")
        output_shapes = {
            key: list(value.shape) for key, value in output.items() if torch.is_tensor(value)
        }
        expected_output_shapes = {
            "tissue_types": [1, int(config["num_tissue_classes"])],
            "nuclei_binary_map": [1, 2, 256, 256],
            "hv_map": [1, 2, 256, 256],
            "nuclei_type_map": [1, int(config["num_nuclei_classes"]), 256, 256],
        }
        if output_shapes != expected_output_shapes:
            raise RuntimeError(
                f"Forward output mismatch: {output_shapes} != {expected_output_shapes}"
            )
    return {
        "base_load_info": load_info,
        "missing_adapter_keys": missing_adapter_keys,
        "unexpected_adapter_keys": unexpected_adapter_keys,
        "missing_mutable_buffer_keys": missing_buffer_keys,
        "unexpected_mutable_buffer_keys": unexpected_buffer_keys,
        "loaded_state_key_count": len(loaded),
        "output_tensor_shapes": output_shapes,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("release_dir", type=Path)
    parser.add_argument(
        "--base-checkpoint",
        type=Path,
        help="Optional local CellViT-SAM-H-x40 checkpoint; enables model loading and forward smoke test",
    )
    parser.add_argument(
        "--skip-forward",
        action="store_true",
        help="With --base-checkpoint, verify model loading but skip the 256x256 CPU forward pass",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    release_dir = args.release_dir.expanduser().resolve()
    weights_path = release_dir / "adapter_model.safetensors"
    config_path = release_dir / "adapter_config.json"
    checksums_path = release_dir / "checksums.json"
    for path in (weights_path, config_path, checksums_path):
        if not path.is_file():
            raise FileNotFoundError(path)
    config = json.loads(config_path.read_text())
    checksums = json.loads(checksums_path.read_text())
    tensors = load_file(str(weights_path), device="cpu")
    reject_private_metadata(config)

    if config.get("base_model") != EXPECTED_BASE_MODEL:
        raise ValueError(f"Unsupported base model: {config.get('base_model')!r}")
    declared_checkpoint = Path(str(config.get("base_checkpoint", ""))).name
    if declared_checkpoint != "CellViT-SAM-H-x40.pth":
        raise ValueError(f"Unexpected declared base checkpoint: {declared_checkpoint!r}")
    validate_mapping(config)
    expected_shapes = config.get("expected_tensor_shapes")
    if not isinstance(expected_shapes, dict) or set(expected_shapes) != set(tensors):
        raise ValueError("Tensor keys do not exactly match expected_tensor_shapes")
    for key, tensor in tensors.items():
        if list(tensor.shape) != expected_shapes[key]:
            raise ValueError(
                f"Shape mismatch for {key}: {list(tensor.shape)} != {expected_shapes[key]}"
            )
    expected_components = config.get("expected_components", [])
    if set(expected_components) != REQUIRED_COMPONENTS:
        raise ValueError(
            f"Expected exactly the selected-method components: {sorted(REQUIRED_COMPONENTS)}"
        )
    for component in expected_components:
        patterns = COMPONENT_PATTERNS[component]
        if not any(any(pattern in key for pattern in patterns) for key in tensors):
            raise ValueError(f"Missing component: {component}")

    adapter_numel = sum(
        tensor.numel()
        for key, tensor in tensors.items()
        if key.startswith("adapter_state_dict.")
    )
    declared_numel = int(config.get("trainable_parameter_count", config.get("trainable_params", -1)))
    if adapter_numel != declared_numel:
        raise ValueError(f"Trainable parameter count mismatch: {adapter_numel} != {declared_numel}")
    expected_sha = checksums["exported"]["adapter_model.safetensors"]
    if sha256(weights_path) != expected_sha:
        raise ValueError("Safetensors SHA256 mismatch")
    expected_config_sha = checksums["exported"]["adapter_config.json"]
    if sha256(config_path) != expected_config_sha:
        raise ValueError("adapter_config.json SHA256 mismatch")
    model_verification = {
        "missing_adapter_keys": None,
        "unexpected_adapter_keys": None,
        "missing_mutable_buffer_keys": None,
        "unexpected_mutable_buffer_keys": None,
        "output_tensor_shapes": {},
    }
    if args.base_checkpoint is not None:
        expected_base_sha = config.get("base_checkpoint_sha256")
        if expected_base_sha and sha256(args.base_checkpoint.expanduser().resolve()) != expected_base_sha:
            raise ValueError("Base checkpoint SHA256 mismatch")
        model_verification = load_into_model(
            config, tensors, args.base_checkpoint.expanduser().resolve(), not args.skip_forward
        )
    print(
        json.dumps(
            {
                "status": "ok",
                "tensor_count": len(tensors),
                "trainable_parameter_count": adapter_numel,
                "base_model": config["base_model"],
                "model_load_tested": args.base_checkpoint is not None,
                "forward_tested": args.base_checkpoint is not None and not args.skip_forward,
                **model_verification,
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
