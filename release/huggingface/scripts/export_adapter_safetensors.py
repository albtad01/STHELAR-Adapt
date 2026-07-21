#!/usr/bin/env python3
"""Safely convert a CellViT adapter-only .pth file to safetensors.

The source checkpoint is loaded on CPU. PyTorch's ``weights_only`` loader is
used when the installed PyTorch exposes it. Output is a new directory
containing ``adapter_model.safetensors``, ``adapter_config.json``, and
``checksums.json``; existing output files are never overwritten.
"""

from __future__ import annotations

import argparse
import codecs
import hashlib
import inspect
import json
import math
import re
from pathlib import Path
from typing import Any

import torch
import numpy as np
from safetensors.torch import load_file, save_file


ALLOWED_TOP_LEVEL_KEYS = {
    "format",
    "format_version",
    "metadata",
    "adapter_state_dict",
    "mutable_buffer_state_dict",
}
SECTION_NAMES = ("adapter_state_dict", "mutable_buffer_state_dict")
COMPONENT_PATTERNS = {
    "lora": ("adapter_q_down", "adapter_q_up", "adapter_v_down", "adapter_v_up"),
    "adaptformer": (".mlp.adapter_",),
    "np_head": ("nuclei_binary_map_decoder.decoder0_header",),
    "hv_head": ("hv_map_decoder.decoder0_header",),
    "nt_head": ("nuclei_type_maps_decoder.decoder0_header",),
}
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


def load_pth_safely(path: Path) -> dict[str, Any]:
    kwargs: dict[str, Any] = {"map_location": "cpu"}
    if "weights_only" in inspect.signature(torch.load).parameters:
        kwargs["weights_only"] = True
    if kwargs.get("weights_only") and hasattr(torch.serialization, "safe_globals"):
        numpy_globals = [
            np.core.multiarray.scalar,
            np.dtype,
            type(np.dtype(np.float64)),
            codecs.encode,
        ]
        with torch.serialization.safe_globals(numpy_globals):
            payload = torch.load(str(path), **kwargs)
    else:
        payload = torch.load(str(path), **kwargs)
    if not isinstance(payload, dict):
        raise TypeError("Checkpoint top level must be a dictionary")
    unexpected = set(payload) - ALLOWED_TOP_LEVEL_KEYS
    missing = ALLOWED_TOP_LEVEL_KEYS - set(payload)
    if unexpected or missing:
        raise ValueError(
            f"Unexpected checkpoint schema; missing={sorted(missing)}, "
            f"unexpected={sorted(unexpected)}"
        )
    if payload["format"] != "cellvit_adapter_checkpoint":
        raise ValueError(f"Unexpected format: {payload['format']!r}")
    if not isinstance(payload["metadata"], dict):
        raise TypeError("metadata must be a dictionary")
    return payload


def collect_tensors(payload: dict[str, Any]) -> tuple[dict[str, torch.Tensor], dict[str, Any]]:
    tensors: dict[str, torch.Tensor] = {}
    shapes: dict[str, list[int]] = {}
    section_counts: dict[str, int] = {}
    for section in SECTION_NAMES:
        state = payload[section]
        if not isinstance(state, dict):
            raise TypeError(f"{section} must be a dictionary")
        section_counts[section] = len(state)
        for key, value in state.items():
            if not isinstance(key, str):
                raise TypeError(f"Non-string key in {section}: {key!r}")
            if not torch.is_tensor(value):
                raise TypeError(
                    f"Non-tensor value in {section}.{key}: {type(value).__name__}; "
                    "move auxiliary values to metadata so they can be exported as JSON"
                )
            flat_key = f"{section}.{key}"
            if flat_key in tensors:
                raise ValueError(f"Duplicate flattened tensor key: {flat_key}")
            tensor = value.detach().cpu().contiguous()
            tensors[flat_key] = tensor
            shapes[flat_key] = list(tensor.shape)
    if not payload["adapter_state_dict"]:
        raise ValueError("adapter_state_dict is empty")
    return tensors, {"expected_tensor_shapes": shapes, "section_counts": section_counts}


def require_components(tensors: dict[str, torch.Tensor], expected: list[str]) -> None:
    keys = tuple(tensors)
    for component in expected:
        patterns = COMPONENT_PATTERNS[component]
        if not any(any(pattern in key for pattern in patterns) for key in keys):
            raise ValueError(f"Missing expected component keys for {component}")


def ensure_json(value: Any) -> None:
    try:
        json.dumps(value, allow_nan=False)
    except (TypeError, ValueError) as exc:
        raise TypeError(
            "Checkpoint metadata is not strict JSON; normalize it before release"
        ) from exc


def sanitize_metadata(value: Any, key: str = "") -> Any:
    """Make metadata public/JSON-safe without retaining local absolute paths."""
    private_path_keys = {
        "base_checkpoint_resolved",
        "original_run_dir",
        "original_config_path",
        "config_path",
        "source_checkpoint",
    }
    if key in private_path_keys:
        if value in (None, ""):
            return None
        return Path(str(value)).name
    if isinstance(value, dict):
        return {str(item_key): sanitize_metadata(item, str(item_key)) for item_key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [sanitize_metadata(item) for item in value]
    if torch.is_tensor(value):
        raise TypeError(f"Tensor found in metadata at {key!r}; tensors belong in a state dictionary")
    if hasattr(value, "item") and callable(value.item):
        value = value.item()
    if isinstance(value, float) and not math.isfinite(value):
        return None
    if isinstance(value, Path):
        value = str(value)
    if isinstance(value, str) and Path(value).is_absolute():
        return Path(value).name
    return value


def reject_private_metadata(value: Any, location: str = "metadata") -> None:
    """Reject paths/usernames that must not enter a public adapter package."""
    if isinstance(value, dict):
        for key, item in value.items():
            reject_private_metadata(item, f"{location}.{key}")
    elif isinstance(value, list):
        for index, item in enumerate(value):
            reject_private_metadata(item, f"{location}[{index}]")
    elif isinstance(value, str):
        if Path(value).is_absolute() or re.match(r"^[A-Za-z]:[\\/]", value):
            raise ValueError(f"Absolute path remains in public metadata at {location}")
        if any(pattern.search(value) for pattern in PRIVATE_METADATA_PATTERNS):
            raise ValueError(f"Private path/username remains in public metadata at {location}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path, help="Adapter-only .pth checkpoint")
    parser.add_argument("output_dir", type=Path, help="New or empty release directory")
    parser.add_argument("--adapter-id", required=True, help="Stable public adapter ID")
    parser.add_argument(
        "--source-config",
        required=True,
        type=Path,
        help="Exact curated YAML config associated with this adapter",
    )
    parser.add_argument(
        "--base-checkpoint-sha256",
        required=True,
        help="Verified SHA256 of the required base checkpoint",
    )
    parser.add_argument(
        "--expected-components",
        nargs="+",
        choices=sorted(COMPONENT_PATTERNS),
        default=["lora", "adaptformer", "np_head", "hv_head", "nt_head"],
        help="Components which must be present (defaults to selected STHELAR-Adapt method)",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    source = args.source.expanduser().resolve()
    source_config = args.source_config.expanduser().resolve()
    output_dir = args.output_dir.expanduser().resolve()
    if not source.is_file() or source.suffix.lower() != ".pth":
        raise FileNotFoundError(f"Expected a .pth source file: {source}")
    if not source_config.is_file() or source_config.suffix.lower() not in {".yaml", ".yml"}:
        raise FileNotFoundError(f"Expected a YAML source config: {source_config}")
    if not re.fullmatch(r"[a-z0-9][a-z0-9_/-]*", args.adapter_id):
        raise ValueError(f"Invalid public adapter ID: {args.adapter_id!r}")
    if not re.fullmatch(r"[0-9a-f]{64}", args.base_checkpoint_sha256):
        raise ValueError("--base-checkpoint-sha256 must be 64 lowercase hex characters")
    output_dir.mkdir(parents=True, exist_ok=True)
    weights_path = output_dir / "adapter_model.safetensors"
    config_path = output_dir / "adapter_config.json"
    checksums_path = output_dir / "checksums.json"
    collisions = [path for path in (weights_path, config_path, checksums_path) if path.exists()]
    if collisions:
        raise FileExistsError(f"Refusing to overwrite: {[str(path) for path in collisions]}")

    payload = load_pth_safely(source)
    tensors, structural = collect_tensors(payload)
    require_components(tensors, args.expected_components)
    metadata = sanitize_metadata(dict(payload["metadata"]))
    metadata.update(
        {
            "format": payload["format"],
            "format_version": payload["format_version"],
            "adapter_id": args.adapter_id,
            "adapter_weight_file": "adapter_model.safetensors",
            "source_filename": source.name,
            "source_config_filename": source_config.name,
            "source_config_sha256": sha256(source_config),
            "base_checkpoint_sha256": args.base_checkpoint_sha256,
            "expected_components": args.expected_components,
            **structural,
        }
    )
    ensure_json(metadata)
    reject_private_metadata(metadata)

    save_file(
        tensors,
        str(weights_path),
        metadata={
            "format": payload["format"],
            "format_version": str(payload["format_version"]),
            "metadata_json": json.dumps(metadata, sort_keys=True),
        },
    )
    reloaded = load_file(str(weights_path), device="cpu")
    if set(reloaded) != set(tensors):
        raise RuntimeError("Round-trip key mismatch after safetensors save")
    for key, original in tensors.items():
        if not torch.equal(original, reloaded[key]):
            raise RuntimeError(f"Round-trip tensor inequality: {key}")

    config_path.write_text(json.dumps(metadata, indent=2, sort_keys=True) + "\n")
    checksums = {
        "source": {"path": source.name, "sha256": sha256(source)},
        "source_config": {
            "path": source_config.name,
            "sha256": sha256(source_config),
        },
        "exported": {
            "adapter_model.safetensors": sha256(weights_path),
            "adapter_config.json": sha256(config_path),
        },
    }
    checksums_path.write_text(json.dumps(checksums, indent=2, sort_keys=True) + "\n")
    print(json.dumps(checksums, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
