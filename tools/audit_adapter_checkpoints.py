#!/usr/bin/env python3
"""Inventory PyTorch ZIP checkpoints without importing or executing PyTorch.

This helper uses a restricted unpickler which accepts only the small set of
globals emitted by ``torch.save`` for this repository's adapter checkpoints.
It reconstructs tensor descriptors (dtype and shape), never tensor contents.
Use the PyTorch-based release verifier for cryptographic tensor validation.
"""

import argparse
import codecs
import collections
import csv
import hashlib
import io
import json
import pickle
import re
import zipfile
from pathlib import Path


class StorageType:
    def __init__(self, dtype):
        self.dtype = dtype


class Storage:
    def __init__(self, dtype, key, size):
        self.dtype = dtype
        self.key = key
        self.size = size


class TensorDescriptor:
    def __init__(self, dtype, shape, storage_key):
        self.dtype = dtype
        self.shape = shape
        self.storage_key = storage_key

    @property
    def numel(self):
        result = 1
        for dim in self.shape:
            result *= dim
        return result


class NumpyDtypeDescriptor:
    def __init__(self, *args):
        self.args = args
        self.state = None

    def __setstate__(self, state):
        self.state = state


def _rebuild_tensor(storage, _offset, shape, _stride, _requires_grad, _hooks):
    return TensorDescriptor(storage.dtype, tuple(int(v) for v in shape), storage.key)


def _numpy_scalar(*_args):
    # Metrics are not needed for the structural inventory. Avoid interpreting
    # NumPy byte payloads in the metadata-only audit.
    return "<numpy-scalar>"


class RestrictedTorchUnpickler(pickle.Unpickler):
    ALLOWED = {
        ("torch._utils", "_rebuild_tensor_v2"): _rebuild_tensor,
        ("torch", "FloatStorage"): StorageType("float32"),
        ("torch", "DoubleStorage"): StorageType("float64"),
        ("torch", "HalfStorage"): StorageType("float16"),
        ("torch", "BFloat16Storage"): StorageType("bfloat16"),
        ("torch", "LongStorage"): StorageType("int64"),
        ("torch", "IntStorage"): StorageType("int32"),
        ("torch", "ShortStorage"): StorageType("int16"),
        ("torch", "CharStorage"): StorageType("int8"),
        ("torch", "ByteStorage"): StorageType("uint8"),
        ("torch", "BoolStorage"): StorageType("bool"),
        ("collections", "OrderedDict"): collections.OrderedDict,
        ("numpy.core.multiarray", "scalar"): _numpy_scalar,
        ("numpy", "dtype"): NumpyDtypeDescriptor,
        ("_codecs", "encode"): codecs.encode,
    }

    def find_class(self, module, name):
        try:
            return self.ALLOWED[(module, name)]
        except KeyError as exc:
            raise pickle.UnpicklingError(
                f"Blocked unexpected checkpoint global: {module}.{name}"
            ) from exc

    def persistent_load(self, persistent_id):
        if not isinstance(persistent_id, tuple) or persistent_id[0] != "storage":
            raise pickle.UnpicklingError(
                f"Blocked unexpected persistent id: {persistent_id!r}"
            )
        _, storage_type, key, _location, size = persistent_id
        if not isinstance(storage_type, StorageType):
            raise pickle.UnpicklingError("Unexpected storage type")
        return Storage(storage_type.dtype, str(key), int(size))


def load_structure(path):
    if not zipfile.is_zipfile(path):
        raise ValueError("not a PyTorch ZIP checkpoint")
    with zipfile.ZipFile(path) as archive:
        candidates = [name for name in archive.namelist() if name.endswith("/data.pkl")]
        if len(candidates) != 1:
            raise ValueError(f"expected one data.pkl, found {len(candidates)}")
        data = archive.read(candidates[0])
    return RestrictedTorchUnpickler(io.BytesIO(data)).load()


def tensor_content_sha256(path, tensors):
    """Hash state keys, descriptors, and raw tensor storage independent of ZIP prefix."""
    digest = hashlib.sha256()
    with zipfile.ZipFile(path) as archive:
        names = archive.namelist()
        prefix = next(name[:-len("data.pkl")] for name in names if name.endswith("/data.pkl"))
        for key, tensor in sorted(tensors.items()):
            digest.update(key.encode("utf-8"))
            digest.update(tensor.dtype.encode("ascii"))
            digest.update(repr(tensor.shape).encode("ascii"))
            digest.update(archive.read(prefix + "data/" + tensor.storage_key))
    return digest.hexdigest()


def sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def infer_tissue(name, metadata):
    tissue_types = metadata.get("tissue_types") or {}
    if isinstance(tissue_types, dict) and tissue_types:
        return "+".join(str(key) for key in tissue_types)
    for tissue in (
        "kidney_liver_tonsil", "pancreatic", "breast", "kidney", "liver",
        "ovary", "colon", "lung", "skin", "tonsil", "bps",
    ):
        if tissue in name.lower():
            return tissue.replace("_", "+")
    return "unknown"


def match_config(repo, metadata, filename):
    run_name = str(metadata.get("run_name") or metadata.get("original_run_name") or "")
    config_root = repo / "configs" / "release"
    candidates = list(config_root.rglob("*.yaml")) if config_root.is_dir() else []
    for candidate in candidates:
        stem = candidate.stem
        if stem.startswith("training_"):
            stem = stem[len("training_"):]
        if stem == run_name:
            return candidate.relative_to(repo).as_posix()
    normalized = filename[:-len("_adapter.pth")] if filename.endswith("_adapter.pth") else filename
    for candidate in candidates:
        stem = candidate.stem
        if stem.startswith("training_"):
            stem = stem[len("training_"):]
        if stem == normalized:
            return candidate.relative_to(repo).as_posix()
    return ""


def is_reported_selected(tissue, seed, metadata):
    if metadata.get("adapter_type") != "lora_adaptformer":
        return False
    if metadata.get("decoder_train_scope") != "heads_only":
        return False
    selected = {
        "KidneyLiverTonsil": {"42", "43"},
        "Liver": {"42"},
        "Kidney": {"42", "43"},
        "Ovary": {"42"},
        "Breast": {"42"},
        "Colon": {"42"},
        "Lung": {"42"},
        "Pancreatic": {"43"},
        "Skin": {"42"},
        "Tonsil": {"43"},
    }
    return seed in selected.get(tissue, set())


def component_flags(keys):
    return {
        "lora": any("adapter_q_" in key or "adapter_v_" in key for key in keys),
        "adaptformer": any(".mlp.adapter_" in key for key in keys),
        "np_head": any("nuclei_binary_map_decoder.decoder0_header" in key for key in keys),
        "hv_head": any("hv_map_decoder.decoder0_header" in key for key in keys),
        "nt_head": any("nuclei_type_maps_decoder.decoder0_header" in key for key in keys),
        "decoder_modules": any(
            ("_decoder." in key and "decoder0_header" not in key)
            or "decoder_adapter" in key
            for key in keys
        ),
    }


def inspect(path, repo):
    payload = load_structure(path)
    if not isinstance(payload, dict):
        raise ValueError("top-level object is not a dictionary")
    metadata = payload.get("metadata") if isinstance(payload.get("metadata"), dict) else {}
    tensor_sections = []
    if isinstance(payload.get("adapter_state_dict"), dict):
        tensor_sections.append(payload["adapter_state_dict"])
    if isinstance(payload.get("mutable_buffer_state_dict"), dict):
        tensor_sections.append(payload["mutable_buffer_state_dict"])
    if not tensor_sections and all(isinstance(k, str) for k in payload):
        tensor_sections = [payload]
    state = {}
    for section in tensor_sections:
        state.update(section)
    tensors = {k: v for k, v in state.items() if isinstance(v, TensorDescriptor)}
    adapter_tensors = {
        k: v
        for k, v in (payload.get("adapter_state_dict") or {}).items()
        if isinstance(v, TensorDescriptor)
    }
    mutable_tensors = {
        k: v
        for k, v in (payload.get("mutable_buffer_state_dict") or {}).items()
        if isinstance(v, TensorDescriptor)
    }
    non_tensors = sorted(k for k, v in state.items() if not isinstance(v, TensorDescriptor))
    dtypes = collections.Counter(tensor.dtype for tensor in tensors.values())
    keys = sorted(tensors)
    flags = component_flags(keys)
    digest = sha256(path)
    name = path.name
    seed_match = re.search(r"seed(\d+)", name, re.IGNORECASE)
    seed = seed_match.group(1) if seed_match else str(metadata.get("random_seed", ""))
    tissue = infer_tissue(name, metadata)
    tissue_title = tissue.replace("+", "") if tissue == "Kidney+Liver+Tonsil" else tissue.title()
    # Metadata is authoritative for KLT's concatenated tissue name.
    if "kidney_liver_tonsil" in name.lower():
        tissue_title = "KidneyLiverTonsil"
    reported = is_reported_selected(tissue_title, seed, metadata)
    # Two same-named seed-42 reruns exist for Colon and Lung. The paper table
    # traces to Colon 2026-06-30T092331 (nested copy, test mPQ 0.204747) and
    # Lung 2026-07-01T025800 (root copy, test mPQ 0.301539), respectively.
    is_root_copy = path.parent.resolve() == (repo / "adapters").resolve()
    if tissue_title == "Colon" and seed == "42":
        reported = reported and not is_root_copy
    if tissue_title == "Lung" and seed == "42":
        reported = reported and is_root_copy
    complete = (
        payload.get("format") == "cellvit_adapter_checkpoint"
        and bool(tensors)
        and not non_tensors
        and all(flags[k] for k in ("lora", "adaptformer", "np_head", "hv_head", "nt_head"))
        if metadata.get("adapter_type") == "lora_adaptformer"
        and metadata.get("decoder_train_scope") == "heads_only"
        else payload.get("format") == "cellvit_adapter_checkpoint" and bool(tensors) and not non_tensors
    )
    action = "release" if reported and complete else "archive locally"
    if not complete or "retest" in name.lower():
        action = "investigate"
    return {
        "original_path": path.relative_to(repo).as_posix(),
        "filename": name,
        "file_size_bytes": str(path.stat().st_size),
        "sha256": digest,
        "tensor_content_sha256": tensor_content_sha256(path, tensors),
        "top_level_checkpoint_keys": ";".join(str(k) for k in payload),
        "state_dict_key_count": str(len(adapter_tensors) if adapter_tensors else len(tensors)),
        "tensor_parameter_count": str(
            sum(t.numel for t in adapter_tensors.values())
            if adapter_tensors
            else sum(t.numel for t in tensors.values())
        ),
        "mutable_buffer_key_count": str(len(mutable_tensors)),
        "mutable_buffer_tensor_element_count": str(sum(t.numel for t in mutable_tensors.values())),
        "checkpoint_tensor_element_count": str(sum(t.numel for t in tensors.values())),
        "dtype_distribution": json.dumps(dict(sorted(dtypes.items())), separators=(",", ":")),
        "lora": str(flags["lora"]).lower(),
        "adaptformer": str(flags["adaptformer"]).lower(),
        "np_head": str(flags["np_head"]).lower(),
        "hv_head": str(flags["hv_head"]).lower(),
        "nt_head": str(flags["nt_head"]).lower(),
        "decoder_modules": str(flags["decoder_modules"]).lower(),
        "likely_tissue": tissue,
        "seed": seed,
        "configuration_method": f"{metadata.get('adapter_type', 'unknown')}:{metadata.get('decoder_train_scope', 'unknown')}",
        "matching_yaml_config": match_config(repo, metadata, name),
        "appears_complete": str(bool(complete)).lower(),
        "exact_duplicate_of": "",
        "reported_in_paper": str(bool(reported)).lower(),
        "recommended_action": action,
        "audit_note": "" if not non_tensors else "non-tensor state keys: " + ";".join(non_tensors),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("adapter_dir", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--repo", type=Path, default=Path.cwd())
    args = parser.parse_args()
    repo = args.repo.resolve()
    paths = sorted(args.adapter_dir.resolve().rglob("*.pth"))
    if not paths:
        raise SystemExit(f"No .pth files found under {args.adapter_dir}")
    rows = []
    errors = []
    for path in paths:
        try:
            rows.append(inspect(path, repo))
        except Exception as exc:  # preserve the row and fail visibly in the manifest
            errors.append((path, exc))
            rows.append({
                "original_path": path.relative_to(repo).as_posix(),
                "filename": path.name,
                "file_size_bytes": str(path.stat().st_size),
                "sha256": sha256(path),
                "appears_complete": "false",
                "recommended_action": "investigate",
                "audit_note": f"STRUCTURAL AUDIT FAILED: {type(exc).__name__}: {exc}",
            })
    by_hash = collections.defaultdict(list)
    for row in rows:
        by_hash[row.get("tensor_content_sha256") or row["sha256"]].append(row)
    for duplicates in by_hash.values():
        if len(duplicates) > 1:
            canonical = duplicates[0]["original_path"]
            for row in duplicates[1:]:
                row["exact_duplicate_of"] = canonical
                if row.get("recommended_action") == "release":
                    row["recommended_action"] = "archive locally"
                    row["audit_note"] = "exact duplicate; release canonical path only"
    fields = [
        "original_path", "filename", "file_size_bytes", "sha256", "tensor_content_sha256",
        "top_level_checkpoint_keys", "state_dict_key_count", "tensor_parameter_count",
        "mutable_buffer_key_count", "mutable_buffer_tensor_element_count",
        "checkpoint_tensor_element_count",
        "dtype_distribution", "lora", "adaptformer", "np_head", "hv_head", "nt_head",
        "decoder_modules", "likely_tissue", "seed", "configuration_method",
        "matching_yaml_config", "appears_complete", "exact_duplicate_of",
        "reported_in_paper", "recommended_action", "audit_note",
    ]
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
    print(f"Wrote {len(rows)} rows to {args.output}; structural failures={len(errors)}")
    if errors:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
