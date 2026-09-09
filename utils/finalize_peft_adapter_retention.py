#!/usr/bin/env python3
"""Delete a redundant PEFT checkpoint only after exact adapter verification."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from datetime import datetime, timezone
from pathlib import Path


REPO = Path(__file__).resolve().parents[1]
REQUIRED_RUN_FILES = (
    "config.yaml",
    "inference_results.json",
    "efficiency_metrics.json",
    "inference_efficiency_metrics.json",
    "checkpoint_retention_metadata.json",
    "logs.log",
)
REQUIRED_OUTPUTS = {
    "hv_map",
    "nuclei_binary_map",
    "nuclei_type_map",
    "tissue_types",
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def repo_path(value: str) -> Path:
    path = Path(value)
    return path.resolve() if path.is_absolute() else (REPO / path).resolve()


def atomic_json(path: Path, payload: dict) -> None:
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--adapter-dir", type=Path, required=True)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    run_dir = args.run_dir.expanduser().resolve()
    adapter_dir = args.adapter_dir.expanduser().resolve()
    checkpoint = run_dir / "checkpoints" / "checkpoint_10.pth"
    verification_path = adapter_dir / "verification.json"
    adapter_path = adapter_dir / "adapter_model.safetensors"
    for path in (checkpoint, verification_path, adapter_path):
        if not path.is_file():
            raise FileNotFoundError(path)
    missing = [name for name in REQUIRED_RUN_FILES if not (run_dir / name).is_file()]
    if missing:
        raise RuntimeError(f"Missing preserved run provenance: {missing}")
    for name in (
        "inference_results.json",
        "efficiency_metrics.json",
        "inference_efficiency_metrics.json",
        "checkpoint_retention_metadata.json",
    ):
        json.loads((run_dir / name).read_text())

    verification = json.loads(verification_path.read_text())
    outputs = verification.get("forward_output_comparison", [])
    if verification.get("status") != "ok":
        raise RuntimeError("Adapter verification status is not ok")
    if verification.get("state_reconstruction") != "exact_all_tensors":
        raise RuntimeError("Adapter state reconstruction is not exact")
    if verification.get("forward_verification") != "exact_all_output_tensors":
        raise RuntimeError("Adapter forward verification is not exact")
    if {row.get("name") for row in outputs} != REQUIRED_OUTPUTS:
        raise RuntimeError("Adapter verification does not cover all required outputs")
    if not all(row.get("exact") is True and row.get("max_abs_difference") == 0.0 for row in outputs):
        raise RuntimeError("At least one adapter output is not bitwise exact")
    if repo_path(str(verification["canonical_checkpoint"])) != checkpoint.resolve():
        raise RuntimeError("Verification record points to a different checkpoint")

    checkpoint_sha = sha256(checkpoint)
    adapter_sha = sha256(adapter_path)
    if checkpoint_sha != verification.get("canonical_checkpoint_sha256"):
        raise RuntimeError("Canonical checkpoint SHA256 mismatch")
    if adapter_sha != verification.get("adapter_sha256"):
        raise RuntimeError("Adapter SHA256 mismatch")

    metadata_path = run_dir / "adapter_checkpoint_retention_metadata.json"
    metadata = {
        "schema_version": 1,
        "status": "validated_dry_run" if args.dry_run else "verified_pending_checkpoint_deletion",
        "run_dir": str(run_dir.relative_to(REPO)),
        "canonical_checkpoint": str(checkpoint.relative_to(REPO)),
        "canonical_checkpoint_bytes": checkpoint.stat().st_size,
        "canonical_checkpoint_sha256": checkpoint_sha,
        "adapter_path": str(adapter_path.relative_to(REPO)),
        "adapter_bytes": adapter_path.stat().st_size,
        "adapter_sha256": adapter_sha,
        "state_reconstruction": verification["state_reconstruction"],
        "forward_verification": verification["forward_verification"],
        "required_outputs": sorted(REQUIRED_OUTPUTS),
        "checkpoint_exists_after_retention": True,
        "validation_timestamp": datetime.now(timezone.utc).isoformat(),
    }
    if args.dry_run:
        print(json.dumps(metadata, indent=2, sort_keys=True))
        return

    atomic_json(metadata_path, metadata)
    checkpoint.unlink()
    metadata.update(
        {
            "status": "adapter_verified_checkpoint_deleted",
            "checkpoint_exists_after_retention": False,
            "deletion_timestamp": datetime.now(timezone.utc).isoformat(),
        }
    )
    atomic_json(metadata_path, metadata)
    print(json.dumps(metadata, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
