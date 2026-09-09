#!/usr/bin/env python3
"""Build the authoritative verified slide-independent adapter manifest."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path


REPO = Path(__file__).resolve().parents[1]
ADAPTER_ROOT = REPO / "release" / "huggingface" / "adapters"
PRESERVED_RUN_FILES = (
    "config.yaml",
    "inference_results.json",
    "efficiency_metrics.json",
    "inference_efficiency_metrics.json",
    "checkpoint_retention_metadata.json",
)


def expected_ids() -> set[str]:
    ids = {
        f"cellvit-sam-h-x40/klt/slideind/fold{fold}/seed{seed}"
        for fold in ("a", "b")
        for seed in (42, 43)
    }
    ids.update(
        f"cellvit-256-x40/klt/slideind/fold{fold}/seed42"
        for fold in ("a", "b")
    )
    for tissue in (
        "breast", "colon", "kidney", "liver", "lung", "ovary",
        "pancreatic", "skin", "tonsil",
    ):
        ids.add(
            f"cellvit-sam-h-x40/tissue_specific/{tissue}/slideind/folda/seed42"
        )
    for tissue in (
        "breast", "colon", "lung", "ovary", "pancreatic", "skin", "tonsil",
    ):
        ids.add(
            f"cellvit-sam-h-x40/tissue_specific/{tissue}/slideind/foldb/seed42"
        )
    return ids


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def relative(path: Path) -> str:
    return path.resolve().relative_to(REPO.resolve()).as_posix()


def repo_path(value: str) -> Path:
    path = Path(value)
    return path if path.is_absolute() else REPO / path


def domain_from_id(adapter_id: str) -> str:
    parts = adapter_id.split("/")
    return parts[2] if parts[1] == "tissue_specific" else "KLT"


def collect(
    source_checkpoints_deleted: bool,
    previous_rows: dict[str, dict[str, str]] | None = None,
) -> list[dict[str, str]]:
    previous_rows = previous_rows or {}
    expected = expected_ids()
    packages = sorted(
        path.parent
        for path in ADAPTER_ROOT.rglob("verification.json")
        if "/slideind/" in path.as_posix()
        and path.as_posix().startswith(
            (
                (ADAPTER_ROOT / "cellvit-sam-h-x40").as_posix(),
                (ADAPTER_ROOT / "cellvit-256-x40").as_posix(),
            )
        )
    )
    rows = []
    found = set()
    for package in packages:
        config_path = package / "adapter_config.json"
        checksums_path = package / "checksums.json"
        weights_path = package / "adapter_model.safetensors"
        verification_path = package / "verification.json"
        for path in (config_path, checksums_path, weights_path, verification_path):
            if not path.is_file():
                raise FileNotFoundError(path)
        config = json.loads(config_path.read_text())
        checksums = json.loads(checksums_path.read_text())
        verification = json.loads(verification_path.read_text())
        adapter_id = config["adapter_id"]
        if adapter_id in found:
            raise RuntimeError(f"Duplicate canonical adapter ID: {adapter_id}")
        found.add(adapter_id)
        if adapter_id not in expected:
            raise RuntimeError(f"Unexpected slide-independent adapter ID: {adapter_id}")
        expected_adapter_sha = checksums["exported"]["adapter_model.safetensors"]
        if sha256(weights_path) != expected_adapter_sha:
            raise RuntimeError(f"Adapter checksum mismatch: {adapter_id}")
        if verification.get("status") != "ok":
            raise RuntimeError(f"Verification is not ok: {adapter_id}")
        if verification.get("state_reconstruction") != "exact_all_tensors":
            raise RuntimeError(f"State reconstruction is not exact: {adapter_id}")
        if verification.get("forward_verification") != "exact_all_output_tensors":
            raise RuntimeError(f"Forward verification is not exact: {adapter_id}")
        outputs = verification.get("forward_output_comparison", [])
        if {row["name"] for row in outputs} != {
            "hv_map", "nuclei_binary_map", "nuclei_type_map", "tissue_types"
        } or not all(row["exact"] for row in outputs):
            raise RuntimeError(f"Output coverage/equality failed: {adapter_id}")
        source_checkpoint = repo_path(verification["canonical_checkpoint"])
        run_dir = source_checkpoint.parent.parent
        checkpoint_exists = source_checkpoint.is_file()
        if checkpoint_exists == source_checkpoints_deleted:
            expectation = "deleted" if source_checkpoints_deleted else "present"
            raise RuntimeError(
                f"Source checkpoint should be {expectation}: {source_checkpoint}"
            )
        missing_provenance = [
            name for name in PRESERVED_RUN_FILES if not (run_dir / name).is_file()
        ]
        if missing_provenance:
            raise RuntimeError(
                f"Missing preserved provenance for {adapter_id}: {missing_provenance}"
            )
        tensor_names = sorted(config["expected_tensor_shapes"])
        adapter_names = [
            name.removeprefix("adapter_state_dict.")
            for name in tensor_names
            if name.startswith("adapter_state_dict.")
        ]
        buffer_names = [
            name.removeprefix("mutable_buffer_state_dict.")
            for name in tensor_names
            if name.startswith("mutable_buffer_state_dict.")
        ]
        fold = adapter_id.split("/slideind/fold", 1)[1][0].upper()
        source_config = repo_path(verification["training_config"])
        rows.append(
            {
                "adapter_id": adapter_id,
                "backbone": config["base_model"],
                "base_checkpoint_identity": Path(config["base_checkpoint"]).name,
                "base_checkpoint_sha256": config["base_checkpoint_sha256"],
                "source_run": relative(run_dir),
                "source_checkpoint": relative(source_checkpoint),
                "source_checkpoint_bytes": str(
                    source_checkpoint.stat().st_size
                    if checkpoint_exists
                    else previous_rows.get(adapter_id, {}).get(
                        "source_checkpoint_bytes", ""
                    )
                ),
                "source_checkpoint_sha256": verification["canonical_checkpoint_sha256"],
                "source_config": relative(source_config),
                "source_config_sha256": config["source_config_sha256"],
                "fold": fold,
                "seed": str(config.get("seed", config.get("random_seed"))),
                "tissue_domain": domain_from_id(adapter_id),
                "trainable_parameter_count": str(config["trainable_parameter_count"]),
                "adapter_parameter_tensor_count": str(len(adapter_names)),
                "adapter_parameter_names": ";".join(adapter_names),
                "mutable_buffer_tensor_count": str(len(buffer_names)),
                "mutable_buffer_names": ";".join(buffer_names),
                "serialized_bytes": str(weights_path.stat().st_size),
                "adapter_path": relative(weights_path),
                "adapter_sha256": expected_adapter_sha,
                "export_timestamp": config["export_timestamp"],
                "verification_timestamp": verification["verification_timestamp"],
                "verification_status": verification["status"],
                "state_reconstruction": verification["state_reconstruction"],
                "forward_verification": verification["forward_verification"],
                "verification_record": relative(verification_path),
                "safe_to_delete": "true",
                "deletion_status": (
                    "deleted_after_verified_export"
                    if source_checkpoints_deleted
                    else "present_verified_safe"
                ),
                "preserved_run_files": ";".join(PRESERVED_RUN_FILES),
            }
        )
    if found != expected:
        raise RuntimeError(
            f"Canonical adapter inventory mismatch: missing={sorted(expected-found)}, "
            f"unexpected={sorted(found-expected)}"
        )
    return sorted(rows, key=lambda row: row["adapter_id"])


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--source-checkpoints-deleted", action="store_true")
    args = parser.parse_args()
    previous_rows = {}
    if args.source_checkpoints_deleted and args.output.is_file():
        with args.output.open(newline="", encoding="utf-8") as handle:
            previous_rows = {
                row["adapter_id"]: row for row in csv.DictReader(handle)
            }
    rows = collect(args.source_checkpoints_deleted, previous_rows)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    checkpoint_bytes = sum(int(row["source_checkpoint_bytes"]) for row in rows)
    adapter_bytes = sum(int(row["serialized_bytes"]) for row in rows)
    print(
        json.dumps(
            {
                "rows": len(rows),
                "checkpoint_bytes": checkpoint_bytes,
                "adapter_bytes": adapter_bytes,
                "output": str(args.output),
                "source_checkpoints_deleted": args.source_checkpoints_deleted,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
