#!/usr/bin/env python3
"""Retain canonical FullFT numerical provenance and delete completed weights.

This utility is intentionally conservative.  It accepts one explicit run,
requires completed checkpoint-10 training/inference evidence, writes and
fsyncs permanent manifests before deletion, and only removes regular FullFT
checkpoint files from that run's checkpoints directory.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
from datetime import datetime, timezone
from pathlib import Path

import yaml


REPO = Path(__file__).resolve().parents[1]
REQUIRED_RUN_FILES = (
    "config.yaml",
    "inference_results.json",
    "efficiency_metrics.json",
    "inference_efficiency_metrics.json",
    "checkpoint_retention_metadata.json",
    "logs.log",
    "inference.log",
)
ALLOWED_WEIGHT_NAME = re.compile(r"(?:checkpoint_[0-9]+|model_best)\.pth")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def relative(path: Path) -> str:
    return str(path.resolve().relative_to(REPO))


def atomic_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + f".tmp-{os.getpid()}")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)
    directory_fd = os.open(path.parent, os.O_RDONLY)
    try:
        os.fsync(directory_fd)
    finally:
        os.close(directory_fd)


def load_json(path: Path) -> dict:
    value = json.loads(path.read_text())
    if not isinstance(value, dict):
        raise RuntimeError(f"Expected JSON object: {path}")
    return value


def validate(args: argparse.Namespace) -> tuple[dict, list[Path]]:
    run_dir = args.run_dir.resolve()
    config_path = args.config.resolve()
    slurm_log = args.slurm_log.resolve()
    provenance_report = args.provenance_report.resolve()
    for path in (run_dir, config_path, slurm_log, provenance_report):
        path.relative_to(REPO)
    if args.slurm_state != "COMPLETED":
        raise RuntimeError("Refusing FullFT cleanup without final Slurm COMPLETED state")
    if not run_dir.is_dir() or not config_path.is_file():
        raise FileNotFoundError(run_dir if not run_dir.is_dir() else config_path)
    if not slurm_log.is_file() or str(args.job_id) not in slurm_log.name:
        raise RuntimeError("Missing or mismatched retained Slurm job log")
    if not provenance_report.is_file() or str(args.job_id) not in provenance_report.read_text(errors="replace"):
        raise RuntimeError("Job ID is absent from the retained provenance report")
    missing = [name for name in REQUIRED_RUN_FILES if not (run_dir / name).is_file()]
    if missing:
        raise RuntimeError(f"Missing retained FullFT provenance: {missing}")

    source_config = yaml.safe_load(config_path.read_text())
    generated_config = yaml.safe_load((run_dir / "config.yaml").read_text())
    if source_config.get("adapters", {}).get("adapter_type") != "fullft":
        raise RuntimeError("Source config is not FullFT")
    if generated_config.get("adapters", {}).get("adapter_type") != "fullft":
        raise RuntimeError("Generated config is not FullFT")
    if source_config.get("random_seed") != args.seed or generated_config.get("random_seed") != args.seed:
        raise RuntimeError("Seed mismatch")
    if int(generated_config.get("training", {}).get("epochs", -1)) != 10:
        raise RuntimeError("Run was not configured for ten epochs")
    if generated_config.get("efficiency", {}).get("fold") != args.fold:
        raise RuntimeError("Fold mismatch")

    training_log = (run_dir / "logs.log").read_text(errors="replace")
    inference_log = (run_dir / "inference.log").read_text(errors="replace")
    if "Epoch: 10/10" not in training_log:
        raise RuntimeError("Training log does not record epoch 10/10")
    checkpoint = run_dir / "checkpoints" / "checkpoint_10.pth"
    if not checkpoint.is_file() or checkpoint.is_symlink():
        raise RuntimeError("Missing safe regular checkpoint_10.pth")
    expected_inference = f"For inference, loading model from {checkpoint}."
    if expected_inference not in inference_log:
        raise RuntimeError("Inference log does not explicitly load checkpoint_10.pth")

    results = load_json(run_dir / "inference_results.json")
    if not all(isinstance(results.get(key), dict) and results[key] for key in (
        "image_metrics", "nuclei_metrics_d", "nuclei_metrics_pq"
    )):
        raise RuntimeError("Canonical TEST JSON lacks reconstructable numerical metrics")
    training_efficiency = load_json(run_dir / "efficiency_metrics.json")
    inference_efficiency = load_json(run_dir / "inference_efficiency_metrics.json")
    retention = load_json(run_dir / "checkpoint_retention_metadata.json")
    if training_efficiency.get("completed") is not True or inference_efficiency.get("completed") is not True:
        raise RuntimeError("Efficiency metadata is not complete")
    if not (
        retention.get("completed") is True
        and retention.get("status") == "retention_complete"
        and retention.get("training_completed") is True
        and retention.get("inference_completed") is True
        and retention.get("inference_used_checkpoint_10") is True
    ):
        raise RuntimeError("Checkpoint-10 retention metadata is incomplete")
    checkpoint_digest = sha256(checkpoint)
    if checkpoint_digest != retention.get("checkpoint_sha256"):
        raise RuntimeError("Checkpoint SHA256 disagrees with canonical retention metadata")

    checkpoint_dir = checkpoint.parent.resolve()
    weights = sorted(checkpoint_dir.glob("*.pth"))
    if not weights:
        raise RuntimeError("No FullFT weight files found")
    for path in weights:
        if path.resolve().parent != checkpoint_dir or path.is_symlink() or not path.is_file():
            raise RuntimeError(f"Unsafe checkpoint path: {path}")
        if ALLOWED_WEIGHT_NAME.fullmatch(path.name) is None:
            raise RuntimeError(f"Unexpected FullFT weight filename: {path.name}")

    parameter_count = training_efficiency.get("total_parameters")
    trainable_count = training_efficiency.get("trainable_parameters")
    if not isinstance(parameter_count, int) or trainable_count != parameter_count:
        raise RuntimeError("FullFT parameter count is absent or not fully trainable")

    weight_records = [
        {
            "path": relative(path),
            "bytes": path.stat().st_size,
            "sha256": checkpoint_digest if path == checkpoint else sha256(path),
        }
        for path in weights
    ]
    payload = {
        "schema_version": 1,
        "policy": "retain_fullft_results_and_provenance_delete_all_model_weights",
        "status": "validated_pending_weight_deletion",
        "validation_timestamp": datetime.now(timezone.utc).isoformat(),
        "job_id": args.job_id,
        "final_slurm_state": args.slurm_state,
        "backbone": args.backbone,
        "tissue_or_domain": args.domain,
        "fold": args.fold,
        "seed": args.seed,
        "run_dir": relative(run_dir),
        "exact_config_path": relative(config_path),
        "source_config_sha256": sha256(config_path),
        "generated_config_path": relative(run_dir / "config.yaml"),
        "generated_config_sha256": sha256(run_dir / "config.yaml"),
        "canonical_checkpoint_filename": "checkpoint_10.pth",
        "canonical_checkpoint_sha256": checkpoint_digest,
        "parameter_count": parameter_count,
        "trainable_parameter_count": trainable_count,
        "canonical_test_metrics_path": relative(run_dir / "inference_results.json"),
        "canonical_test_metrics_sha256": sha256(run_dir / "inference_results.json"),
        "training_efficiency_path": relative(run_dir / "efficiency_metrics.json"),
        "training_efficiency_sha256": sha256(run_dir / "efficiency_metrics.json"),
        "inference_efficiency_path": relative(run_dir / "inference_efficiency_metrics.json"),
        "inference_efficiency_sha256": sha256(run_dir / "inference_efficiency_metrics.json"),
        "checkpoint_retention_metadata_path": relative(run_dir / "checkpoint_retention_metadata.json"),
        "checkpoint_retention_metadata_sha256": sha256(run_dir / "checkpoint_retention_metadata.json"),
        "training_log_path": relative(run_dir / "logs.log"),
        "inference_log_path": relative(run_dir / "inference.log"),
        "slurm_log_path": relative(slurm_log),
        "job_provenance_report": relative(provenance_report),
        "test_explicitly_used_checkpoint_10": True,
        "numerical_results_reconstructable_without_checkpoint": True,
        "deleted_weight_files": weight_records,
        "bytes_scheduled_for_deletion": sum(row["bytes"] for row in weight_records),
        "weights_exist_after_retention": True,
        "retained_files": [
            relative(run_dir / name) for name in REQUIRED_RUN_FILES
        ] + [relative(config_path), relative(slurm_log), relative(provenance_report)],
    }
    return payload, weights


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    # Slurm array task identifiers use the canonical ``<array>_<task>`` form.
    # Keep this as a string so every retention manifest names the exact
    # scientific task rather than only the shared array parent.
    parser.add_argument("--job-id", required=True)
    parser.add_argument("--backbone", required=True)
    parser.add_argument("--domain", required=True)
    parser.add_argument("--fold", choices=("A", "B"), required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--slurm-state", required=True)
    parser.add_argument("--slurm-log", type=Path, required=True)
    parser.add_argument("--provenance-report", type=Path, required=True)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    payload, weights = validate(args)
    if args.dry_run:
        print(json.dumps(payload, indent=2, sort_keys=True))
        return

    run_manifest = args.run_dir.resolve() / "fullft_weights_retention_manifest.json"
    central_manifest = (
        REPO / "reports" / "fullft_retention_manifests" /
        f"job_{args.job_id}_{args.domain.lower()}_fold{args.fold}_seed{args.seed}.json"
    )
    atomic_json(run_manifest, payload)
    atomic_json(central_manifest, payload)

    for path in weights:
        path.unlink()

    remaining = sorted(path.name for path in args.run_dir.resolve().glob("checkpoints/*.pth"))
    if remaining:
        raise RuntimeError(f"FullFT weight deletion postcondition failed: {remaining}")
    payload.update({
        "status": "fullft_weights_deleted",
        "deletion_timestamp": datetime.now(timezone.utc).isoformat(),
        "weights_exist_after_retention": False,
        "remaining_pth_files": remaining,
    })
    atomic_json(run_manifest, payload)
    atomic_json(central_manifest, payload)
    print(json.dumps({
        "job_id": args.job_id,
        "status": payload["status"],
        "bytes_deleted": payload["bytes_scheduled_for_deletion"],
        "run_manifest": relative(run_manifest),
        "central_manifest": relative(central_manifest),
    }, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
