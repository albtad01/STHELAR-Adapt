#!/usr/bin/env python3
"""Safely retain only checkpoint_10.pth after a completed campaign run.

This is deliberately a post-run operation.  It never changes trainer behaviour,
and it refuses to remove anything unless final inference demonstrably used the
epoch-10 checkpoint and all completion artifacts are valid.
"""

import argparse
import hashlib
import json
import os
import re
import time
from pathlib import Path
from typing import Any, Dict, List

import yaml


VALIDATION_RE = re.compile(r"Validation epoch stats:.*bPQ-Score:\s*([0-9.]+)")


def _atomic_json(path: Path, payload: Dict[str, Any]) -> None:
    temporary = path.with_name(".{}.tmp-{}".format(path.name, os.getpid()))
    try:
        with temporary.open("w") as handle:
            json.dump(payload, handle, indent=2, sort_keys=True)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(str(temporary), str(path))
    finally:
        if temporary.exists():
            temporary.unlink()


def _load_json(path: Path) -> Dict[str, Any]:
    with path.open() as handle:
        value = json.load(handle)
    if not isinstance(value, dict):
        raise RuntimeError("Expected a JSON object: {}".format(path))
    return value


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _within(path: Path, parent: Path) -> bool:
    try:
        return os.path.commonpath([str(path), str(parent)]) == str(parent)
    except ValueError:
        return False


def _matching_run(config_path: Path, repo: Path, not_before: float) -> Path:
    with config_path.open() as handle:
        source_config = yaml.safe_load(handle)
    log_root = Path(source_config["logging"]["log_dir"])
    if not log_root.is_absolute():
        log_root = repo / log_root
    log_root = log_root.resolve()
    allowed_root = (repo / "run").resolve()
    if not _within(log_root, allowed_root):
        raise RuntimeError("Refusing log root outside repository run/: {}".format(log_root))
    if not log_root.is_dir():
        raise RuntimeError("Configured log root does not exist: {}".format(log_root))

    # A recovery inference can pass the generated config from an exact,
    # pre-existing run.  Generated configs store logging.log_dir as that run
    # directory itself, whereas source configs point at the parent log root.
    # Recognize only this unambiguous, in-run case; normal training finalization
    # continues through the candidate search below unchanged.
    direct_run = config_path.parent.resolve()
    if log_root == direct_run and (direct_run / "checkpoints").is_dir():
        if config_path.stat().st_mtime < not_before:
            raise RuntimeError(
                "Exact run config predates requested recovery boundary: {}".format(
                    config_path
                )
            )
        return direct_run

    candidates: List[Path] = []
    for candidate in log_root.iterdir():
        generated_config = candidate / "config.yaml"
        if not candidate.is_dir() or not generated_config.is_file():
            continue
        if generated_config.stat().st_mtime < not_before:
            continue
        try:
            with generated_config.open() as handle:
                run_config = yaml.safe_load(handle)
        except Exception:
            continue
        if (
            run_config.get("random_seed") == source_config.get("random_seed")
            and run_config.get("logging", {}).get("log_comment")
            == source_config.get("logging", {}).get("log_comment")
        ):
            candidates.append(candidate.resolve())
    if len(candidates) != 1:
        raise RuntimeError(
            "Expected exactly one run created after {:.3f}; found {}: {}".format(
                not_before, len(candidates), [str(path) for path in candidates]
            )
        )
    return candidates[0]


def _validation_history(log_path: Path) -> Dict[str, Any]:
    lines = log_path.read_text(errors="replace").splitlines()
    scores: List[float] = []
    best_epochs: List[int] = []
    for index, line in enumerate(lines):
        match = VALIDATION_RE.search(line)
        if not match:
            continue
        scores.append(float(match.group(1)))
        if index + 1 < len(lines) and "New best model - save checkpoint" in lines[index + 1]:
            best_epochs.append(len(scores))
    if len(scores) != 10:
        raise RuntimeError(
            "Expected 10 validation records in {}; found {}".format(log_path, len(scores))
        )
    if not best_epochs:
        raise RuntimeError("No validation-best checkpoint event found in {}".format(log_path))
    best_epoch = best_epochs[-1]
    return {
        "selection_metric": "validation bPQ",
        "validation_scores_by_epoch": scores,
        "best_validation_epoch": best_epoch,
        "best_validation_score": scores[best_epoch - 1],
        "final_epoch_validation_score": scores[-1],
        "best_epoch_is_10": best_epoch == 10,
    }


def finalize(config_path: Path, repo: Path, not_before: float) -> Path:
    config_path = config_path.resolve()
    repo = repo.resolve()
    run_dir = _matching_run(config_path, repo, not_before)
    checkpoint_dir = (run_dir / "checkpoints").resolve()
    checkpoint_10 = checkpoint_dir / "checkpoint_10.pth"
    if not checkpoint_dir.is_dir() or not checkpoint_10.is_file() or checkpoint_10.is_symlink():
        raise RuntimeError("Missing safe regular epoch-10 checkpoint: {}".format(checkpoint_10))

    training_metrics = _load_json(run_dir / "efficiency_metrics.json")
    inference_metrics = _load_json(run_dir / "inference_efficiency_metrics.json")
    _load_json(run_dir / "inference_results.json")
    if training_metrics.get("completed") is not True:
        raise RuntimeError("Training efficiency record is not completed")
    if inference_metrics.get("completed") is not True:
        raise RuntimeError("Inference efficiency record is not completed")

    inference_log = run_dir / "inference.log"
    inference_text = inference_log.read_text(errors="replace")
    expected_line = "For inference, loading model from {}.".format(checkpoint_10)
    if expected_line not in inference_text:
        raise RuntimeError("Final inference did not document use of {}".format(checkpoint_10))

    history = _validation_history(run_dir / "logs.log")
    checkpoint_size = checkpoint_10.stat().st_size
    checkpoint_sha256 = _sha256(checkpoint_10)
    metadata_path = run_dir / "checkpoint_retention_metadata.json"
    metadata: Dict[str, Any] = {
        "completed": False,
        "status": "validated_cleanup_pending",
        "policy": "retain_checkpoint_10_only_after_successful_final_inference",
        "primary_checkpoint": str(checkpoint_10),
        "checkpoint_size_bytes": checkpoint_size,
        "checkpoint_sha256": checkpoint_sha256,
        "training_completed": True,
        "inference_completed": True,
        "inference_used_checkpoint_10": True,
        "source_training_log": str(run_dir / "logs.log"),
        "source_inference_log": str(inference_log),
        "validated_at_unix": time.time(),
        **history,
    }
    _atomic_json(metadata_path, metadata)

    removable: List[Path] = []
    for path in checkpoint_dir.glob("*.pth"):
        resolved_parent = path.resolve().parent
        if resolved_parent != checkpoint_dir:
            raise RuntimeError("Refusing unexpected checkpoint path: {}".format(path))
        if path.name == "model_best.pth" or re.fullmatch(r"checkpoint_[1-9]\.pth", path.name):
            if path.is_symlink() or not path.is_file():
                raise RuntimeError("Refusing non-regular removable checkpoint: {}".format(path))
            removable.append(path)

    removed = []
    for path in sorted(removable):
        record = {"path": str(path), "size_bytes": path.stat().st_size}
        path.unlink()
        removed.append(record)

    remaining = sorted(path.name for path in checkpoint_dir.glob("*.pth"))
    if remaining != ["checkpoint_10.pth"]:
        raise RuntimeError(
            "Retention postcondition failed; remaining .pth files: {}".format(remaining)
        )
    metadata.update(
        {
            "completed": True,
            "status": "retention_complete",
            "removed_checkpoints": removed,
            "remaining_pth_files": remaining,
            "completed_at_unix": time.time(),
        }
    )
    _atomic_json(metadata_path, metadata)
    return metadata_path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--repo", type=Path, default=Path.cwd())
    parser.add_argument("--not-before-epoch", type=float, required=True)
    args = parser.parse_args()
    metadata_path = finalize(args.config, args.repo, args.not_before_epoch)
    print("Checkpoint retention complete: {}".format(metadata_path))


if __name__ == "__main__":
    main()
