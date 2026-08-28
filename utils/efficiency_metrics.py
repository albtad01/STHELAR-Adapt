"""Atomic operational-efficiency metrics for training and inference runs."""

import atexit
import csv
import json
import os
import platform
import time
from pathlib import Path
from typing import Any, Dict, Optional

import torch


def _count_csv_rows(path: Path) -> Optional[int]:
    if not path.is_file():
        return None
    with path.open(newline="") as handle:
        return sum(1 for _ in csv.DictReader(handle))


def dataset_patch_counts(experiment_config: Dict[str, Any]) -> Dict[str, Optional[int]]:
    data = experiment_config.get("data", {})
    dataset_path = Path(str(data.get("dataset_path", "")))
    return {
        "train_patch_count": _count_csv_rows(dataset_path / "cell_count_train.csv"),
        "validation_patch_count": _count_csv_rows(
            dataset_path / "cell_count_valid.csv"
        ),
        "test_patch_count": _count_csv_rows(dataset_path / "cell_count_test.csv"),
    }


def parameter_counts(model: torch.nn.Module) -> Dict[str, float]:
    total = int(sum(parameter.numel() for parameter in model.parameters()))
    trainable = int(
        sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)
    )
    percentage = float(100.0 * trainable / total) if total else 0.0
    return {
        "trainable_parameters": trainable,
        "total_parameters": total,
        "trainable_percentage": percentage,
    }


def backbone_identity(experiment_config: Dict[str, Any]) -> Optional[str]:
    """Return a stable, magnification-explicit backbone label for aggregation."""
    efficiency = experiment_config.get("efficiency", {})
    explicit = efficiency.get("backbone")
    if explicit:
        return str(explicit)
    backbone = str(experiment_config.get("model", {}).get("backbone", "")).lower()
    magnification = experiment_config.get("data", {}).get("magnification")
    suffix = " x{}".format(magnification) if magnification is not None else ""
    if backbone == "sam-h":
        return "CellViT-SAM-H{}".format(suffix)
    if backbone == "vit256":
        return "CellViT-256{}".format(suffix)
    return backbone or None


class EfficiencyRecorder:
    """Write crash-distinguishable JSON with atomic replacement semantics."""

    SCHEMA_VERSION = 1

    def __init__(
        self,
        output_path: Path,
        mode: str,
        experiment_config: Dict[str, Any],
        model: torch.nn.Module,
        device: str,
        batch_size: int,
        patch_count: Optional[int] = None,
    ) -> None:
        self.output_path = Path(output_path)
        self.mode = str(mode)
        self.device = str(device)
        self._start_time: Optional[float] = None
        self._epoch_start: Optional[float] = None
        self._cuda_enabled = bool(
            torch.cuda.is_available() and self.device.startswith("cuda")
        )
        efficiency = experiment_config.get("efficiency", {})
        self.data: Dict[str, Any] = {
            "schema_version": self.SCHEMA_VERSION,
            "mode": self.mode,
            "completed": False,
            "status": "initialized",
            "fold": efficiency.get("fold"),
            "tissue": efficiency.get("tissue"),
            "method": efficiency.get("method"),
            "backbone": backbone_identity(experiment_config),
            "partition": os.environ.get("SLURM_JOB_PARTITION"),
            "seed": experiment_config.get("random_seed"),
            "batch_size": int(batch_size),
            "amp_mixed_precision": bool(
                experiment_config.get("training", {}).get("mixed_precision", False)
            ),
            "device": self.device,
            "gpu_model": (
                torch.cuda.get_device_name(torch.cuda.current_device())
                if self._cuda_enabled
                else None
            ),
            "cuda_version": torch.version.cuda,
            "pytorch_version": torch.__version__,
            "python_version": platform.python_version(),
            "epoch_time_seconds": [],
            "error": None,
        }
        self.data.update(parameter_counts(model))
        if self.mode == "training":
            trainable_names = [
                name for name, parameter in model.named_parameters()
                if parameter.requires_grad
            ]
            self.data["trainable_parameter_names"] = trainable_names
            self.data["trainable_module_names"] = sorted(
                {name.rsplit(".", 1)[0] for name in trainable_names}
            )
        self.data.update(dataset_patch_counts(experiment_config))
        if patch_count is not None:
            self.data["inference_patch_count"] = int(patch_count)
        self._write()
        atexit.register(self._finalize_incomplete_at_exit)

    def _finalize_incomplete_at_exit(self) -> None:
        """Persist elapsed/peak values when Python exits before finalization."""
        if self.data.get("completed", False):
            return
        if self._start_time is not None:
            self._finish_common()
        self.data["status"] = "interrupted_or_failed"
        self.data["completed"] = False
        self._write()

    def _synchronize(self) -> None:
        if self._cuda_enabled:
            torch.cuda.synchronize()

    def _write(self) -> None:
        self.output_path.parent.mkdir(parents=True, exist_ok=True)
        temporary = self.output_path.with_name(
            ".{}.tmp-{}".format(self.output_path.name, os.getpid())
        )
        try:
            with temporary.open("w") as handle:
                json.dump(self.data, handle, indent=2, sort_keys=True)
                handle.flush()
                os.fsync(handle.fileno())
            os.replace(str(temporary), str(self.output_path))
        finally:
            if temporary.exists():
                temporary.unlink()

    def start(self) -> None:
        self._synchronize()
        if self._cuda_enabled:
            torch.cuda.reset_peak_memory_stats()
        self._start_time = time.perf_counter()
        self.data["status"] = "running"
        self.data["completed"] = False
        self._write()

    def start_epoch(self) -> None:
        self._synchronize()
        self._epoch_start = time.perf_counter()

    def finish_epoch(self) -> float:
        if self._epoch_start is None:
            raise RuntimeError("finish_epoch called without start_epoch")
        self._synchronize()
        elapsed = float(time.perf_counter() - self._epoch_start)
        self.data["epoch_time_seconds"].append(elapsed)
        self._epoch_start = None
        self._write()
        return elapsed

    def _finish_common(self) -> None:
        if self._start_time is None:
            return
        self._synchronize()
        elapsed = float(time.perf_counter() - self._start_time)
        if self.mode == "training":
            self.data["training_time_seconds"] = elapsed
            self.data["training_time_hours"] = elapsed / 3600.0
            epochs = self.data.get("epoch_time_seconds", [])
            self.data["mean_seconds_per_epoch"] = (
                float(sum(epochs) / len(epochs)) if epochs else None
            )
        else:
            self.data["inference_time_seconds"] = elapsed
            count = self.data.get("inference_patch_count")
            self.data["patches_per_second"] = (
                float(count / elapsed) if count is not None and elapsed > 0 else None
            )
        allocated = int(torch.cuda.max_memory_allocated()) if self._cuda_enabled else 0
        reserved = int(torch.cuda.max_memory_reserved()) if self._cuda_enabled else 0
        self.data["peak_cuda_memory_allocated_bytes"] = allocated
        self.data["peak_cuda_memory_reserved_bytes"] = reserved
        self.data["peak_cuda_memory_allocated_gib"] = allocated / float(2**30)
        self.data["peak_cuda_memory_reserved_gib"] = reserved / float(2**30)

    def finish_interval(self, status: str) -> None:
        self._finish_common()
        self.data["status"] = status
        self.data["completed"] = False
        self._write()

    def finalize_training(self, checkpoint_path: Optional[Path]) -> None:
        if "training_time_seconds" not in self.data:
            self._finish_common()
        checkpoint = Path(checkpoint_path) if checkpoint_path is not None else None
        size = int(checkpoint.stat().st_size) if checkpoint and checkpoint.is_file() else None
        self.data["checkpoint_path"] = str(checkpoint) if checkpoint else None
        self.data["checkpoint_size_bytes"] = size
        self.data["checkpoint_size_gib"] = size / float(2**30) if size is not None else None
        self.data["status"] = "completed"
        self.data["completed"] = True
        self._write()
        atexit.unregister(self._finalize_incomplete_at_exit)

    def finalize_inference(self) -> None:
        self._finish_common()
        self.data["status"] = "completed"
        self.data["completed"] = True
        self._write()
        atexit.unregister(self._finalize_incomplete_at_exit)

    def fail(self, error: BaseException) -> None:
        if self._start_time is not None:
            self._finish_common()
        self.data["status"] = "failed"
        self.data["completed"] = False
        self.data["error"] = "{}: {}".format(type(error).__name__, error)
        self._write()
        atexit.unregister(self._finalize_incomplete_at_exit)
