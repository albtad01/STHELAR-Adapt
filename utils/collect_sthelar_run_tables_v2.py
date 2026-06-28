import argparse
import csv
import re
import subprocess
from datetime import datetime
from pathlib import Path

import yaml


def read_text(path: Path) -> str:
    if path is None or not path.exists():
        return ""
    return path.read_text(errors="ignore")


def read_yaml(path: Path) -> dict:
    if path is None or not path.exists():
        return {}
    try:
        with path.open("r") as f:
            return yaml.safe_load(f) or {}
    except Exception:
        return {}


def get_nested(d, keys, default=""):
    cur = d
    for k in keys:
        if not isinstance(cur, dict) or k not in cur:
            return default
        cur = cur[k]
    return cur


def as_list_str(x):
    if x is None:
        return ""
    if isinstance(x, list):
        return ",".join(str(v) for v in x)
    return str(x)


def parse_timestamp(line: str):
    m = re.match(r"(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}),(\d+)", line)
    if not m:
        return None
    base = m.group(1)
    ms = m.group(2)[:3].ljust(3, "0")
    return datetime.strptime(base + "," + ms, "%Y-%m-%d %H:%M:%S,%f")


def parse_trainable_report(text: str) -> dict:
    out = {
        "total_params": "",
        "trainable_params": "",
        "trainable_ratio_percent": "",
    }

    patterns = {
        "total_params": [
            r"total_params:\s*([0-9,]+)",
            r"Total params:\s*([0-9,]+)",
        ],
        "trainable_params": [
            r"trainable_params:\s*([0-9,]+)",
            r"Trainable params:\s*([0-9,]+)",
        ],
        "trainable_ratio_percent": [
            r"trainable_ratio_percent:\s*([0-9.]+)",
        ],
    }

    for key, pats in patterns.items():
        for pat in pats:
            m = re.search(pat, text)
            if m:
                out[key] = m.group(1).replace(",", "")
                break

    if out["total_params"] and out["trainable_params"] and not out["trainable_ratio_percent"]:
        total = float(out["total_params"])
        trainable = float(out["trainable_params"])
        out["trainable_ratio_percent"] = f"{100.0 * trainable / total:.4f}"

    return out


def parse_epoch_metrics(log_text: str, run_name: str):
    epoch_re = re.compile(r"Epoch:\s+(\d+)/(\d+)")

    train_re = re.compile(
        r"Training epoch stats:\s+Loss:\s+([0-9.]+).*?"
        r"Binary-Cell-Dice:\s+([0-9.]+).*?"
        r"Binary-Cell-Jacard:\s+([0-9.]+)"
    )

    val_re = re.compile(
        r"Validation epoch stats:\s+Loss:\s+([0-9.]+).*?"
        r"Binary-Cell-Dice:\s+([0-9.]+).*?"
        r"Binary-Cell-Jacard:\s+([0-9.]+).*?"
        r"bPQ-Score:\s+([0-9.]+).*?"
        r"mPQ-Score:\s+([0-9.]+)"
    )

    lr_re = re.compile(r"Old lr:\s+([0-9.eE+-]+)\s+-\s+New lr:\s+([0-9.eE+-]+)")

    rows = []
    current = {}
    current_epoch = None
    current_epoch_start = None
    last_train_end = None

    def flush():
        nonlocal current
        if current_epoch is not None and current:
            current["run_name"] = run_name
            current["epoch"] = current_epoch
            rows.append(current)
        current = {}

    for line in log_text.splitlines():
        ts = parse_timestamp(line)

        m = epoch_re.search(line)
        if m:
            flush()
            current_epoch = int(m.group(1))
            current_epoch_start = ts
            last_train_end = None
            current = {
                "max_epochs": int(m.group(2)),
                "epoch_start_time": ts.isoformat(sep=" ") if ts else "",
                "new_best": 0,
            }
            continue

        m = train_re.search(line)
        if m:
            current["train_loss"] = float(m.group(1))
            current["train_Dice"] = float(m.group(2))
            current["train_Jaccard"] = float(m.group(3))
            if ts:
                last_train_end = ts
                current["train_end_time"] = ts.isoformat(sep=" ")
                if current_epoch_start:
                    current["train_minutes"] = round((ts - current_epoch_start).total_seconds() / 60.0, 2)
            continue

        m = val_re.search(line)
        if m:
            current["val_loss"] = float(m.group(1))
            current["val_Dice"] = float(m.group(2))
            current["val_Jaccard"] = float(m.group(3))
            current["val_bPQ"] = float(m.group(4))
            current["val_mPQ"] = float(m.group(5))
            if ts:
                current["val_end_time"] = ts.isoformat(sep=" ")
                if current_epoch_start:
                    current["epoch_wall_minutes"] = round((ts - current_epoch_start).total_seconds() / 60.0, 2)
                if last_train_end:
                    current["validation_minutes"] = round((ts - last_train_end).total_seconds() / 60.0, 2)
            continue

        m = lr_re.search(line)
        if m:
            current["old_lr"] = float(m.group(1))
            current["new_lr"] = float(m.group(2))
            continue

        if "New best model" in line:
            current["new_best"] = 1

    flush()
    return rows


def parse_test_metrics(inference_text: str):
    patterns = {
        "test_Dice": r"Binary-Cell-Dice-Mean:\s*([0-9.]+)",
        "test_Jaccard": r"Binary-Cell-Jacard-Mean:\s*([0-9.]+)",
        "tissue_accuracy": r"Tissue-Multiclass-Accuracy:\s*([0-9.]+)",
        "test_bPQ": r"(?m)^bPQ:\s*([0-9.]+)",
        "test_bDQ": r"(?m)^bDQ:\s*([0-9.]+)",
        "test_bSQ": r"(?m)^bSQ:\s*([0-9.]+)",
        "test_mPQ": r"(?m)^mPQ:\s*([0-9.]+)",
        "test_mDQ": r"(?m)^mDQ:\s*([0-9.]+)",
        "test_mSQ": r"(?m)^mSQ:\s*([0-9.]+)",
        "test_F1": r"f1_detection:\s*([0-9.]+)",
        "test_precision": r"precision_detection:\s*([0-9.]+)",
        "test_recall": r"recall_detection:\s*([0-9.]+)",
    }

    out = {}
    for key, pat in patterns.items():
        m = re.search(pat, inference_text)
        out[key] = m.group(1) if m else ""

    m = re.search(r"For inference, loading model from\s+(.+?\.pth)", inference_text)
    out["inference_checkpoint"] = m.group(1).strip() if m else ""

    return out


def infer_effective_adapter_type(config_adapter_type: str, inference_text: str, config: dict):
    adapter_type = config_adapter_type or ""

    if "No adapters added" in inference_text and adapter_type in {"lora", "adaptformer", "lora_ntonly", "plora"}:
        return f"{adapter_type}_problem_no_adapters_added"

    return adapter_type


def infer_status(log_text: str, inference_text: str, slurm_state: str = ""):
    if slurm_state:
        st = slurm_state.upper()
        if "TIMEOUT" in st:
            if "Binary Dataset metrics" in inference_text:
                return "timeout_after_test"
            if "Finished run" in log_text:
                return "timeout_during_or_after_inference"
            return "timeout_during_training"
        if "FAILED" in st:
            return "failed"
        if "COMPLETED" in st and "Binary Dataset metrics" in inference_text:
            return "completed_with_test"
        if "COMPLETED" in st and "Finished run" in log_text:
            return "training_finished_no_test"

    combined = (log_text + "\n" + inference_text).lower()

    if "binary dataset metrics" in combined:
        return "completed_with_test"
    if "traceback" in combined or "keyerror" in combined or "runtimeerror" in combined:
        return "failed"
    if "finished run" in combined:
        return "training_finished_no_test"
    if "epoch:" in combined:
        return "partial_or_timeout"
    return "unknown"


def build_slurm_map(slurm_log_dir: Path):
    """
    Map config absolute path -> job_id using logs/*.out.
    It expects lines like:
    Config: /path/to/config.yaml
    """
    out = {}

    if not slurm_log_dir.exists():
        return out

    for f in slurm_log_dir.glob("*.out"):
        text = read_text(f)
        m_cfg = re.search(r"Config:\s*(.+?\.ya?ml)", text)
        m_job = re.search(r"_(\d+)\.out$", f.name)
        if m_cfg and m_job:
            out[str(Path(m_cfg.group(1)).resolve())] = {
                "slurm_job_id": m_job.group(1),
                "slurm_stdout": str(f),
                "slurm_stderr": str(f.with_suffix(".err")),
            }

    return out


def query_sacct(job_id: str):
    if not job_id:
        return {}

    try:
        cmd = [
            "sacct",
            "-j", job_id,
            "--noheader",
            "--parsable2",
            "--format=JobID,State,ExitCode,Elapsed,Timelimit,MaxRSS",
        ]
        res = subprocess.run(cmd, capture_output=True, text=True, timeout=10)
        if res.returncode != 0:
            return {}

        for line in res.stdout.splitlines():
            parts = line.split("|")
            if len(parts) < 6:
                continue
            if parts[0] == job_id:
                return {
                    "slurm_state": parts[1],
                    "slurm_exit_code": parts[2],
                    "elapsed": parts[3],
                    "time_limit": parts[4],
                    "max_rss": parts[5],
                }
    except Exception:
        return {}

    return {}


def best_epoch_values(epoch_rows):
    valid = [r for r in epoch_rows if r.get("val_mPQ") not in ("", None)]
    if not valid:
        return {
            "best_val_mPQ": "",
            "best_val_bPQ": "",
            "best_val_Dice": "",
            "best_epoch_mPQ": "",
            "best_epoch_bPQ": "",
            "best_epoch_Dice": "",
        }

    best_mPQ = max(valid, key=lambda r: float(r.get("val_mPQ", -1)))
    best_bPQ = max(valid, key=lambda r: float(r.get("val_bPQ", -1)))
    best_Dice = max(valid, key=lambda r: float(r.get("val_Dice", -1)))

    return {
        "best_val_mPQ": best_mPQ.get("val_mPQ", ""),
        "best_val_bPQ": best_bPQ.get("val_bPQ", ""),
        "best_val_Dice": best_Dice.get("val_Dice", ""),
        "best_epoch_mPQ": best_mPQ.get("epoch", ""),
        "best_epoch_bPQ": best_bPQ.get("epoch", ""),
        "best_epoch_Dice": best_Dice.get("epoch", ""),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-root", default="run")
    parser.add_argument("--slurm-log-dir", default="logs")
    parser.add_argument("--use-sacct", action="store_true")
    parser.add_argument("--out-summary", default="reports/runs_summary.csv")
    parser.add_argument("--out-epochs", default="reports/epoch_metrics.csv")
    args = parser.parse_args()

    run_root = Path(args.run_root)
    slurm_map = build_slurm_map(Path(args.slurm_log_dir))

    summary_rows = []
    all_epoch_rows = []

    for config_path in sorted(run_root.rglob("config.yaml")):
        run_dir = config_path.parent
        timestamped_run = run_dir.name

        config = read_yaml(config_path)
        log_path = run_dir / "logs.log"
        inference_path = run_dir / "inference.log"

        log_text = read_text(log_path)
        inference_text = read_text(inference_path)

        run_name = get_nested(config, ["logging", "log_comment"], default=timestamped_run)
        dataset_path = get_nested(config, ["data", "dataset_path"])

        manifest_path = run_dir / "dataset_manifest.yaml"
        if not manifest_path.exists() and dataset_path:
            fallback = Path(dataset_path) / "split_manifest.yaml"
            if fallback.exists():
                manifest_path = fallback

        manifest = read_yaml(manifest_path)

        adapter_conf = get_nested(config, ["adapters"], default={})
        adapter_type = adapter_conf.get("adapter_type", "")

        lora_conf = adapter_conf.get("lora", {}) if isinstance(adapter_conf.get("lora", {}), dict) else {}
        adapt_conf = adapter_conf.get("adaptformer", {}) if isinstance(adapter_conf.get("adaptformer", {}), dict) else {}

        epoch_rows = parse_epoch_metrics(log_text, run_name)
        all_epoch_rows.extend(epoch_rows)

        best_vals = best_epoch_values(epoch_rows)
        trainable = parse_trainable_report(log_text)
        test_metrics = parse_test_metrics(inference_text)

        config_abs = str(config_path.resolve())
        slurm_info = slurm_map.get(config_abs, {})
        if args.use_sacct and slurm_info.get("slurm_job_id"):
            slurm_info.update(query_sacct(slurm_info["slurm_job_id"]))

        effective_adapter_type = infer_effective_adapter_type(adapter_type, inference_text, config)
        has_inference = bool(test_metrics.get("test_mPQ"))

        status = infer_status(
            log_text=log_text,
            inference_text=inference_text,
            slurm_state=slurm_info.get("slurm_state", ""),
        )

        row = {
            "run_name": run_name,
            "timestamped_run": timestamped_run,
            "status": status,
            "slurm_job_id": slurm_info.get("slurm_job_id", ""),
            "slurm_state": slurm_info.get("slurm_state", ""),
            "slurm_exit_code": slurm_info.get("slurm_exit_code", ""),
            "elapsed": slurm_info.get("elapsed", ""),
            "time_limit": slurm_info.get("time_limit", ""),
            "max_rss": slurm_info.get("max_rss", ""),

            "dataset": get_nested(config, ["data", "dataset"]),
            "dataset_path": dataset_path,
            "magnification": get_nested(config, ["data", "magnification"]),
            "tissue": manifest.get("tissue", "") or manifest.get("tissue_name", "") or ",".join(get_nested(config, ["dataset_config", "tissue_types"], default={}).keys()),
            "label_mode": manifest.get("label_mode", ""),
            "num_classes": get_nested(config, ["data", "num_nuclei_classes"]),
            "split": manifest.get("actual_strategy", manifest.get("requested_strategy", "")),
            "train_slides": as_list_str(manifest.get("train_slides", manifest.get("slides_by_split", {}).get("train", ""))),
            "valid_slides": as_list_str(manifest.get("valid_slides", manifest.get("slides_by_split", {}).get("valid", ""))),
            "test_slides": as_list_str(manifest.get("test_slides", manifest.get("slides_by_split", {}).get("test", ""))),
            "max_patches_per_slide": manifest.get("max_patches_per_slide", ""),
            "counts_train": get_nested(manifest, ["counts_after_split", "train"]),
            "counts_valid": get_nested(manifest, ["counts_after_split", "valid"]),
            "counts_test": get_nested(manifest, ["counts_after_split", "test"]),
            "counts_total": get_nested(manifest, ["counts_after_split", "total"]),
            "uses_spatial_valid_inside_train_slide": manifest.get("uses_spatial_valid_inside_train_slide", ""),

            "pretrained_checkpoint": get_nested(config, ["model", "pretrained"]),

            "adapter_type": adapter_type,
            "effective_adapter_type": effective_adapter_type,
            "rank": lora_conf.get("rank", ""),
            "alpha": lora_conf.get("alpha", ""),
            "targets": as_list_str(lora_conf.get("targets", "")),
            "dropout": lora_conf.get("dropout", ""),
            "reduction": adapt_conf.get("reduction", adapt_conf.get("reduction_factor", "")),
            "activation": adapt_conf.get("activation", ""),

            "epochs": get_nested(config, ["training", "epochs"]),
            "seed": get_nested(config, ["random_seed"]),
            "batch_size": get_nested(config, ["training", "batch_size"]),
            "lr": get_nested(config, ["training", "optimizer_hyperparameter", "lr"]),
            "unfreeze_encoder": get_nested(config, ["training", "unfreeze_encoder"]),
            "unfreeze_epoch": get_nested(config, ["training", "unfreeze_epoch"]),
            "total_params": trainable.get("total_params", ""),
            "trainable_params": trainable.get("trainable_params", ""),
            "trainable_ratio_percent": trainable.get("trainable_ratio_percent", ""),

            **best_vals,

            "has_inference": has_inference,
            "inference_checkpoint": test_metrics.get("inference_checkpoint", ""),
            "test_mPQ": test_metrics.get("test_mPQ", ""),
            "test_bPQ": test_metrics.get("test_bPQ", ""),
            "test_Dice": test_metrics.get("test_Dice", ""),
            "test_Jaccard": test_metrics.get("test_Jaccard", ""),
            "test_F1": test_metrics.get("test_F1", ""),
            "test_precision": test_metrics.get("test_precision", ""),
            "test_recall": test_metrics.get("test_recall", ""),
            "test_mDQ": test_metrics.get("test_mDQ", ""),
            "test_mSQ": test_metrics.get("test_mSQ", ""),
            "test_bDQ": test_metrics.get("test_bDQ", ""),
            "test_bSQ": test_metrics.get("test_bSQ", ""),
            "tissue_accuracy": test_metrics.get("tissue_accuracy", ""),

            "config_path": str(config_path),
            "dataset_manifest_path": str(manifest_path) if manifest_path and manifest_path.exists() else "",
            "log_path": str(log_path) if log_path.exists() else "",
            "inference_log_path": str(inference_path) if inference_path.exists() else "",
            "run_dir": str(run_dir),
            "notes": get_nested(config, ["logging", "notes"]),
        }

        summary_rows.append(row)

    summary_fields = [
        "run_name", "timestamped_run", "status",
        "slurm_job_id", "slurm_state", "slurm_exit_code", "elapsed", "time_limit", "max_rss",

        "dataset", "dataset_path", "magnification", "tissue", "label_mode", "num_classes",
        "split", "train_slides", "valid_slides", "test_slides", "max_patches_per_slide",
        "counts_train", "counts_valid", "counts_test", "counts_total",
        "uses_spatial_valid_inside_train_slide",
        "pretrained_checkpoint",

        "adapter_type", "effective_adapter_type", "rank", "alpha", "targets", "dropout",
        "reduction", "activation",

        "epochs", "seed", "batch_size", "lr", "unfreeze_encoder", "unfreeze_epoch",
        "total_params", "trainable_params", "trainable_ratio_percent",

        "best_val_mPQ", "best_val_bPQ", "best_val_Dice",
        "best_epoch_mPQ", "best_epoch_bPQ", "best_epoch_Dice",

        "has_inference", "inference_checkpoint",
        "test_mPQ", "test_bPQ", "test_Dice", "test_Jaccard", "test_F1",
        "test_precision", "test_recall", "test_mDQ", "test_mSQ", "test_bDQ", "test_bSQ",
        "tissue_accuracy",

        "config_path", "dataset_manifest_path", "log_path", "inference_log_path",
        "run_dir", "notes",
    ]

    epoch_fields = [
        "run_name", "epoch", "max_epochs",
        "epoch_start_time", "train_end_time", "val_end_time",
        "train_minutes", "validation_minutes", "epoch_wall_minutes",
        "train_loss", "train_Dice", "train_Jaccard",
        "val_loss", "val_Dice", "val_Jaccard", "val_bPQ", "val_mPQ",
        "old_lr", "new_lr", "new_best",
    ]

    Path(args.out_summary).parent.mkdir(parents=True, exist_ok=True)

    with open(args.out_summary, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=summary_fields)
        writer.writeheader()
        for row in summary_rows:
            writer.writerow({k: row.get(k, "") for k in summary_fields})

    with open(args.out_epochs, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=epoch_fields)
        writer.writeheader()
        for row in all_epoch_rows:
            writer.writerow({k: row.get(k, "") for k in epoch_fields})

    print(f"Wrote {len(summary_rows)} run rows to {args.out_summary}")
    print(f"Wrote {len(all_epoch_rows)} epoch rows to {args.out_epochs}")


if __name__ == "__main__":
    main()