import re
import csv
import yaml
import argparse
from pathlib import Path


def get_nested(d, keys, default=""):
    cur = d
    for k in keys:
        if not isinstance(cur, dict) or k not in cur:
            return default
        cur = cur[k]
    return cur


def first_existing(paths):
    for p in paths:
        if p.exists():
            return p
    return None


def parse_trainable_report(text):
    out = {
        "trainable_params": "",
        "trainable_ratio_percent": "",
        "total_params": "",
    }

    m = re.search(r"total_params:\s*([0-9,]+)", text)
    if m:
        out["total_params"] = m.group(1).replace(",", "")

    m = re.search(r"trainable_params:\s*([0-9,]+)", text)
    if m:
        out["trainable_params"] = m.group(1).replace(",", "")

    m = re.search(r"trainable_ratio_percent:\s*([0-9.]+)", text)
    if m:
        out["trainable_ratio_percent"] = m.group(1)

    # fallback from torch summary
    if not out["total_params"]:
        m = re.search(r"Total params:\s*([0-9,]+)", text)
        if m:
            out["total_params"] = m.group(1).replace(",", "")

    if not out["trainable_params"]:
        m = re.search(r"Trainable params:\s*([0-9,]+)", text)
        if m:
            out["trainable_params"] = m.group(1).replace(",", "")

    if out["trainable_params"] and out["total_params"] and not out["trainable_ratio_percent"]:
        tr = float(out["trainable_params"])
        tot = float(out["total_params"])
        out["trainable_ratio_percent"] = f"{100.0 * tr / tot:.4f}"

    return out


def parse_epoch_metrics(log_text, run_name):
    epoch_re = re.compile(r"Epoch:\s+(\d+)/(\d+)")
    train_re = re.compile(
        r"Training epoch stats:\s+Loss:\s+([0-9.]+).*?"
        r"Binary-Cell-Dice:\s+([0-9.]+).*?"
        r"Binary-Cell-Jacard:\s+([0-9.]+)",
    )
    val_re = re.compile(
        r"Validation epoch stats:\s+Loss:\s+([0-9.]+).*?"
        r"Binary-Cell-Dice:\s+([0-9.]+).*?"
        r"Binary-Cell-Jacard:\s+([0-9.]+).*?"
        r"bPQ-Score:\s+([0-9.]+).*?"
        r"mPQ-Score:\s+([0-9.]+)",
    )
    lr_re = re.compile(r"Old lr:\s+([0-9.eE+-]+)\s+-\s+New lr:\s+([0-9.eE+-]+)")
    best_re = re.compile(r"New best model")

    rows = []
    current_epoch = None
    current = {}

    for line in log_text.splitlines():
        m = epoch_re.search(line)
        if m:
            if current_epoch is not None and current:
                current["run_name"] = run_name
                current["epoch"] = current_epoch
                rows.append(current)
            current_epoch = int(m.group(1))
            current = {"max_epochs": int(m.group(2)), "new_best": 0}

        m = train_re.search(line)
        if m:
            current["train_loss"] = float(m.group(1))
            current["train_Dice"] = float(m.group(2))
            current["train_Jaccard"] = float(m.group(3))

        m = val_re.search(line)
        if m:
            current["val_loss"] = float(m.group(1))
            current["val_Dice"] = float(m.group(2))
            current["val_Jaccard"] = float(m.group(3))
            current["val_bPQ"] = float(m.group(4))
            current["val_mPQ"] = float(m.group(5))

        m = lr_re.search(line)
        if m:
            current["old_lr"] = float(m.group(1))
            current["new_lr"] = float(m.group(2))

        if best_re.search(line):
            current["new_best"] = 1

    if current_epoch is not None and current:
        current["run_name"] = run_name
        current["epoch"] = current_epoch
        rows.append(current)

    return rows


def parse_test_metrics(text):
    patterns = {
        "test_Dice": r"Binary-Cell-Dice-Mean:\s*([0-9.]+)",
        "test_Jaccard": r"Binary-Cell-Jacard-Mean:\s*([0-9.]+)",
        "test_bPQ": r"\nbPQ:\s*([0-9.]+)",
        "test_mPQ": r"\nmPQ:\s*([0-9.]+)",
        "test_mDQ": r"\nmDQ:\s*([0-9.]+)",
        "test_mSQ": r"\nmSQ:\s*([0-9.]+)",
        "test_F1": r"f1_detection:\s*([0-9.]+)",
        "test_precision": r"precision_detection:\s*([0-9.]+)",
        "test_recall": r"recall_detection:\s*([0-9.]+)",
    }

    out = {}
    for key, pat in patterns.items():
        m = re.search(pat, text)
        out[key] = m.group(1) if m else ""
    return out


def infer_status(log_text, test_metrics):
    lower = log_text.lower()
    if test_metrics.get("test_mPQ"):
        return "completed_with_test"
    if "finished run" in lower:
        return "training_finished_no_test"
    if "traceback" in lower or "keyerror" in lower or "error" in lower:
        return "failed"
    if "epoch:" in lower:
        return "partial_or_timeout"
    return "unknown"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-root", default="run")
    parser.add_argument("--out-summary", default="reports/runs_summary.csv")
    parser.add_argument("--out-epochs", default="reports/epoch_metrics.csv")
    args = parser.parse_args()

    run_root = Path(args.run_root)
    summary_rows = []
    epoch_rows = []

    for config_path in sorted(run_root.rglob("config.yaml")):
        run_dir = config_path.parent
        log_path = run_dir / "logs.log"

        try:
            config = yaml.safe_load(config_path.read_text()) or {}
        except Exception as e:
            print(f"Skipping config with YAML error: {config_path} ({e})")
            continue

        log_text = log_path.read_text(errors="ignore") if log_path.exists() else ""

        run_name = get_nested(config, ["logging", "log_comment"], default=run_dir.name)
        adapter_type = get_nested(config, ["adapters", "adapter_type"])

        lora_conf = get_nested(config, ["adapters", "lora"], default={})
        adapt_conf = get_nested(config, ["adapters", "adaptformer"], default={})

        trainable = parse_trainable_report(log_text)
        epochs = parse_epoch_metrics(log_text, run_name)
        epoch_rows.extend(epochs)

        test_metrics = parse_test_metrics(log_text)
        status = infer_status(log_text, test_metrics)

        best_val_mPQ = ""
        best_val_bPQ = ""
        best_val_Dice = ""

        val_epochs = [r for r in epochs if "val_mPQ" in r]
        if val_epochs:
            best = max(val_epochs, key=lambda r: r.get("val_mPQ", -1))
            best_val_mPQ = best.get("val_mPQ", "")
            best_val_bPQ = best.get("val_bPQ", "")
            best_val_Dice = best.get("val_Dice", "")

        row = {
            "run_name": run_name,
            "status": status,
            "dataset": get_nested(config, ["data", "dataset"]),
            "dataset_path": get_nested(config, ["data", "dataset_path"]),
            "magnification": get_nested(config, ["data", "magnification"]),
            "tissue": ",".join(get_nested(config, ["dataset_config", "tissue_types"], default={}).keys()),
            "label_mode": "9class",
            "num_classes": get_nested(config, ["data", "num_nuclei_classes"]),
            "split": "slide-level",
            "train_slides": "",
            "valid_slides": "",
            "test_slides": "",
            "max_patches_per_slide": "",
            "pretrained_checkpoint": get_nested(config, ["model", "pretrained"]),

            "adapter_type": adapter_type,
            "rank": lora_conf.get("rank", ""),
            "alpha": lora_conf.get("alpha", ""),
            "targets": ",".join(lora_conf.get("targets", [])) if isinstance(lora_conf.get("targets", []), list) else lora_conf.get("targets", ""),
            "dropout": lora_conf.get("dropout", ""),
            "reduction": adapt_conf.get("reduction", adapt_conf.get("reduction_factor", "")) if isinstance(adapt_conf, dict) else "",
            "activation": adapt_conf.get("activation", "") if isinstance(adapt_conf, dict) else "",

            "epochs": get_nested(config, ["training", "epochs"]),
            "seed": get_nested(config, ["random_seed"]),
            "batch_size": get_nested(config, ["training", "batch_size"]),
            "lr": get_nested(config, ["training", "optimizer_hyperparameter", "lr"]),
            "unfreeze_encoder": get_nested(config, ["training", "unfreeze_encoder"]),
            "unfreeze_epoch": get_nested(config, ["training", "unfreeze_epoch"]),
            "trainable_params": trainable["trainable_params"],
            "trainable_ratio_percent": trainable["trainable_ratio_percent"],

            "best_val_mPQ": best_val_mPQ,
            "best_val_bPQ": best_val_bPQ,
            "best_val_Dice": best_val_Dice,

            **test_metrics,

            "config_path": str(config_path),
            "log_path": str(log_path) if log_path.exists() else "",
            "run_dir": str(run_dir),
            "notes": get_nested(config, ["logging", "notes"]),
        }

        summary_rows.append(row)

    summary_fields = [
        "run_name", "status", "dataset", "dataset_path", "magnification", "tissue",
        "label_mode", "num_classes", "split", "train_slides", "valid_slides",
        "test_slides", "max_patches_per_slide", "pretrained_checkpoint",
        "adapter_type", "rank", "alpha", "targets", "dropout", "reduction", "activation",
        "epochs", "seed", "batch_size", "lr", "unfreeze_encoder", "unfreeze_epoch",
        "trainable_params", "trainable_ratio_percent",
        "best_val_mPQ", "best_val_bPQ", "best_val_Dice",
        "test_mPQ", "test_bPQ", "test_Dice", "test_Jaccard", "test_F1",
        "test_precision", "test_recall", "test_mDQ", "test_mSQ",
        "config_path", "log_path", "run_dir", "notes",
    ]

    epoch_fields = [
        "run_name", "epoch", "max_epochs",
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
        for row in epoch_rows:
            writer.writerow({k: row.get(k, "") for k in epoch_fields})

    print(f"Wrote {len(summary_rows)} runs to {args.out_summary}")
    print(f"Wrote {len(epoch_rows)} epoch rows to {args.out_epochs}")


if __name__ == "__main__":
    main()