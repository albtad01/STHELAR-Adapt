#!/usr/bin/env python3
"""
Collect CellViT/STHELAR inference metrics from inference.log files.

Example:
python utils/collect_inference_results.py \
  --run-root run \
  --output-csv run/summary_inference_results.csv
"""

from __future__ import annotations

import argparse
import csv
import re
from pathlib import Path
from typing import Optional


METRIC_PATTERNS = {
    "dice": r"Binary-Cell-Dice-Mean:\s*([0-9.eE+-]+)",
    "jaccard": r"Binary-Cell-Jacard-Mean:\s*([0-9.eE+-]+)",
    "bPQ": r"bPQ:\s*([0-9.eE+-]+)",
    "bDQ": r"bDQ:\s*([0-9.eE+-]+)",
    "bSQ": r"bSQ:\s*([0-9.eE+-]+)",
    "mPQ": r"mPQ:\s*([0-9.eE+-]+)",
    "mDQ": r"mDQ:\s*([0-9.eE+-]+)",
    "mSQ": r"mSQ:\s*([0-9.eE+-]+)",
    "f1_detection": r"f1_detection:\s*([0-9.eE+-]+)",
    "precision_detection": r"precision_detection:\s*([0-9.eE+-]+)",
    "recall_detection": r"recall_detection:\s*([0-9.eE+-]+)",
    "tissue_accuracy": r"Tissue-Multiclass-Accuracy:\s*([0-9.eE+-]+)",
}


def parse_float(pattern: str, text: str) -> Optional[float]:
    match = re.search(pattern, text)
    if match is None:
        return None
    return float(match.group(1))


def infer_adapter_from_text_or_name(text: str, run_name: str) -> str:
    lower_text = text.lower()
    lower_name = run_name.lower()

    if "_ntonly_" in lower_name or lower_name.endswith("_ntonly"):
        return "NTonly"
    if "_freeze_" in lower_name or lower_name.endswith("_freeze"):
        return "freeze"
    if "_all_" in lower_name or lower_name.endswith("_all"):
        return "all"

    if "adding adapters: lora" in lower_text or "_lora_" in lower_name or lower_name.endswith("_lora"):
        return "lora"
    if "adding adapters: adaptformer" in lower_text or "_adaptformer_" in lower_name or lower_name.endswith("_adaptformer"):
        return "adaptformer"
    if "adding adapters: bottleneck" in lower_text or "_bottleneck_" in lower_name:
        return "bottleneck"
    if "adding adapters: plora" in lower_text or "_plora_" in lower_name:
        return "plora"

    if "no adapters added" in lower_text:
        return "none_or_all"

    return "unknown"


def find_config_path(log_path: Path) -> Optional[Path]:
    candidate = log_path.parent / "config.yaml"
    if candidate.exists():
        return candidate
    return None


def extract_dataset_path_from_config(config_path: Optional[Path]) -> Optional[str]:
    if config_path is None:
        return None

    text = config_path.read_text(errors="replace")
    match = re.search(r"dataset_path:\s*[\"']?([^\"'\n]+)[\"']?", text)
    if match:
        return match.group(1).strip()
    return None


def collect_results(run_root: Path) -> list[dict[str, object]]:
    logs = sorted(run_root.glob("**/inference.log"))
    rows = []

    for log_path in logs:
        text = log_path.read_text(errors="replace")

        # Expected structure:
        # run/<experiment_name>/log/<timestamped_run>/inference.log
        timestamped_run = log_path.parent.name
        experiment_name = log_path.parent.parent.parent.name if len(log_path.parts) >= 4 else "unknown"

        config_path = find_config_path(log_path)
        dataset_path = extract_dataset_path_from_config(config_path)

        row: dict[str, object] = {
            "experiment_name": experiment_name,
            "timestamped_run": timestamped_run,
            "adapter": infer_adapter_from_text_or_name(text, experiment_name),
            "log_path": str(log_path),
            "config_path": str(config_path) if config_path else "",
            "dataset_path": dataset_path or "",
        }

        for metric_name, pattern in METRIC_PATTERNS.items():
            row[metric_name] = parse_float(pattern, text)

        rows.append(row)

    return rows


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--run-root",
        type=str,
        default="run",
        help="Root directory containing CellViT run folders.",
    )
    parser.add_argument(
        "--output-csv",
        type=str,
        default=None,
        help="Optional CSV output path.",
    )
    args = parser.parse_args()

    run_root = Path(args.run_root).expanduser().resolve()
    rows = collect_results(run_root)

    if len(rows) == 0:
        print(f"No inference.log files found under: {run_root}")
        return

    fieldnames = [
        "experiment_name",
        "timestamped_run",
        "adapter",
        "dice",
        "jaccard",
        "bPQ",
        "bDQ",
        "bSQ",
        "mPQ",
        "mDQ",
        "mSQ",
        "f1_detection",
        "precision_detection",
        "recall_detection",
        "tissue_accuracy",
        "dataset_path",
        "log_path",
        "config_path",
    ]

    print("\nCollected inference results:\n")
    print(
        f"{'experiment_name':55s} {'adapter':12s} "
        f"{'Dice':>8s} {'bPQ':>8s} {'mPQ':>8s} {'F1':>8s} {'Prec':>8s} {'Rec':>8s}"
    )
    print("-" * 125)

    for row in rows:
        print(
            f"{str(row['experiment_name'])[:55]:55s} "
            f"{str(row['adapter'])[:12]:12s} "
            f"{row.get('dice') if row.get('dice') is not None else float('nan'):8.4f} "
            f"{row.get('bPQ') if row.get('bPQ') is not None else float('nan'):8.4f} "
            f"{row.get('mPQ') if row.get('mPQ') is not None else float('nan'):8.4f} "
            f"{row.get('f1_detection') if row.get('f1_detection') is not None else float('nan'):8.4f} "
            f"{row.get('precision_detection') if row.get('precision_detection') is not None else float('nan'):8.4f} "
            f"{row.get('recall_detection') if row.get('recall_detection') is not None else float('nan'):8.4f}"
        )

    if args.output_csv:
        output_csv = Path(args.output_csv).expanduser().resolve()
        output_csv.parent.mkdir(parents=True, exist_ok=True)

        with open(output_csv, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            for row in rows:
                writer.writerow({k: row.get(k, "") for k in fieldnames})

        print(f"\nSaved CSV summary to: {output_csv}")


if __name__ == "__main__":
    main()