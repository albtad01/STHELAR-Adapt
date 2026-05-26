#!/usr/bin/env python3
"""
Utilities to report total/trainable parameters of a PyTorch model.

Recommended use in STHELAR-Adapt:
1) Import this utility inside the code path where the CellViT model has already been
   created, pretrained weights have been loaded, adapters have been inserted, and
   freezing rules have been applied.
2) Call write_model_param_report(model, experiment_name, adapter_type, output_path).

This is more reliable than trying to reconstruct trainable parameters only from
inference.log, because requires_grad depends on the exact adapter/freezing logic
executed in the experiment code.

Example hook inside experiment_cellvit_pannuke.py, after adapter insertion/freezing:

from pathlib import Path
from utils.model_param_report import write_model_param_report

write_model_param_report(
    model=model,
    experiment_name=self.run_name if hasattr(self, "run_name") else "unknown",
    adapter_type=adapter_type,
    output_path=Path(self.logdir) / "trainable_params_report.txt",
)

If self.logdir / self.run_name do not exist in your class, replace them with the
appropriate local variables already used for the run/log folder.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Optional

try:
    import torch
    import torch.nn as nn
except Exception as exc:
    raise RuntimeError(
        "This script requires PyTorch. Run it inside the same environment used for CellViT."
    ) from exc


def format_count(n: int) -> str:
    return f"{n:,}"


def summarize_trainable_parameters(
    model: nn.Module,
    max_modules: int = 200,
) -> dict[str, object]:
    total_params = 0
    trainable_params = 0
    trainable_tensors: list[dict[str, object]] = []

    for name, param in model.named_parameters():
        n = param.numel()
        total_params += n
        if param.requires_grad:
            trainable_params += n
            trainable_tensors.append(
                {
                    "name": name,
                    "shape": tuple(param.shape),
                    "num_params": n,
                }
            )

    ratio = 100.0 * trainable_params / total_params if total_params > 0 else 0.0

    prefixes: list[str] = []
    seen = set()
    for item in trainable_tensors:
        name = str(item["name"])
        parts = name.split(".")
        if len(parts) >= 3:
            prefix = ".".join(parts[:3])
        elif len(parts) >= 2:
            prefix = ".".join(parts[:2])
        else:
            prefix = parts[0]

        if prefix not in seen:
            prefixes.append(prefix)
            seen.add(prefix)

    return {
        "total_params": total_params,
        "trainable_params": trainable_params,
        "trainable_ratio_percent": ratio,
        "num_trainable_tensors": len(trainable_tensors),
        "trainable_tensors": trainable_tensors[:max_modules],
        "trainable_module_prefixes": prefixes[:max_modules],
        "num_trainable_tensors_total": len(trainable_tensors),
        "num_trainable_module_prefixes_total": len(prefixes),
    }


def render_param_report(
    summary: dict[str, object],
    experiment_name: str,
    adapter_type: str,
) -> str:
    total_params = int(summary["total_params"])
    trainable_params = int(summary["trainable_params"])
    ratio = float(summary["trainable_ratio_percent"])

    lines: list[str] = []
    lines.append("===== MODEL PARAMETER REPORT =====")
    lines.append(f"experiment_name: {experiment_name}")
    lines.append(f"adapter_type: {adapter_type}")
    lines.append(f"total_params: {format_count(total_params)}")
    lines.append(f"trainable_params: {format_count(trainable_params)}")
    lines.append(f"trainable_ratio_percent: {ratio:.6f}")
    lines.append(f"num_trainable_tensors: {summary['num_trainable_tensors_total']}")
    lines.append(f"num_trainable_module_prefixes: {summary['num_trainable_module_prefixes_total']}")
    lines.append("")

    lines.append("----- Trainable module prefixes -----")
    prefixes = summary["trainable_module_prefixes"]
    if prefixes:
        for prefix in prefixes:
            lines.append(str(prefix))
    else:
        lines.append("(none)")
    lines.append("")

    lines.append("----- Trainable parameter tensors -----")
    tensors = summary["trainable_tensors"]
    if tensors:
        for item in tensors:
            lines.append(
                f"{item['name']:90s} "
                f"shape={str(item['shape']):25s} "
                f"params={format_count(int(item['num_params']))}"
            )
    else:
        lines.append("(none)")

    return "\n".join(lines)


def write_model_param_report(
    model: nn.Module,
    experiment_name: str,
    adapter_type: str,
    output_path: Optional[Path | str] = None,
    max_modules: int = 200,
    print_to_stdout: bool = True,
) -> dict[str, object]:
    summary = summarize_trainable_parameters(model, max_modules=max_modules)
    report = render_param_report(summary, experiment_name, adapter_type)

    if print_to_stdout:
        print(report)

    if output_path is not None:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(report)
        print(f"Saved parameter report to: {output_path}")

    return summary


def inspect_checkpoint_params(checkpoint_path: Path | str) -> None:
    checkpoint_path = Path(checkpoint_path)
    ckpt = torch.load(checkpoint_path, map_location="cpu")

    if isinstance(ckpt, dict):
        if "model_state_dict" in ckpt:
            state = ckpt["model_state_dict"]
        elif "state_dict" in ckpt:
            state = ckpt["state_dict"]
        elif "model" in ckpt and isinstance(ckpt["model"], dict):
            state = ckpt["model"]
        else:
            state = ckpt
    else:
        raise TypeError(f"Unsupported checkpoint type: {type(ckpt)}")

    tensor_items = {k: v for k, v in state.items() if hasattr(v, "numel")}
    total = sum(v.numel() for v in tensor_items.values())

    print("===== CHECKPOINT PARAMETER COUNT =====")
    print(f"checkpoint_path: {checkpoint_path}")
    print(f"num_tensors: {len(tensor_items)}")
    print(f"stored_params: {format_count(total)}")
    print("")
    print("WARNING: this does not tell you trainable_params.")
    print("requires_grad is not stored in a normal state_dict.")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--checkpoint",
        type=str,
        default=None,
        help=(
            "Optional fallback mode: count tensors stored in a checkpoint. "
            "This does NOT report trainable parameters."
        ),
    )
    args = parser.parse_args()

    if args.checkpoint is None:
        print(__doc__)
        print("\nNo standalone model was provided.")
        print(
            "Use write_model_param_report(model, ...) inside the CellViT experiment "
            "after adapters/freezing have been applied."
        )
        return

    inspect_checkpoint_params(args.checkpoint)


if __name__ == "__main__":
    main()
