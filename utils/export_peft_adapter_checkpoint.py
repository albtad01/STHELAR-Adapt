#!/usr/bin/env python3
"""Export a HF-ready lightweight CellViT PEFT adapter package from a full checkpoint."""

import argparse
import json
import shutil
import sys
from pathlib import Path

import torch
import yaml

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from utils.export_adapter_checkpoint import export_adapter_checkpoint


def _load_exported_payload(path: Path) -> dict:
    payload = torch.load(str(path), map_location="cpu")
    if not isinstance(payload, dict) or "metadata" not in payload:
        raise ValueError(f"Unsupported exported adapter checkpoint: {path}")
    return payload


def _write_safetensors(payload: dict, out_path: Path) -> bool:
    try:
        from safetensors.torch import save_file
    except ImportError:
        return False

    tensors = {}
    for section in ("adapter_state_dict", "mutable_buffer_state_dict"):
        prefix = section + "."
        for name, tensor in payload.get(section, {}).items():
            tensors[prefix + name] = tensor.detach().cpu()
    metadata = {
        "format": payload.get("format", "cellvit_adapter_checkpoint"),
        "format_version": str(payload.get("format_version", 1)),
        "metadata_json": json.dumps(payload["metadata"], sort_keys=True),
    }
    save_file(tensors, str(out_path), metadata=metadata)
    return True


def _copy_if_exists(src: Path, dst: Path) -> None:
    if src.is_file():
        shutil.copy2(src, dst)


def _write_label_mapping(dataset_config: dict, out_path: Path) -> None:
    label_mapping = {
        "nuclei_types": dataset_config.get("nuclei_types", {}),
        "tissue_types": dataset_config.get("tissue_types", {}),
    }
    out_path.write_text(yaml.safe_dump(label_mapping, sort_keys=False))


def _write_readme(out_dir: Path, metadata: dict, weight_name: str) -> None:
    run_name = metadata.get("run_name", "cellvit-peft-adapter")
    readme = f"""---
library_name: pytorch
tags:
  - cellvit
  - sthelar
  - peft
  - adapter
---

# {run_name}

Lightweight PEFT adapter weights for CellViT-SAM-H x40.

This repository is intended to contain only adapter/trainable weights, not the
full CellViT base checkpoint. The required base checkpoint is:

`{metadata.get("base_checkpoint", "models/pretrained/CellViT-SAM-H-x40.pth")}`

## Files

- `{weight_name}`: adapter weights and changed mutable buffers.
- `adapter_config.yaml`: adapter metadata and architecture settings.
- `training_config.yaml`: original training config.
- `dataset_config.yaml`, `label_mapping.yaml`, `types.csv`: label metadata.
- `inference_example.py`: minimal loading example.

## Loading

```bash
python inference_example.py \\
  --repo-dir . \\
  --base-checkpoint /path/to/CellViT-SAM-H-x40.pth
```

The loading example reconstructs the CellViT-SAM-H x40 architecture, inserts the
same PEFT modules, loads the base checkpoint, then applies adapter weights.
"""
    (out_dir / "README.md").write_text(readme)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", required=True, help="Run dir or timestamped log dir.")
    parser.add_argument(
        "--checkpoint",
        required=True,
        help="Full checkpoint path or filename under RUN_DIR/checkpoints.",
    )
    parser.add_argument("--out-dir", required=True, help="HF-style output directory.")
    parser.add_argument(
        "--format",
        choices=("auto", "safetensors", "pth"),
        default="auto",
        help="auto prefers safetensors when installed, otherwise writes .pth.",
    )
    parser.add_argument(
        "--base-checkpoint",
        default=None,
        help="Override model.pretrained from the training config.",
    )
    parser.add_argument("--verify-load", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    out_dir = Path(args.out_dir).expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    staging_pth = out_dir / ".adapter_model.export_tmp.pth"
    exported_pth = export_adapter_checkpoint(
        run_dir=args.run_dir,
        checkpoint_path=args.checkpoint,
        out_path=staging_pth,
        base_checkpoint=args.base_checkpoint,
        verify_load=args.verify_load,
    )
    payload = _load_exported_payload(exported_pth)
    metadata = payload["metadata"]

    weight_name = "adapter_model.pth"
    if args.format in {"auto", "safetensors"}:
        safetensors_path = out_dir / "adapter_model.safetensors"
        wrote_safetensors = _write_safetensors(payload, safetensors_path)
        if wrote_safetensors:
            weight_name = safetensors_path.name
            staging_pth.unlink(missing_ok=True)
        elif args.format == "safetensors":
            raise RuntimeError("safetensors is not installed; use --format auto or --format pth")

    if weight_name == "adapter_model.pth":
        final_pth = out_dir / weight_name
        if exported_pth != final_pth:
            exported_pth.replace(final_pth)

    metadata["adapter_weight_file"] = weight_name
    (out_dir / "adapter_config.yaml").write_text(
        yaml.safe_dump(metadata, sort_keys=False)
    )
    (out_dir / "adapter_metadata.json").write_text(
        json.dumps(metadata, indent=2, sort_keys=True) + "\n"
    )

    run_config = Path(metadata["original_config_path"])
    _copy_if_exists(run_config, out_dir / "training_config.yaml")

    dataset_path = Path(yaml.safe_load(run_config.read_text())["data"]["dataset_path"])
    dataset_config_path = dataset_path / "dataset_config.yaml"
    dataset_config = {}
    if dataset_config_path.is_file():
        dataset_config = yaml.safe_load(dataset_config_path.read_text()) or {}
        _copy_if_exists(dataset_config_path, out_dir / "dataset_config.yaml")
    _copy_if_exists(dataset_path / "types.csv", out_dir / "types.csv")
    if dataset_config:
        _write_label_mapping(dataset_config, out_dir / "label_mapping.yaml")

    shutil.copy2(REPO_ROOT / "utils" / "load_peft_adapter_example.py", out_dir / "inference_example.py")
    (out_dir / "requirements.txt").write_text(
        "torch\npyyaml\nnumpy\nsafetensors\n"
    )
    _write_readme(out_dir, metadata, weight_name)

    full_size = Path(args.checkpoint)
    if not full_size.is_absolute():
        candidate = Path(args.run_dir).expanduser().resolve() / "checkpoints" / args.checkpoint
        full_size = candidate if candidate.is_file() else full_size
    adapter_size = (out_dir / weight_name).stat().st_size
    if full_size.is_file():
        ratio = full_size.stat().st_size / adapter_size
        print(f"Full checkpoint: {full_size.stat().st_size / (1024 ** 3):.2f} GiB")
        print(f"Adapter weights: {adapter_size / (1024 ** 2):.2f} MiB")
        print(f"Size reduction: {ratio:.1f}x smaller")
    print(f"HF adapter package written to: {out_dir}")
    print(f"Adapter weight file: {weight_name}")


if __name__ == "__main__":
    main()
