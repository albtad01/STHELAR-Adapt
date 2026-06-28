#!/usr/bin/env python3

from __future__ import annotations

import argparse
from copy import deepcopy
from pathlib import Path

import yaml


DATASET_PATH = "/gpfs/workdir/taddeial/workspace/Datasets/cellvit_ready/sthelar40x_tonsil_9class_slide"
PROJECT_NAME = "sthelar40x_tonsil_9class_adapters"


def load_yaml(path: Path) -> dict:
    with open(path, "r") as f:
        return yaml.safe_load(f)


def save_yaml(cfg: dict, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        yaml.safe_dump(cfg, f, sort_keys=False)


def common_update(cfg: dict, run_name: str, epochs: int) -> dict:
    cfg = deepcopy(cfg)

    cfg.setdefault("logging", {})
    cfg["logging"]["project"] = PROJECT_NAME
    cfg["logging"]["log_comment"] = run_name
    cfg["logging"]["wandb_dir"] = f"run/{run_name}/wandb"
    cfg["logging"]["log_dir"] = f"run/{run_name}/log"

    cfg["logging"]["notes"] = (
        f"CellViT-SAM-H x40 on STHELAR 40x tonsil 9-class slide-level split. "
        f"Run: {run_name}."
    )

    cfg["logging"]["tags"] = [
        "sthelar",
        "40x",
        "tonsil",
        "single_tissue",
        "slide_level",
        "9class",
        "pretrained",
        "e5" if epochs == 5 else f"e{epochs}",
    ]

    cfg.setdefault("data", {})
    cfg["data"]["dataset"] = "STHELAR"
    cfg["data"]["dataset_path"] = DATASET_PATH
    cfg["data"]["train_folds"] = ["train"]
    cfg["data"]["val_folds"] = ["valid"]
    cfg["data"]["test_folds"] = ["test"]
    cfg["data"]["num_nuclei_classes"] = 10
    cfg["data"]["num_tissue_classes"] = 1
    cfg["data"]["input_shape"] = 256
    cfg["data"]["magnification"] = 40

    cfg.setdefault("training", {})
    cfg["training"]["epochs"] = epochs
    cfg["training"]["unfreeze_encoder"] = False
    cfg["training"]["unfreeze_epoch"] = 999
    cfg["training"]["optimizer"] = "AdamW"
    cfg["training"].setdefault("optimizer_hyperparameter", {})
    cfg["training"]["optimizer_hyperparameter"]["lr"] = 5.0e-5
    cfg["training"]["optimizer_hyperparameter"]["betas"] = [0.85, 0.85]

    cfg["eval_checkpoint"] = "latest_checkpoint.pth"

    return cfg


def make_freeze(template: dict, epochs: int) -> tuple[str, dict]:
    run_name = f"sthelar40x_tonsil_9class_slide_freeze_e{epochs}"
    cfg = common_update(template, run_name, epochs)

    cfg["adapters"] = {"adapter_type": "freeze"}
    cfg["logging"]["tags"].append("freeze")

    return run_name, cfg


def make_lora_ntonly_r4(template: dict, epochs: int) -> tuple[str, dict]:
    run_name = f"sthelar40x_tonsil_9class_slide_lora_ntonly_r4_a4_lr5e-5_e{epochs}"
    cfg = common_update(template, run_name, epochs)

    cfg["adapters"] = {
        "adapter_type": "lora_ntonly",
        "lora": {
            "rank": 4,
            "alpha": 4,
            "targets": ["q", "v"],
            "dropout": 0.0,
        },
    }

    cfg["logging"]["tags"] += ["lora_ntonly", "rank4", "alpha4", "lr5e-5"]

    return run_name, cfg


def make_lora_r8(template: dict, epochs: int) -> tuple[str, dict]:
    run_name = f"sthelar40x_tonsil_9class_slide_lora_r8_a8_lr5e-5_e{epochs}"
    cfg = common_update(template, run_name, epochs)

    cfg["adapters"] = {
        "adapter_type": "lora",
        "lora": {
            "rank": 8,
            "alpha": 8,
            "targets": ["q", "v"],
            "dropout": 0.0,
        },
    }

    cfg["logging"]["tags"] += ["lora", "rank8", "alpha8", "lr5e-5"]

    return run_name, cfg


def make_adaptformer(template: dict, epochs: int) -> tuple[str, dict]:
    run_name = f"sthelar40x_tonsil_9class_slide_adaptformer_gelu_red16_lr5e-5_e{epochs}"
    cfg = common_update(template, run_name, epochs)

    cfg["adapters"] = {
        "adapter_type": "adaptformer",
        "adaptformer": {
            "activation": "GELU",
            "reduction": 16,
        },
    }

    cfg["logging"]["tags"] += ["adaptformer", "gelu", "red16", "lr5e-5"]

    return run_name, cfg


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--template",
        type=str,
        default="configs/examples/training_sthelar40x_bps_9class_slide_lora_r8_a8_lr5e-5_clean_e3.yaml",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="configs/examples",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=5,
    )
    parser.add_argument(
        "--make-debug",
        action="store_true",
    )
    args = parser.parse_args()

    template_path = Path(args.template)
    output_dir = Path(args.output_dir)

    template = load_yaml(template_path)

    makers = [
        make_freeze,
        make_lora_ntonly_r4,
        make_lora_r8,
        make_adaptformer,
    ]

    generated = []

    if args.make_debug:
        debug_run_name, debug_cfg = make_lora_ntonly_r4(template, epochs=1)
        debug_run_name = debug_run_name.replace("_e1", "_debug_e1")
        debug_cfg = common_update(debug_cfg, debug_run_name, epochs=1)
        debug_cfg["adapters"] = {
            "adapter_type": "lora_ntonly",
            "lora": {
                "rank": 4,
                "alpha": 4,
                "targets": ["q", "v"],
                "dropout": 0.0,
            },
        }
        debug_cfg["logging"]["tags"] += ["debug", "lora_ntonly", "rank4", "alpha4"]

        debug_path = output_dir / f"training_{debug_run_name}.yaml"
        save_yaml(debug_cfg, debug_path)
        generated.append(debug_path)

    for maker in makers:
        run_name, cfg = maker(template, args.epochs)
        out_path = output_dir / f"training_{run_name}.yaml"
        save_yaml(cfg, out_path)
        generated.append(out_path)

    launch_script = output_dir / "launch_tonsil_adapter_runs.sh"
    with open(launch_script, "w") as f:
        f.write("#!/bin/bash\n")
        f.write("set -euo pipefail\n\n")
        for path in generated:
            if "debug" in path.name:
                continue
            f.write(f'sbatch ruche/slurm_train.sh "{path}"\n')

    print("Generated YAML files:")
    for path in generated:
        print(f"  {path}")

    print("\nGenerated launch script:")
    print(f"  {launch_script}")

    if args.make_debug:
        debug_files = [p for p in generated if "debug" in p.name]
        if debug_files:
            print("\nRecommended first smoke test:")
            print(f'  sbatch ruche/slurm_train.sh "{debug_files[0]}"')


if __name__ == "__main__":
    main()
