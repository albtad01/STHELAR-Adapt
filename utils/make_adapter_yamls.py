#!/usr/bin/env python3
"""
Generate the next STHELAR-Adapt LoRA YAML configs.

This script creates:
  - training_sthelar40x_bps_9class_slide_lora_r4_a4_lr5e-5_e3.yaml
  - training_sthelar40x_bps_9class_slide_lora_r8_a8_lr5e-5_e3_seed42.yaml
  - training_sthelar40x_bps_9class_slide_lora_r8_a8_lr2e-5_e3.yaml

Run from the STHELAR-Adapt repository root:

python utils/make_next_lora_yamls.py

Then launch:

for cfg in \
  configs/examples/training_sthelar40x_bps_9class_slide_lora_r4_a4_lr5e-5_e3.yaml \
  configs/examples/training_sthelar40x_bps_9class_slide_lora_r8_a8_lr5e-5_e3_seed42.yaml \
  configs/examples/training_sthelar40x_bps_9class_slide_lora_r8_a8_lr2e-5_e3.yaml
do
  sbatch ruche/slurm_train.sh "$cfg"
done
"""

from __future__ import annotations

from pathlib import Path
import copy
import yaml


OUT_DIR = Path("configs/examples")
OUT_DIR.mkdir(parents=True, exist_ok=True)

BASE_DATASET_PATH = (
    "/gpfs/workdir/taddeial/workspace/Datasets/"
    "cellvit_ready/sthelar40x_breast_pancreatic_skin_9class_slide"
)

BASE = {
    "adapters": {
        "adapter_type": "all",
    },
    "logging": {
        "mode": "offline",
        "project": "sthelar40x_bps_9class_adapters",
        "notes": "",
        "log_comment": "",
        "tags": [
            "sthelar",
            "40x",
            "multi_tissue",
            "slide_level",
            "9class",
            "pretrained",
        ],
        "wandb_dir": "",
        "log_dir": "",
        "level": "debug",
        "log_images": False,
    },
    "random_seed": 19,
    "gpu": 0,
    "data": {
        "dataset": "STHELAR",
        "dataset_path": BASE_DATASET_PATH,
        "train_folds": ["train"],
        "val_folds": ["valid"],
        "test_folds": ["test"],
        "num_nuclei_classes": 10,
        "num_tissue_classes": 1,
        "input_shape": 256,
        "magnification": 40,
    },
    "model": {
        "backbone": "SAM-H",
        "pretrained_encoder": None,
        "pretrained": "models/pretrained/CellViT-SAM-H-x40.pth",
        "embed_dim": 1280,
        "input_channels": 3,
        "depth": 32,
        "num_heads": 16,
        "extract_layers": 4,
        "shared_decoders": False,
    },
    "training": {
        "batch_size": 4,
        "epochs": 3,
        "unfreeze_epoch": 2,
        "optimizer": "AdamW",
        "optimizer_hyperparameter": {
            "lr": 0.0001,
            "betas": [0.85, 0.85],
        },
        "early_stopping_patience": 10,
        "scheduler": {
            "scheduler_type": "exponential",
        },
        "sampling_strategy": "cell",
        "sampling_gamma": 0.85,
        "mixed_precision": True,
        "eval_every": 1,
    },
    "transformations": {
        "randomrotate90": {"p": 0.5},
        "horizontalflip": {"p": 0.5},
        "verticalflip": {"p": 0.5},
        "downscale": {"p": 0.5, "scale": 0.2},
        "blur": {"p": 0.5, "blur_limit": 10},
        "gaussnoise": {"p": 0.5, "var_limit": 10},
        "colorjitter": {"p": 0.5, "scale_setting": 0.25, "scale_color": 0.1},
        "superpixels": {"p": 0.5},
        "zoomblur": {"p": 0.5},
        "randomsizedcrop": {"p": 0.5},
        "elastictransform": {"p": 0.5},
        "normalize": {
            "mean": [0.5, 0.5, 0.5],
            "std": [0.5, 0.5, 0.5],
        },
    },
    "eval_checkpoint": "latest_checkpoint.pth",
}


RUNS = [
    {
        "name": "sthelar40x_bps_9class_slide_lora_r4_a4_lr5e-5_e3",
        "adapter_type": "lora",
        "adapter_cfg": {"lora": {"rank": 4, "alpha": 4}},
        "seed": 19,
        "lr": 0.00005,
        "tags": ["lora", "rank4", "alpha4", "lr5e-5", "e3"],
        "notes": (
            "CellViT-SAM-H x40 on STHELAR 40x BPS 9-class slide-level split "
            "with LoRA rank 4 alpha 4 and lower learning rate."
        ),
    },
    {
        "name": "sthelar40x_bps_9class_slide_lora_r8_a8_lr5e-5_e3_seed42",
        "adapter_type": "lora",
        "adapter_cfg": {"lora": {"rank": 8, "alpha": 8}},
        "seed": 42,
        "lr": 0.00005,
        "tags": ["lora", "rank8", "alpha8", "lr5e-5", "e3", "seed42"],
        "notes": (
            "Seed-42 repeat of CellViT-SAM-H x40 on STHELAR 40x BPS 9-class "
            "slide-level split with LoRA rank 8 alpha 8 and lower learning rate."
        ),
    },
    {
        "name": "sthelar40x_bps_9class_slide_lora_r8_a8_lr2e-5_e3",
        "adapter_type": "lora",
        "adapter_cfg": {"lora": {"rank": 8, "alpha": 8}},
        "seed": 19,
        "lr": 0.00002,
        "tags": ["lora", "rank8", "alpha8", "lr2e-5", "e3"],
        "notes": (
            "CellViT-SAM-H x40 on STHELAR 40x BPS 9-class slide-level split "
            "with LoRA rank 8 alpha 8 and an even lower learning rate."
        ),
    },
]


def main() -> None:
    for run in RUNS:
        cfg = copy.deepcopy(BASE)

        cfg["adapters"] = {"adapter_type": run["adapter_type"]}
        cfg["adapters"].update(run["adapter_cfg"])

        cfg["logging"]["notes"] = run["notes"]
        cfg["logging"]["log_comment"] = run["name"]
        cfg["logging"]["tags"] = BASE["logging"]["tags"] + run["tags"]
        cfg["logging"]["wandb_dir"] = f"run/{run['name']}/wandb"
        cfg["logging"]["log_dir"] = f"run/{run['name']}/log"

        cfg["random_seed"] = run["seed"]
        cfg["training"]["optimizer_hyperparameter"]["lr"] = run["lr"]

        out_path = OUT_DIR / f"training_{run['name']}.yaml"
        with out_path.open("w") as f:
            yaml.safe_dump(cfg, f, sort_keys=False)

        print(f"Written {out_path}")


if __name__ == "__main__":
    main()
