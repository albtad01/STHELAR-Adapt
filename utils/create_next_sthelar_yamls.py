import os
from pathlib import Path
import yaml


OUT = Path("configs/examples")
OUT.mkdir(parents=True, exist_ok=True)
DATA_ROOT = os.environ.get("DATA_ROOT", "${DATA_ROOT}")
STHELAR_ROOT = os.environ.get("STHELAR_ROOT", "${STHELAR_ROOT}")


def write_yaml(path, data):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as f:
        yaml.safe_dump(data, f, sort_keys=False)
    print(f"Wrote {path}")


def load_yaml(path):
    with open(path, "r") as f:
        return yaml.safe_load(f)


def common_transformations():
    return {
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
    }


def base_training_config(
    run_name,
    project,
    notes,
    dataset_path,
    adapter_block,
    num_nuclei_classes,
    seed=42,
    batch_size=4,
    epochs=10,
    lr=5.0e-5,
):
    return {
        "adapters": adapter_block,
        "logging": {
            "mode": "offline",
            "project": project,
            "notes": notes,
            "log_comment": run_name,
            "tags": run_name.split("_"),
            "wandb_dir": f"run/{run_name}/wandb",
            "log_dir": f"run/{run_name}/log",
            "level": "debug",
            "log_images": False,
        },
        "random_seed": seed,
        "gpu": 0,
        "data": {
            "dataset": "STHELAR",
            "dataset_path": dataset_path,
            "train_folds": ["train"],
            "val_folds": ["valid"],
            "test_folds": ["test"],
            "num_nuclei_classes": num_nuclei_classes,
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
            "batch_size": batch_size,
            "epochs": epochs,
            "unfreeze_epoch": 999,
            "optimizer": "AdamW",
            "optimizer_hyperparameter": {
                "lr": lr,
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
            "unfreeze_encoder": False,
        },
        "transformations": common_transformations(),
        "eval_checkpoint": "latest_checkpoint.pth",
    }


# ============================================================
# 1. Liver 5-class spatial preprocessing
# ============================================================

preprocess_liver_5class_spatial = {
    "sthelar_root": STHELAR_ROOT,
    "output_root": f"{DATA_ROOT}/sthelar40x_liver_5class_spatial",
    "tissue_name": "Liver",
    "tissue": "liver",
    "slide_ids": ["liver_s0", "liver_s1"],
    "strategy": "spatial",

    # For spatial strategy, the converter should split inside selected slides.
    # These fields are kept for compatibility/metadata.
    "train_slides": ["liver_s0", "liver_s1"],
    "valid_slides": [],
    "test_slides": [],

    "split_axis": "x",
    "boundary_margin": 64,
    "train_frac": 0.70,
    "valid_frac": 0.15,
    "test_frac": 0.15,
    "train_frac_inside_train_slide": 0.85,
    "valid_frac_inside_train_slide": 0.15,
    "max_patches_per_slide": None,
    "max_per_split": None,
    "random_seed": 42,
    "overwrite": True,
    "keep_tmp": False,

    "label_mode": "5class",
    "label_column": "cells_final_label_group",
    "ignore_labels": None,

    "nuclei_types": {
        "Background": 0,
        "Immune": 1,
        "Stromal": 2,
        "Epithelial": 3,
        "Melanocyte": 4,
        "Other": 5,
    },

    "label_mapping": {
        "T_NK": "Immune",
        "B_Plasma": "Immune",
        "Myeloid": "Immune",
        "Blood_vessel": "Stromal",
        "Fibroblast_Myofibroblast": "Stromal",
        "Epithelial": "Epithelial",
        "Melanocyte": "Melanocyte",
        "Specialized": "Other",
        "Other": "Other",
    },

    "fallback_class": "Other",
}

write_yaml(
    OUT / "preprocessing_sthelar40x_liver_5class_spatial.yaml",
    preprocess_liver_5class_spatial,
)


# ============================================================
# 2. Liver 5-class spatial AdaptFormer e10
# ============================================================

liver5_dataset = f"{DATA_ROOT}/sthelar40x_liver_5class_spatial"

run_name = "sthelar40x_liver_5class_spatial_adaptformer_gelu_red16_lr5e-5_e10_seed42_CLEAN"

cfg = base_training_config(
    run_name=run_name,
    project="sthelar40x_liver_5class_adapters",
    notes=(
        "CellViT-SAM-H x40 with AdaptFormer GELU red16 on STHELAR 40x "
        "liver 5-class spatial split. "
        f"Run: {run_name}."
    ),
    dataset_path=liver5_dataset,
    adapter_block={
        "adapter_type": "adaptformer",
        "adaptformer": {
            "activation": "GELU",
            "reduction": 16,
        },
    },
    num_nuclei_classes=6,
    seed=42,
)

write_yaml(
    OUT / "training_sthelar40x_liver_5class_spatial_adaptformer_gelu_red16_lr5e-5_e10_seed42_CLEAN.yaml",
    cfg,
)


# ============================================================
# 3. Liver 5-class spatial LoRA r8 e10
# ============================================================

run_name = "sthelar40x_liver_5class_spatial_lora_r8_a8_lr5e-5_e10_seed42_CLEAN"

cfg = base_training_config(
    run_name=run_name,
    project="sthelar40x_liver_5class_adapters",
    notes=(
        "CellViT-SAM-H x40 with LoRA r8 alpha8 on STHELAR 40x "
        "liver 5-class spatial split. "
        f"Run: {run_name}."
    ),
    dataset_path=liver5_dataset,
    adapter_block={
        "adapter_type": "lora",
        "lora": {
            "rank": 8,
            "alpha": 8,
            "targets": ["q", "v"],
            "dropout": 0.0,
        },
    },
    num_nuclei_classes=6,
    seed=42,
)

write_yaml(
    OUT / "training_sthelar40x_liver_5class_spatial_lora_r8_a8_lr5e-5_e10_seed42_CLEAN.yaml",
    cfg,
)


# ============================================================
# 4. Tonsil 9-class AdaptFormer red8 e10 seed42
# ============================================================

run_name = "sthelar40x_tonsil_9class_slide_adaptformer_gelu_red8_lr5e-5_e10_seed42_CLEAN"

cfg = base_training_config(
    run_name=run_name,
    project="sthelar40x_tonsil_9class_adapters",
    notes=(
        "CellViT-SAM-H x40 with AdaptFormer GELU red8 on STHELAR 40x "
        "tonsil 9-class slide-level split. "
        f"Run: {run_name}."
    ),
    dataset_path=f"{DATA_ROOT}/sthelar40x_tonsil_9class_slide",
    adapter_block={
        "adapter_type": "adaptformer",
        "adaptformer": {
            "activation": "GELU",
            "reduction": 8,
        },
    },
    num_nuclei_classes=10,
    seed=42,
)

write_yaml(
    OUT / "training_sthelar40x_tonsil_9class_slide_adaptformer_gelu_red8_lr5e-5_e10_seed42_CLEAN.yaml",
    cfg,
)


# ============================================================
# 5. Tonsil 9-class full fine-tuning attempt e10 seed42
# ============================================================

run_name = "sthelar40x_tonsil_9class_slide_fullft_lr5e-5_e10_seed42_CLEAN"

cfg = base_training_config(
    run_name=run_name,
    project="sthelar40x_tonsil_9class_adapters",
    notes=(
        "CellViT-SAM-H x40 full fine-tuning attempt on STHELAR 40x "
        "tonsil 9-class slide-level split. "
        f"Run: {run_name}. IMPORTANT: verify trainable params; true full fine-tuning should unfreeze the encoder."
    ),
    dataset_path=f"{DATA_ROOT}/sthelar40x_tonsil_9class_slide",
    adapter_block={
        "adapter_type": "all",
    },
    num_nuclei_classes=10,
    seed=42,
)

cfg["training"]["unfreeze_encoder"] = True
cfg["training"]["unfreeze_epoch"] = 0

write_yaml(
    OUT / "training_sthelar40x_tonsil_9class_slide_fullft_lr5e-5_e10_seed42_CLEAN.yaml",
    cfg,
)


# ============================================================
# 6. Tonsil 9-class frozen baseline e10 seed42
# ============================================================

run_name = "sthelar40x_tonsil_9class_slide_freeze_e10_seed42_CLEAN"

cfg = base_training_config(
    run_name=run_name,
    project="sthelar40x_tonsil_9class_adapters",
    notes=(
        "Frozen CellViT-SAM-H x40 baseline on STHELAR 40x "
        "tonsil 9-class slide-level split. "
        f"Run: {run_name}."
    ),
    dataset_path=f"{DATA_ROOT}/sthelar40x_tonsil_9class_slide",
    adapter_block={
        "adapter_type": "freeze",
    },
    num_nuclei_classes=10,
    seed=42,
)

cfg["training"]["epochs"] = 10
cfg["training"]["unfreeze_encoder"] = False
cfg["training"]["unfreeze_epoch"] = 999

write_yaml(
    OUT / "training_sthelar40x_tonsil_9class_slide_freeze_e10_seed42_CLEAN.yaml",
    cfg,
)
