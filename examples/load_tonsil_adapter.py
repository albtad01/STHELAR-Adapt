#!/usr/bin/env python3
"""Load a Tonsil 9-class adapter on CellViT-SAM-H-x40."""

import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from utils.cellvit_adapter_hub import add_tonsil_adapter, load_cellvit_base


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--base-checkpoint",
        default="models/pretrained/CellViT-SAM-H-x40.pth",
    )
    parser.add_argument("--adapter-checkpoint", required=True)
    parser.add_argument(
        "--config",
        default="configs/examples/training_sthelar40x_tonsil_9class_slide_lora_adaptformer_r8_a8_red16_decoder_conv_adapters_lr5e-5_e10_seed42_CLEAN.yaml",
    )
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()

    cellvit = load_cellvit_base(
        model_name="cellvit-sam-h-x40",
        base_checkpoint=args.base_checkpoint,
        config_path=args.config,
        device=args.device,
    )
    add_tonsil_adapter(cellvit, args.adapter_checkpoint)
    print("Tonsil adapter is ready for inference.")


if __name__ == "__main__":
    main()
