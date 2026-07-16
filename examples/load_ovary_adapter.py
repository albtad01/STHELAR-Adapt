#!/usr/bin/env python3
"""Load the Ovary 5-class STHELAR adapter on CellViT-SAM-H-x40."""

import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

import hub


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--base-checkpoint",
        default="models/pretrained/CellViT-SAM-H-x40.pth",
        help="Path to the CellViT-SAM-H-x40 pretrained checkpoint.",
    )
    parser.add_argument(
        "--adapter",
        default="ovary_5",
        help=(
            "Adapter alias, local adapter .pth/.safetensors, HF-style adapter "
            "directory, or Hugging Face repo id."
        ),
    )
    parser.add_argument("--device", default="cpu")
    parser.add_argument(
        "--prefer-hf",
        action="store_true",
        help="Resolve registered aliases through Hugging Face instead of local files.",
    )
    parser.add_argument(
        "--print-metadata",
        action="store_true",
        help="Print adapter metadata after loading.",
    )
    args = parser.parse_args()

    cellvit = hub.model(
        "cellvit-sam-h-x40",
        base_checkpoint=args.base_checkpoint,
        device=args.device,
    )
    hub.load_sthelar_adapter(
        cellvit,
        args.adapter,
        prefer_hf=args.prefer_hf,
        print_metadata=args.print_metadata,
    )
    print(f"OK: {args.adapter} adapter loaded on CellViT-SAM-H-x40")


if __name__ == "__main__":
    main()
