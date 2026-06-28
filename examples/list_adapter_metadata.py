#!/usr/bin/env python3
"""Print metadata from a CellViT STHELAR adapter checkpoint."""

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from utils.cellvit_adapter_hub import get_adapter_metadata


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("adapter_checkpoint")
    args = parser.parse_args()
    print(
        json.dumps(
            get_adapter_metadata(args.adapter_checkpoint),
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
