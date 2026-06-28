#!/bin/bash

set -euo pipefail

RUN_ROOT="${1:-run}"

echo "Scanning runs under: $RUN_ROOT"

find "$RUN_ROOT" -name config.yaml | while read -r CFG; do
  RUN_DIR=$(dirname "$CFG")

  DATASET_PATH=$(python - <<PY
import yaml
from pathlib import Path

cfg_path = Path("$CFG")
with cfg_path.open("r") as f:
    cfg = yaml.safe_load(f) or {}

print(cfg.get("data", {}).get("dataset_path", ""))
PY
)

  if [ -n "$DATASET_PATH" ] && [ -f "$DATASET_PATH/split_manifest.yaml" ]; then
    cp "$DATASET_PATH/split_manifest.yaml" "$RUN_DIR/dataset_manifest.yaml"
    echo "Copied manifest to $RUN_DIR"
  else
    echo "Missing split_manifest for $RUN_DIR"
  fi
done