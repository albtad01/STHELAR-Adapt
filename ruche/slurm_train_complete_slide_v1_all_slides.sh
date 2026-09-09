#!/bin/bash
#SBATCH --job-name=cs_v1_allslides
#SBATCH --time=24:00:00
#SBATCH --partition=gpua100
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err

# Complete-slide-v1 deployment training. The canonical epoch-10 checkpoint is
# retained; only intermediate checkpoints and the redundant temporary .pth
# adapter are removed, and only after exact safetensors verification succeeds.
set -euo pipefail

REPO="${SLURM_SUBMIT_DIR:-$(pwd)}"
TRAIN_CONFIG="${1:?Usage: sbatch $0 TRAIN_CONFIG PREPROCESSING_CONFIG}"
PREP_CONFIG="${2:?Usage: sbatch $0 TRAIN_CONFIG PREPROCESSING_CONFIG}"
if [[ "$TRAIN_CONFIG" != /* ]]; then TRAIN_CONFIG="$REPO/$TRAIN_CONFIG"; fi
if [[ "$PREP_CONFIG" != /* ]]; then PREP_CONFIG="$REPO/$PREP_CONFIG"; fi
for path in "$TRAIN_CONFIG" "$PREP_CONFIG"; do
  if [[ ! -f "$path" ]]; then echo "ERROR: missing config: $path"; exit 1; fi
done

: "${DATA_ROOT:?Set DATA_ROOT to the prepared CellViT dataset root before submission}"
export DATA_ROOT
cd "$REPO"
mkdir -p logs adapters/complete_slide_v1_staging adapters/slide_exp_safetensors

module purge
module load "${MODULE_NAME:-miniconda3/25.5.1/none-none}"
source activate "${CONDA_ENV:-cellvit39}"

python utils/generate_slide_independent_fold.py --config "$PREP_CONFIG"

# This validated wrapper trains/inferes with checkpoint_10 and removes only
# model_best/checkpoints 1-9 after successful final validation inference.
source "$REPO/ruche/slurm_train_checkpoint10_only.sh" "$TRAIN_CONFIG"

RUN_NAME="$(python -c 'import sys,yaml; print(yaml.safe_load(open(sys.argv[1]))["logging"]["log_comment"])' "$TRAIN_CONFIG")"
STAGING_DIR="$(python -c 'import sys,yaml; print(yaml.safe_load(open(sys.argv[1]))["adapter_export"]["output_dir"])' "$TRAIN_CONFIG")"
if [[ "$STAGING_DIR" != /* ]]; then STAGING_DIR="$REPO/$STAGING_DIR"; fi
SOURCE_ADAPTER="$STAGING_DIR/${RUN_NAME}_adapter.pth"
FINAL_DIR="$REPO/adapters/slide_exp_safetensors/$RUN_NAME"

RUN_DIR="$(python - "$TRAIN_CONFIG" "$REPO" "$STARTED_AT_EPOCH" <<'PY'
import sys
from pathlib import Path
import yaml
config_path, repo, not_before = Path(sys.argv[1]), Path(sys.argv[2]), float(sys.argv[3])
source = yaml.safe_load(config_path.read_text())
root = Path(source["logging"]["log_dir"])
if not root.is_absolute(): root = repo / root
candidates = []
for candidate in root.iterdir():
    generated = candidate / "config.yaml"
    retained = candidate / "checkpoint_retention_metadata.json"
    if not generated.is_file() or generated.stat().st_mtime < not_before or not retained.is_file(): continue
    current = yaml.safe_load(generated.read_text())
    if current.get("random_seed") == source.get("random_seed") and current.get("logging", {}).get("log_comment") == source.get("logging", {}).get("log_comment"):
        candidates.append(candidate.resolve())
if len(candidates) != 1: raise RuntimeError(f"Expected one new run, found {candidates}")
print(candidates[0])
PY
)"

if [[ ! -f "$SOURCE_ADAPTER" ]]; then echo "ERROR: missing exported adapter: $SOURCE_ADAPTER"; exit 1; fi
if [[ -e "$FINAL_DIR" ]]; then echo "ERROR: refusing to overwrite: $FINAL_DIR"; exit 1; fi

python tools/export_adapter_safetensors.py \
  "$SOURCE_ADAPTER" "$FINAL_DIR" \
  --adapter-id "complete_slide_v1/${RUN_NAME,,}" \
  --source-config "$TRAIN_CONFIG" \
  --base-checkpoint-sha256 "b324c10fddb0f80f5ab03a0459453a4c4848866934daf63435b46749a6b278cf" \
  --expected-components lora adaptformer np_head hv_head nt_head

python tools/verify_released_adapter.py \
  "$FINAL_DIR" \
  --base-checkpoint "$REPO/models/pretrained/CellViT-SAM-H-x40.pth" \
  --training-config "$TRAIN_CONFIG" \
  --canonical-checkpoint "$RUN_DIR/checkpoints/checkpoint_10.pth" \
  --output-json "$FINAL_DIR/verification.json"

rm -f -- "$SOURCE_ADAPTER"
python tools/finalize_complete_slide_adapter_release.py

echo "VERIFIED adapter: $FINAL_DIR/adapter_model.safetensors"
echo "RETAINED canonical source: $RUN_DIR/checkpoints/checkpoint_10.pth"
