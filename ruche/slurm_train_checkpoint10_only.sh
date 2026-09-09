#!/bin/bash
#SBATCH --job-name=train_sthelar_ckpt10
#SBATCH --time=24:00:00
#SBATCH --partition=gpua100
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err

# Campaign-specific wrapper: train/infer normally, then safely retain only
# checkpoint_10.pth.  Shared trainer/checkpoint behaviour is not modified.
set -euo pipefail

REPO="${SLURM_SUBMIT_DIR:-$(pwd)}"
CONFIG="${1:?Usage: sbatch ruche/slurm_train_checkpoint10_only.sh CONFIG}"
if [[ "$CONFIG" != /* ]]; then
  CONFIG="$REPO/$CONFIG"
fi
if [[ ! -f "$CONFIG" ]]; then
  echo "ERROR: config file not found: $CONFIG"
  exit 1
fi

MODULE_NAME="${MODULE_NAME:-miniconda3/25.5.1/none-none}"
CONDA_ENV="${CONDA_ENV:-cellvit39}"
STARTED_AT_EPOCH="$(date +%s)"

cd "$REPO"
mkdir -p logs
module purge
module load "$MODULE_NAME"
source activate "$CONDA_ENV"

echo "===== JOB INFO ====="
echo "Host:        $(hostname)"
echo "Date:        $(date)"
echo "Repo:        $REPO"
echo "Config:      $CONFIG"
echo "Python:      $(which python)"
echo "Retention:   checkpoint_10.pth only, after successful final inference"
python -V
echo "===================="

echo "===== START TRAINING ====="
python cell_segmentation/run_cellvit.py --config "$CONFIG"
echo "===== TRAINING AND FINAL INFERENCE DONE ====="

echo "===== START CHECKPOINT RETENTION ====="
python utils/finalize_checkpoint10_retention.py \
  --config "$CONFIG" \
  --repo "$REPO" \
  --not-before-epoch "$STARTED_AT_EPOCH"
echo "===== CHECKPOINT RETENTION DONE ====="
