#!/bin/bash
#SBATCH --job-name=sthelar_g3_peft_v100
#SBATCH --time=24:00:00
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --array=0-2%1
#SBATCH --output=logs/%x_%j_%a.out
#SBATCH --error=logs/%x_%j_%a.err

set -euo pipefail

# Concurrency is limited to %1 for the first V100 stability pass.
# If 32GB V100 memory is confirmed safe, change %1 to %2 or %3.
CONFIGS=(
  "configs/train_sthelar40x_colon_5class_spatial_margin128_lora_adaptformer_heads_only_seed42.yaml"
  "configs/train_sthelar40x_lung_5class_spatial_margin128_lora_adaptformer_heads_only_seed42.yaml"
  "configs/train_sthelar40x_skin_5class_spatial_margin128_lora_adaptformer_heads_only_seed42.yaml"
)

REPO="${SLURM_SUBMIT_DIR:-$(pwd)}"
CONFIG="${CONFIGS[$SLURM_ARRAY_TASK_ID]}"
MODULE_NAME="${MODULE_NAME:-miniconda3/25.5.1/none-none}"
CONDA_ENV="${CONDA_ENV:-cellvit39}"

cd "$REPO"
mkdir -p logs
module purge
module load "$MODULE_NAME"
source activate "$CONDA_ENV"

echo "===== JOB INFO ====="
echo "Host:   $(hostname)"
echo "Date:   $(date)"
echo "Repo:   $REPO"
echo "Array:  ${SLURM_ARRAY_TASK_ID}"
echo "Config: $CONFIG"
echo "Python: $(which python)"
python -V
echo "===================="

python cell_segmentation/run_cellvit.py --config "$CONFIG"
