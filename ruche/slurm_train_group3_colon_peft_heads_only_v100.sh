#!/bin/bash
#SBATCH --job-name=sthelar_g3_colon_peft_v100
#SBATCH --time=24:00:00
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err

set -euo pipefail

REPO="${REPO_ROOT:-${SLURM_SUBMIT_DIR:-$PWD}}"
CONFIG="configs/train_sthelar40x_colon_5class_spatial_margin128_lora_adaptformer_heads_only_seed42.yaml"
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
echo "Config: $CONFIG"
echo "Python: $(which python)"
python -V
echo "===================="

python cell_segmentation/run_cellvit.py --config "$CONFIG"
