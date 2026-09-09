#!/bin/bash
set -euo pipefail

usage() {
  cat <<'USAGE'
Usage:
  bash ruche/submit_tissue_fullft_after_preprocessing.sh [--execute] [--group 1|2|3]

Default mode prints the dependency sbatch commands and exits without submitting.
Use --execute to submit after an interactive confirmation prompt.
USAGE
}

EXECUTE=0
GROUP="all"

while [[ $# -gt 0 ]]; do
  case "$1" in
    --execute)
      EXECUTE=1
      shift
      ;;
    --group)
      GROUP="${2:?missing group after --group}"
      shift 2
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      echo "ERROR: unknown argument: $1" >&2
      usage >&2
      exit 2
      ;;
  esac
done

if [[ "$GROUP" != "all" && "$GROUP" != "1" && "$GROUP" != "2" && "$GROUP" != "3" ]]; then
  echo "ERROR: --group must be one of: 1, 2, 3" >&2
  exit 2
fi

: "${DATA_ROOT:?Set DATA_ROOT to the prepared CellViT dataset root}"

declare -a COMMANDS=()
declare -a CONFIGS=()
declare -a DATASETS=()

add_job() {
  local dependency="$1"
  local config="$2"
  local dataset="$3"
  COMMANDS+=("sbatch --dependency=afterok:${dependency} --exclude=ruche-gpu16 ruche/slurm_train.sh ${config}")
  CONFIGS+=("$config")
  DATASETS+=("$dataset")
}

if [[ "$GROUP" == "all" || "$GROUP" == "1" ]]; then
  add_job 1170941 configs/examples/training_sthelar40x_liver_5class_spatial_margin128_fullft_lr1e-5_e10_seed42_CLEAN.yaml "$DATA_ROOT/sthelar40x_liver_5class_spatial_margin128"
  add_job 1170941 configs/examples/training_sthelar40x_kidney_5class_spatial_margin128_fullft_lr1e-5_e10_seed42_CLEAN.yaml "$DATA_ROOT/sthelar40x_kidney_5class_spatial_margin128"
  add_job 1170941 configs/examples/training_sthelar40x_tonsil_5class_spatial_margin128_fullft_lr1e-5_e10_seed42_CLEAN.yaml "$DATA_ROOT/sthelar40x_tonsil_5class_spatial_margin128"
fi

if [[ "$GROUP" == "all" || "$GROUP" == "2" ]]; then
  add_job 1170942 configs/examples/training_sthelar40x_breast_5class_spatial_margin128_fullft_lr1e-5_e10_seed42_CLEAN.yaml "$DATA_ROOT/sthelar40x_breast_5class_spatial_margin128"
  add_job 1170942 configs/examples/training_sthelar40x_colon_5class_spatial_margin128_fullft_lr1e-5_e10_seed42_CLEAN.yaml "$DATA_ROOT/sthelar40x_colon_5class_spatial_margin128"
  add_job 1170942 configs/examples/training_sthelar40x_lung_5class_spatial_margin128_fullft_lr1e-5_e10_seed42_CLEAN.yaml "$DATA_ROOT/sthelar40x_lung_5class_spatial_margin128"
fi

if [[ "$GROUP" == "all" || "$GROUP" == "3" ]]; then
  add_job 1170943 configs/examples/training_sthelar40x_ovary_5class_spatial_margin128_fullft_lr1e-5_e10_seed42_CLEAN.yaml "$DATA_ROOT/sthelar40x_ovary_5class_spatial_margin128"
  add_job 1170943 configs/examples/training_sthelar40x_pancreatic_5class_spatial_margin128_fullft_lr1e-5_e10_seed42_CLEAN.yaml "$DATA_ROOT/sthelar40x_pancreatic_5class_spatial_margin128"
  add_job 1170943 configs/examples/training_sthelar40x_skin_5class_spatial_margin128_fullft_lr1e-5_e10_seed42_CLEAN.yaml "$DATA_ROOT/sthelar40x_skin_5class_spatial_margin128"
fi

echo "Checking configs and dataset paths..."
for i in "${!CONFIGS[@]}"; do
  config="${CONFIGS[$i]}"
  dataset="${DATASETS[$i]}"
  if [[ ! -f "$config" ]]; then
    echo "ERROR: missing config: $config" >&2
    exit 1
  fi
  if [[ ! -d "$dataset" ]]; then
    echo "WARNING: dataset path does not exist yet; expected after preprocessing dependency: $dataset" >&2
  fi
done

echo
echo "Commands:"
printf '%s\n' "${COMMANDS[@]}"

if [[ "$EXECUTE" -ne 1 ]]; then
  echo
  echo "Dry run only. Re-run with --execute to submit."
  exit 0
fi

echo
read -r -p "Submit ${#COMMANDS[@]} training jobs now? Type yes to continue: " answer
if [[ "$answer" != "yes" ]]; then
  echo "Aborted."
  exit 0
fi

for command in "${COMMANDS[@]}"; do
  echo "+ $command"
  eval "$command"
done
