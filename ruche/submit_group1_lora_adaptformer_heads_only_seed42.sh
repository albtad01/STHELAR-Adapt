#!/usr/bin/env bash
set -euo pipefail

execute=false
if [[ "${1:-}" == "--execute" ]]; then
  execute=true
elif [[ $# -gt 0 ]]; then
  echo "Usage: $0 [--execute]" >&2
  exit 2
fi

configs=(
  "configs/examples/training_sthelar40x_liver_5class_spatial_margin128_lora_adaptformer_r8_a8_red16_decoder_heads_only_lr5e-5_e10_seed42_CLEAN.yaml"
  "configs/examples/training_sthelar40x_kidney_5class_spatial_margin128_lora_adaptformer_r8_a8_red16_decoder_heads_only_lr5e-5_e10_seed42_CLEAN.yaml"
  "configs/examples/training_sthelar40x_ovary_5class_spatial_margin128_lora_adaptformer_r8_a8_red16_decoder_heads_only_lr5e-5_e10_seed42_CLEAN.yaml"
)

for config in "${configs[@]}"; do
  cmd=(sbatch --exclude=ruche-gpu16 ruche/slurm_train.sh "$config")
  if [[ "$execute" == true ]]; then
    "${cmd[@]}"
  else
    printf '[dry-run] '
    printf '%q ' "${cmd[@]}"
    printf '\n'
  fi
done
