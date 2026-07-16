#!/bin/bash
set -euo pipefail

sbatch ruche/slurm_train.sh "configs/examples/training_sthelar40x_tonsil_9class_slide_freeze_e5.yaml"
sbatch ruche/slurm_train.sh "configs/examples/training_sthelar40x_tonsil_9class_slide_lora_ntonly_r4_a4_lr5e-5_e5.yaml"
sbatch ruche/slurm_train.sh "configs/examples/training_sthelar40x_tonsil_9class_slide_lora_r8_a8_lr5e-5_e5.yaml"
sbatch ruche/slurm_train.sh "configs/examples/training_sthelar40x_tonsil_9class_slide_adaptformer_gelu_red16_lr5e-5_e5.yaml"
