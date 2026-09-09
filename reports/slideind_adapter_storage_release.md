# Slide-independent adapter storage release

Date: 2026-08-28

## Outcome

The canonical completed Selected-PEFT slide-independent runs were converted to
the repository's existing safetensors adapter format. Twenty CellViT-SAM-H-x40
packages and two CellViT-256-x40 packages were exported and independently
verified before their canonical `checkpoint_10.pth` files were deleted.

The authoritative per-adapter deletion and provenance table is
[`release/huggingface/slideind_adapter_manifest_verified.csv`](../release/huggingface/slideind_adapter_manifest_verified.csv).
It records the backbone and base hash, source run/checkpoint/config, fold, seed,
tissue/domain, trainable parameter count, every exported parameter and mutable
buffer name, serialized bytes, adapter and source SHA256 values, timestamps,
verification state, deletion safety decision, and final deletion status.

No training or inference jobs were launched for this storage pass. All
verification forwards ran locally and deterministically on CPU.

## Canonical inventory

| Backbone | Experiment family | Fold/seed coverage | Packages | Adapter bytes |
|---|---|---:|---:|---:|
| CellViT-SAM-H-x40 | KLT slide-independent | A/B, seeds 42 and 43 | 4 | 127,241,064 |
| CellViT-SAM-H-x40 | Tissue-specific slide-independent | Fold A seed42: all nine tissues; Fold B seed42: Breast, Colon, Lung, Ovary, Pancreatic, Skin, Tonsil | 16 | 508,964,552 |
| CellViT-256-x40 | KLT slide-independent | A/B, seed42 | 2 | 3,223,344 |
| **Total** |  |  | **22** | **639,428,960 (609.807 MiB)** |

Kidney and Liver Fold B seed42 were not completed at export time and no
adapter was fabricated. Their later pending/retry jobs are outside this
inventory. Likewise, externally submitted KLT seed44 and CellViT-256 seed43
jobs use separate run and release paths and were not touched.

Canonical adapter IDs follow the existing archive convention:

- `cellvit-sam-h-x40/klt/slideind/fold{a,b}/seed{42,43}`
- `cellvit-sam-h-x40/tissue_specific/<tissue>/slideind/fold{a,b}/seed42`
- `cellvit-256-x40/klt/slideind/fold{a,b}/seed42`

Each final directory contains `adapter_model.safetensors`,
`adapter_config.json`, `checksums.json`, and `verification.json`. No duplicate
canonical ID was created.

## Contents and reconstruction contract

The packages reuse `utils/adapter_checkpoint.py` and the established release
exporter/loader. They contain only:

- LoRA Q/V parameters;
- AdaptFormer parameters;
- final NP, HV, and NT head parameters;
- the trainable tissue classifier state required by the configuration; and
- the 105 mutable normalization buffers required for exact reconstruction.

SAM-H packages contain 296 trainable parameter tensors plus 105 mutable
buffers (7,908,779 trainable parameters). CellViT-256 packages contain 114
trainable parameter tensors plus 105 mutable buffers (374,198 trainable
parameters). Optimizer, scheduler, GradScaler, epoch state, and duplicated
frozen backbone tensors are absent.

Required retained bases:

- `CellViT-SAM-H-x40.pth`, SHA256
  `b324c10fddb0f80f5ab03a0459453a4c4848866934daf63435b46749a6b278cf`
- `CellViT-256-x40.pth`, SHA256
  `ee3986922fc500353db3d7692c566e19e1c694c8f10e47cf49dfd90160fc3b2b`

## Verification gates

Before deletion, every package passed all of the following:

1. construct a fresh architecture-matched model from the shared pretrained base;
2. load only the final archived adapter package;
3. compare all reconstructed model-state tensors with the canonical checkpoint;
4. run a deterministic `1 x 3 x 256 x 256` linspace input;
5. require bitwise equality for `nuclei_binary_map`, `hv_map`,
   `nuclei_type_map`, and `tissue_types`;
6. verify package/config/base/source hashes and metadata; and
7. independently reload the package from its final archive location.

Every record reports `state_reconstruction=exact_all_tensors` and
`forward_verification=exact_all_output_tensors`, with maximum absolute output
difference 0.0. After each small deletion batch, the final archive was loaded
again from the retained base and the run's configuration/results/efficiency/
retention metadata and symlink integrity were checked.

## Storage accounting

| Point | Ruche quota | File count | Notes |
|---|---:|---:|---|
| Before export | 463 GiB / 500 GiB (92%) | 182,060 | Baseline immediately before this pass |
| After export | 463 GiB / 500 GiB (92%) | 182,488 | Quota is displayed in rounded GiB |
| After deletion | 420 GiB / 500 GiB (83%) | 182,480 | 80 GiB reported remaining |

The 22 deleted checkpoints occupied exactly 58,275,734,854 bytes
(54.273507 GiB). The retained adapters occupy 639,428,960 bytes
(609.806976 MiB), a 91.14-to-1 reduction relative to those source files.
The observed rounded quota reduction is 43 GiB rather than the exact deleted
size because independent active jobs were writing new run artifacts during
this pass. The exact reclaim figure above is derived from the individually
recorded deleted file sizes; quota values are contemporaneous observations.

## Deliberately retained

- all run directories and their `config.yaml`, `inference_results.json`,
  `efficiency_metrics.json`, `inference_efficiency_metrics.json`,
  `checkpoint_retention_metadata.json`, logs, and scientific outputs;
- split manifests, validation metadata, reports, and result CSVs;
- public pretrained CellViT backbone checkpoints;
- all LP, FullFT, and NT-header checkpoints;
- datasets, predictions, and active/pending-run files; and
- all historical within-slide training checkpoints and compact source adapter
  files.

The 12 historical COMPAYL within-slide release packages and
`verification_summary.json` were audited. Their published safetensors files
have round-trip verification, but the release manifest points to compact
source adapter `.pth` files rather than uniquely freezing the relationship to
every local multi-GiB training `model_best.pth` and all downstream consumers.
Those training checkpoints were therefore conservatively retained; none was
deleted in this pass.

## Exact deleted paths

Only the following canonical Selected-PEFT `checkpoint_10.pth` files were
deleted. Their containing directories remain intact.

1. `run/sthelar40x_klt_5class_slideind_foldA_cellvit256_lora_adaptformer_r8_a8_red16_heads_e10_seed42/log/2026-08-25T114238_sthelar40x_klt_5class_slideind_foldA_cellvit256_lora_adaptformer_r8_a8_red16_heads_e10_seed42/checkpoints/checkpoint_10.pth`
2. `run/sthelar40x_klt_5class_slideind_foldB_cellvit256_lora_adaptformer_r8_a8_red16_heads_e10_seed42/log/2026-08-25T114238_sthelar40x_klt_5class_slideind_foldB_cellvit256_lora_adaptformer_r8_a8_red16_heads_e10_seed42/checkpoints/checkpoint_10.pth`
3. `run/sthelar40x_klt_5class_slideind_foldA_lora_adaptformer_r8_a8_red16_heads_e10_seed42/log/2026-08-19T105648_sthelar40x_klt_5class_slideind_foldA_lora_adaptformer_r8_a8_red16_heads_e10_seed42/checkpoints/checkpoint_10.pth`
4. `run/sthelar40x_klt_5class_slideind_foldB_lora_adaptformer_r8_a8_red16_heads_e10_seed42/log/2026-08-20T112527_sthelar40x_klt_5class_slideind_foldB_lora_adaptformer_r8_a8_red16_heads_e10_seed42/checkpoints/checkpoint_10.pth`
5. `run/sthelar40x_klt_5class_slideind_foldA_lora_adaptformer_r8_a8_red16_heads_e10_seed43/log/2026-08-24T122637_sthelar40x_klt_5class_slideind_foldA_lora_adaptformer_r8_a8_red16_heads_e10_seed43/checkpoints/checkpoint_10.pth`
6. `run/sthelar40x_klt_5class_slideind_foldB_lora_adaptformer_r8_a8_red16_heads_e10_seed43/log/2026-08-24T122637_sthelar40x_klt_5class_slideind_foldB_lora_adaptformer_r8_a8_red16_heads_e10_seed43/checkpoints/checkpoint_10.pth`
7. `run/sthelar40x_breast_5class_slideind_foldA_lora_adaptformer_r8_a8_red16_heads_e10_seed42/log/2026-08-26T105512_sthelar40x_breast_5class_slideind_foldA_lora_adaptformer_r8_a8_red16_heads_e10_seed42/checkpoints/checkpoint_10.pth`
8. `run/sthelar40x_colon_5class_slideind_foldA_lora_adaptformer_r8_a8_red16_heads_e10_seed42/log/2026-08-26T110313_sthelar40x_colon_5class_slideind_foldA_lora_adaptformer_r8_a8_red16_heads_e10_seed42/checkpoints/checkpoint_10.pth`
9. `run/sthelar40x_kidney_5class_slideind_foldA_lora_adaptformer_r8_a8_red16_heads_e10_seed42/log/2026-08-25T220941_sthelar40x_kidney_5class_slideind_foldA_lora_adaptformer_r8_a8_red16_heads_e10_seed42/checkpoints/checkpoint_10.pth`
10. `run/sthelar40x_liver_5class_slideind_foldA_lora_adaptformer_r8_a8_red16_heads_e10_seed42/log/2026-08-25T220941_sthelar40x_liver_5class_slideind_foldA_lora_adaptformer_r8_a8_red16_heads_e10_seed42/checkpoints/checkpoint_10.pth`
11. `run/sthelar40x_lung_5class_slideind_foldA_lora_adaptformer_r8_a8_red16_heads_e10_seed42/log/2026-08-26T141800_sthelar40x_lung_5class_slideind_foldA_lora_adaptformer_r8_a8_red16_heads_e10_seed42/checkpoints/checkpoint_10.pth`
12. `run/sthelar40x_ovary_5class_slideind_foldA_lora_adaptformer_r8_a8_red16_heads_e10_seed42/log/2026-08-26T105512_sthelar40x_ovary_5class_slideind_foldA_lora_adaptformer_r8_a8_red16_heads_e10_seed42/checkpoints/checkpoint_10.pth`
13. `run/sthelar40x_pancreatic_5class_slideind_foldA_lora_adaptformer_r8_a8_red16_heads_e10_seed42/log/2026-08-26T105512_sthelar40x_pancreatic_5class_slideind_foldA_lora_adaptformer_r8_a8_red16_heads_e10_seed42/checkpoints/checkpoint_10.pth`
14. `run/sthelar40x_skin_5class_slideind_foldA_lora_adaptformer_r8_a8_red16_heads_e10_seed42/log/2026-08-26T164216_sthelar40x_skin_5class_slideind_foldA_lora_adaptformer_r8_a8_red16_heads_e10_seed42/checkpoints/checkpoint_10.pth`
15. `run/sthelar40x_tonsil_5class_slideind_foldA_lora_adaptformer_r8_a8_red16_heads_e10_seed42/log/2026-08-25T192051_sthelar40x_tonsil_5class_slideind_foldA_lora_adaptformer_r8_a8_red16_heads_e10_seed42/checkpoints/checkpoint_10.pth`
16. `run/sthelar40x_breast_5class_slideind_foldB_lora_adaptformer_r8_a8_red16_heads_e10_seed42/log/2026-08-26T225723_sthelar40x_breast_5class_slideind_foldB_lora_adaptformer_r8_a8_red16_heads_e10_seed42/checkpoints/checkpoint_10.pth`
17. `run/sthelar40x_colon_5class_slideind_foldB_lora_adaptformer_r8_a8_red16_heads_e10_seed42/log/2026-08-27T154219_sthelar40x_colon_5class_slideind_foldB_lora_adaptformer_r8_a8_red16_heads_e10_seed42/checkpoints/checkpoint_10.pth`
18. `run/sthelar40x_lung_5class_slideind_foldB_lora_adaptformer_r8_a8_red16_heads_e10_seed42/log/2026-08-27T154613_sthelar40x_lung_5class_slideind_foldB_lora_adaptformer_r8_a8_red16_heads_e10_seed42/checkpoints/checkpoint_10.pth`
19. `run/sthelar40x_ovary_5class_slideind_foldB_lora_adaptformer_r8_a8_red16_heads_e10_seed42/log/2026-08-27T154219_sthelar40x_ovary_5class_slideind_foldB_lora_adaptformer_r8_a8_red16_heads_e10_seed42/checkpoints/checkpoint_10.pth`
20. `run/sthelar40x_pancreatic_5class_slideind_foldB_lora_adaptformer_r8_a8_red16_heads_e10_seed42/log/2026-08-26T225723_sthelar40x_pancreatic_5class_slideind_foldB_lora_adaptformer_r8_a8_red16_heads_e10_seed42/checkpoints/checkpoint_10.pth`
21. `run/sthelar40x_skin_5class_slideind_foldB_lora_adaptformer_r8_a8_red16_heads_e10_seed42/log/2026-08-26T230626_sthelar40x_skin_5class_slideind_foldB_lora_adaptformer_r8_a8_red16_heads_e10_seed42/checkpoints/checkpoint_10.pth`
22. `run/sthelar40x_tonsil_5class_slideind_foldB_lora_adaptformer_r8_a8_red16_heads_e10_seed42/log/2026-08-25T192051_sthelar40x_tonsil_5class_slideind_foldB_lora_adaptformer_r8_a8_red16_heads_e10_seed42/checkpoints/checkpoint_10.pth`
