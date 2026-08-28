# Tissue-specific Fold-A result snapshot — reconciled 2026-08-27

Values below are final `checkpoint_10.pth` whole-held-out-slide test results.
Validation metrics are not used as test results. The earlier 2026-08-26
eight-tissue snapshot has now been reconciled with the completed Breast run.

## Campaign status

| Tissue | Job | State | Final artifacts |
|---|---:|---|---|
| Breast | 1482333 | COMPLETED | complete; one checkpoint retained |
| Kidney | 1482332 | COMPLETED recovery inference | complete; one checkpoint retained |
| Liver | 1480495 | COMPLETED | complete; one checkpoint retained |
| Ovary | 1482335 | COMPLETED | complete; one checkpoint retained |
| Pancreatic | 1482334 | COMPLETED | complete; one checkpoint retained |
| Tonsil | 1478431 | COMPLETED | complete; one checkpoint retained |
| Colon | 1482336 | COMPLETED | complete; one checkpoint retained |
| Lung | 1482337 | COMPLETED | complete; one checkpoint retained |
| Skin | 1482338 | COMPLETED | complete; one checkpoint retained |

The Kidney singleton-batch recovery succeeded without retraining. All nine
Fold-A directions now have a readable `checkpoint_10.pth`, final
`inference_results.json`, both efficiency JSONs, and atomic retention metadata.

## Held-out-slide results and within-slide deltas

`F1type` is the paper-compatible unweighted macro-F1 over foreground classes
with nonzero matched true support, computed on matched nuclei only.

| Tissue | bPQ | mPQ | F1 detection | F1type | ΔbPQ | ΔmPQ | ΔF1det | ΔF1type |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Breast | 0.4113 | 0.2386 | 0.8129 | 0.5000 | -0.1154 | -0.0962 | -0.0295 | -0.0873 |
| Kidney | 0.5495 | 0.0904 | 0.8478 | 0.1492 | +0.0828 | -0.1401 | +0.0207 | -0.3260 |
| Liver | 0.4801 | 0.1715 | 0.8432 | 0.2649 | -0.1081 | -0.2210 | -0.0485 | -0.3024 |
| Ovary | 0.4488 | 0.2226 | 0.8344 | 0.5054 | -0.0774 | -0.0772 | -0.0071 | -0.1128 |
| Pancreatic | 0.3774 | 0.1239 | 0.7338 | 0.2788 | -0.0078 | -0.1150 | -0.0045 | -0.3894 |
| Tonsil | 0.4522 | 0.2158 | 0.8383 | 0.4879 | +0.0125 | -0.0187 | +0.0156 | -0.1076 |
| Colon | 0.2847 | 0.1495 | 0.7510 | 0.4265 | -0.0415 | -0.0523 | -0.0396 | -0.1782 |
| Lung | 0.5083 | 0.1934 | 0.8430 | 0.4042 | -0.0355 | -0.1081 | -0.0207 | -0.2229 |
| Skin | 0.4401 | 0.1342 | 0.7432 | 0.2425 | +0.1113 | -0.0898 | +0.0181 | -0.4353 |

Across all nine tissues, the tissue-level mean deltas are bPQ -0.0199, mPQ
-0.1020, F1 detection -0.0106, and F1type -0.2402. Every tissue loses mPQ and
F1type, whereas bPQ and F1 detection improve in three. Restricting to the seven
seed-42 comparisons without the Pancreatic/Tonsil checkpoint/seed mismatch gives
mean ΔbPQ -0.0262, ΔmPQ -0.1121, ΔF1 detection -0.0152, and ΔF1type -0.2379.

This supports a specific interpretation: class-agnostic detection and instance
segmentation are comparatively robust to slide shift, while phenotype typing
is not. Skin and Kidney are the clearest examples: bPQ improves despite large
F1type losses. Breast has the largest bPQ decrease, while Liver has the largest
mPQ decrease.
Pancreatic and Tonsil retain an old-seed43/new-seed42 and checkpoint-policy
caveat and should not be treated as perfectly matched comparisons.

The existing machine-readable matched table is
`reports/tissue_specific_foldA_within_vs_slideind.csv`; it remains the original
2026-08-26 eight-result snapshot and its Breast row is stale. This read-only
scientific audit did not rewrite CSV outputs; use the final Breast values above
until the table is regenerated from the existing inference JSON.

## Historical recommendations (superseded as planning registry)

The bullets below preserve the state of the 2026-08-26 snapshot. The sole
current forward-looking registry is `reports/paper_future_experiments.md`; use
that file rather than this historical section for planning.

No new run should be submitted until Breast completes and storage is reclaimed.

1. Highest-value seed-44 replication: SAM-H Selected PEFT Fold A and Fold B.
   These two jobs would give the primary selected method three seeds on both
   slide-independent folds.  Frozen has no meaningful training seed.
2. For a fully balanced core comparison, subsequently add LP Fold A/B seed44
   and FullFT Fold A/B seed44.  FullFT should be staged one fold at a time due
   to its approximately 7.82-GiB persistent checkpoint and approximately
   15.64-GiB two-checkpoint transient footprint per active run.
3. Do not prioritize NT-header1 seed44: it is a secondary typing ablation, not
   the selected method.  Run it only if the revised paper makes a formal claim
   comparing NT-header1 with Selected PEFT.
4. Do not prioritize tissue-specific seed44.  A new optimization seed does not
   add an independent slide.  Reciprocal Fold-B tissue experiments, especially
   Kidney, Liver, Skin, and Pancreatic, have greater biological value because
   they measure direction-dependent slide shift.

Approximate final checkpoint additions for a complete SAM-H seed44 core are:
Selected PEFT A/B 5.39 GiB, LP A/B about 5.2 GiB, and FullFT A/B about
15.64 GiB, for roughly 26 GiB persistent storage before transient copies.

## Read-only checkpoint candidates

Nothing in this section was deleted.  Current slide-independent
`checkpoint_10.pth` files, release artifacts, pretrained weights, and published
adapter sources remain protected.

### Automatic current-run cleanup

Do not manually touch the active Breast files.  Its wrapper should supersede
`checkpoint_9.pth` with `checkpoint_10.pth` and remove `model_best.pth` only
after successful final inference, reclaiming about 2.696 GiB net.

### Highest-confidence owner-approval candidate

| Size GiB | Checkpoint | Evidence |
|---:|---|---|
| 2.696 | `run/sthelar40x_breast_5class_spatial_margin128_cap50000_lora_adaptformer_r8_a8_red16_decoder_heads_only_lr5e-5_e10_seed43_CLEAN/log/2026-06-30T234009_sthelar40x_breast_5class_spatial_margin128_cap50000_lora_adaptformer_r8_a8_red16_decoder_heads_only_lr5e-5_e10_seed43_CLEAN/checkpoints/model_best.pth` | interrupted at epoch 7; no final inference; not referenced by either release manifest; preserve config/log provenance |

### Exploratory candidates requiring explicit approval

The prior storage inventory marks these runs `KEEP_USEFUL`, not
`CANDIDATE_DELETE`.  They are not referenced by paper tables, adapter manifests,
published adapters, or qualitative/QC code according to that inventory, and
their inference results/configs/logs survive.  Before deletion, preserve a
SHA256 and verify the result JSON one final time.

| Size GiB | Experiment/checkpoint |
|---:|---|
| 7.821 | `run/sthelar40x_liver_5class_spatial_fullft_lr1e-5_e10_seed42_KEEPALL_FISHER/log/2026-06-24T161831_sthelar40x_liver_5class_spatial_fullft_lr1e-5_e10_seed42_KEEPALL_FISHER/checkpoints/model_best.pth` |
| 7.821 | `run/sthelar40x_bps_9class_slide_fullft_lr1e-5_e10_seed42_CLEAN_TRUE/log/2026-06-21T160034_sthelar40x_bps_9class_slide_fullft_lr1e-5_e10_seed42_CLEAN_TRUE/checkpoints/model_best.pth` |
| 2.702 | `run/sthelar40x_tonsil_9class_slide_lora_adaptformer_r8_a8_red16_decoder_conv_adapters_lr5e-5_e10_seed42_CLEAN/log/2026-06-20T184700_sthelar40x_tonsil_9class_slide_lora_adaptformer_r8_a8_red16_decoder_conv_adapters_lr5e-5_e10_seed42_CLEAN/checkpoints/model_best.pth` |
| 2.698 | `run/sthelar40x_tonsil_9class_slide_lora_adaptformer_r8_a8_red16_decoder_last_stage_lr5e-5_e10_seed42_CLEAN/log/2026-06-20T201957_sthelar40x_tonsil_9class_slide_lora_adaptformer_r8_a8_red16_decoder_last_stage_lr5e-5_e10_seed42_CLEAN/checkpoints/model_best.pth` |
| 2.688 | `run/sthelar40x_tonsil_9class_slide_vera_adaptformer_r8_a8_red16_decoder_conv_adapters_lr5e-5_e10_seed42_CLEAN/log/2026-06-21T041310_sthelar40x_tonsil_9class_slide_vera_adaptformer_r8_a8_red16_decoder_conv_adapters_lr5e-5_e10_seed42_CLEAN/checkpoints/model_best.pth` |
| 2.702 | `run/sthelar40x_liver_5class_spatial_lora_adaptformer_r8_a8_red16_decoder_conv_adapters_lr5e-5_e10_seed42_CLEAN/log/2026-06-19T153620_sthelar40x_liver_5class_spatial_lora_adaptformer_r8_a8_red16_decoder_conv_adapters_lr5e-5_e10_seed42_CLEAN/checkpoints/model_best.pth` |
| 2.698 | `run/sthelar40x_liver_5class_spatial_lora_adaptformer_r8_a8_red16_decoder_last_stage_lr5e-5_e10_seed42_CLEAN/log/2026-06-20T113354_sthelar40x_liver_5class_spatial_lora_adaptformer_r8_a8_red16_decoder_last_stage_lr5e-5_e10_seed42_CLEAN/checkpoints/model_best.pth` |
| 2.696 | `run/sthelar40x_liver_5class_spatial_lora_adaptformer_r8_a8_red16_decoder_heads_only_lr5e-5_e10_seed42_CLEAN/log/2026-06-20T113354_sthelar40x_liver_5class_spatial_lora_adaptformer_r8_a8_red16_decoder_heads_only_lr5e-5_e10_seed42_CLEAN/checkpoints/model_best.pth` |
| 2.688 | `run/sthelar40x_liver_5class_spatial_vera_adaptformer_r8_a8_red16_decoder_conv_adapters_lr5e-5_e10_seed42_CLEAN/log/2026-06-20T202033_sthelar40x_liver_5class_spatial_vera_adaptformer_r8_a8_red16_decoder_conv_adapters_lr5e-5_e10_seed42_CLEAN/checkpoints/model_best.pth` |

These exploratory checkpoints total approximately 34.52 GiB.  Including the
incomplete Breast seed43 checkpoint gives approximately 37.21 GiB of potential
owner-approved reclaim.  The two exploratory FullFT files alone provide about
15.64 GiB.  Deletion should be recorded path-by-path in the storage ledger; it
must not be based on test performance.

> **2026-08-26 update:** the owner approved all ten candidates above. They
> were deleted after path/evidence verification and SHA256 capture, reclaiming
> exactly 39,955,935,504 bytes (37.211865 GiB). See
> `reports/storage_cleanup_owner_approved_20260826.md`. No result JSON, config,
> log, canonical checkpoint, dataset, adapter, or release artifact was removed.
