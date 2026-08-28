# Slide-independent KLT seed-42 jobs — 2026-08-19

## Resumed-session scheduler reconciliation (2026-08-19 11:11 CEST)

The original submissions below were recovered from this report, `sacct`, and
the SLURM logs after the interrupted session. They were **not duplicated**.
Five original jobs failed before useful work with Ruche's
`CUDA-capable device(s) is/are busy or unavailable` error on `ruche-gpu17`.
Only those five conditions were resubmitted, with `ruche-gpu16` and
`ruche-gpu17` excluded. The materialized folds had already validated, so the
replacement jobs correctly have no new dependency.

| Fold | Condition | Type | Active job ID | Config | Dependency | GPU / partition | Submitted (Europe/Paris) | State at 11:05 CEST | Recovered/retry note |
|---|---|---|---:|---|---|---|---|---|---|
| A | Frozen | inference | 1465774 | `configs/slide_exp/training/evaluation_sthelar40x_klt_5class_slideind_foldA_frozen_seed42.yaml` | submitted as `afterok:1465772`; now satisfied | 1 GPU, `gpua100`; `ruche-gpu16` excluded | 2026-08-19 10:53:59 | RUNNING on `ruche-gpu17` | Recovered original job. |
| A | LP | training | 1465814 | `configs/slide_exp/training/training_sthelar40x_klt_5class_slideind_foldA_lp_final_heads_seed42.yaml` | none; fold A validation already completed | 1 GPU, `gpua100`; `ruche-gpu16,ruche-gpu17` excluded | 2026-08-19 11:04:30 | PENDING (`QOSMaxGRESPerUser`) | Replaces failed 1465775; no duplicate successful job. |
| A | Selected PEFT | training | 1465776 | `configs/slide_exp/training/training_sthelar40x_klt_5class_slideind_foldA_lora_adaptformer_heads_seed42.yaml` | submitted as `afterok:1465772`; now satisfied | 1 GPU, `gpua100`; `ruche-gpu16` excluded | 2026-08-19 10:54:06 | RUNNING on `ruche-gpu17` | Recovered original job. |
| A | FullFT | training | 1465777 | `configs/slide_exp/training/training_sthelar40x_klt_5class_slideind_foldA_fullft_seed42.yaml` | submitted as `afterok:1465772`; now satisfied | 1 GPU, `gpua100`; `ruche-gpu16` excluded | 2026-08-19 10:54:09 | RUNNING on `ruche-gpu11` | Recovered original job. |
| B | Frozen | inference | 1465815 | `configs/slide_exp/training/evaluation_sthelar40x_klt_5class_slideind_foldB_frozen_seed42.yaml` | none; fold B validation already completed | 1 GPU, `gpua100`; `ruche-gpu16,ruche-gpu17` excluded | 2026-08-19 11:04:40 | PENDING (`QOSMaxGRESPerUser`) | Replaces failed 1465778. |
| B | LP | training | 1465816 | `configs/slide_exp/training/training_sthelar40x_klt_5class_slideind_foldB_lp_final_heads_seed42.yaml` | none; fold B validation already completed | 1 GPU, `gpua100`; `ruche-gpu16,ruche-gpu17` excluded | 2026-08-19 11:04:48 | PENDING (`QOSMaxGRESPerUser`) | Replaces failed 1465779. |
| B | Selected PEFT | training | 1465821 | `configs/slide_exp/training/training_sthelar40x_klt_5class_slideind_foldB_lora_adaptformer_heads_seed42.yaml` | none; fold B validation already completed | 1 GPU, `gpua100`; `ruche-gpu16,ruche-gpu17` excluded | 2026-08-19 11:10:19 | PENDING (`Priority`) | Replaces failed 1465780. |
| B | FullFT | training | 1465822 | `configs/slide_exp/training/training_sthelar40x_klt_5class_slideind_foldB_fullft_seed42.yaml` | none; fold B validation already completed | 1 GPU, `gpua100`; `ruche-gpu16,ruche-gpu17` excluded | 2026-08-19 11:10:27 | PENDING (`Priority`) | Replaces failed 1465781. |

Original failed jobs retained for provenance: 1465775 (Fold A LP), 1465778
(Fold B Frozen), 1465779 (Fold B LP), 1465780 (Fold B Selected PEFT), and
1465781 (Fold B FullFT). Their partial run/log artifacts were not deleted or
overwritten.

All downstream jobs were submitted with `afterok` dependencies on their fold's
metadata-only validation job. Both preprocessing jobs subsequently completed with
exit code `0:0`; SLURM therefore released the eight dependent A100 jobs, which were
pending for priority at the time of this report.

| Fold | Condition | Preprocess job ID | Dependent job ID | Job type | Config | Submitted (Europe/Paris) | Submitted dependency |
|---|---|---:|---:|---|---|---|---|
| A | Preprocess/validate | 1465772 | — | metadata-only fold validation | `configs/slide_exp/preprocessing/preprocessing_sthelar40x_klt_5class_slideind_foldA_margin128.yaml` | 2026-08-19 10:53:41 | — |
| A | Frozen | 1465772 | 1465774 | class-agnostic inference | `configs/slide_exp/training/evaluation_sthelar40x_klt_5class_slideind_foldA_frozen_seed42.yaml` | 2026-08-19 10:53:59 | `afterok:1465772` |
| A | LP | 1465772 | 1465775 | training + inference | `configs/slide_exp/training/training_sthelar40x_klt_5class_slideind_foldA_lp_final_heads_seed42.yaml` | 2026-08-19 10:54:03 | `afterok:1465772` |
| A | Selected PEFT | 1465772 | 1465776 | training + inference | `configs/slide_exp/training/training_sthelar40x_klt_5class_slideind_foldA_lora_adaptformer_heads_seed42.yaml` | 2026-08-19 10:54:06 | `afterok:1465772` |
| A | FullFT | 1465772 | 1465777 | training + inference | `configs/slide_exp/training/training_sthelar40x_klt_5class_slideind_foldA_fullft_seed42.yaml` | 2026-08-19 10:54:09 | `afterok:1465772` |
| B | Preprocess/validate | 1465773 | — | metadata-only fold validation | `configs/slide_exp/preprocessing/preprocessing_sthelar40x_klt_5class_slideind_foldB_margin128.yaml` | 2026-08-19 10:53:43 | — |
| B | Frozen | 1465773 | 1465778 | class-agnostic inference | `configs/slide_exp/training/evaluation_sthelar40x_klt_5class_slideind_foldB_frozen_seed42.yaml` | 2026-08-19 10:54:12 | `afterok:1465773` |
| B | LP | 1465773 | 1465779 | training + inference | `configs/slide_exp/training/training_sthelar40x_klt_5class_slideind_foldB_lp_final_heads_seed42.yaml` | 2026-08-19 10:54:15 | `afterok:1465773` |
| B | Selected PEFT | 1465773 | 1465780 | training + inference | `configs/slide_exp/training/training_sthelar40x_klt_5class_slideind_foldB_lora_adaptformer_heads_seed42.yaml` | 2026-08-19 10:54:18 | `afterok:1465773` |
| B | FullFT | 1465773 | 1465781 | training + inference | `configs/slide_exp/training/training_sthelar40x_klt_5class_slideind_foldB_fullft_seed42.yaml` | 2026-08-19 10:54:21 | `afterok:1465773` |

## Dependency graph

```text
1465772  Fold A metadata validation (COMPLETED 0:0)
├── 1465774  Frozen Fold A evaluation
├── 1465775  LP Fold A training + inference
├── 1465776  Selected PEFT Fold A training + inference
└── 1465777  FullFT Fold A training + inference

1465773  Fold B metadata validation (COMPLETED 0:0)
├── 1465778  Frozen Fold B evaluation
├── 1465779  LP Fold B training + inference
├── 1465780  Selected PEFT Fold B training + inference
└── 1465781  FullFT Fold B training + inference
```

All eight dependent jobs request `gpua100`, one GPU, eight CPUs, and 64 GiB RAM;
`ruche-gpu16` was excluded consistently. Training jobs use the shared 24-hour policy;
Frozen inference uses an 8-hour limit because it performs no optimization.

## Validated fold contents

| Fold | Train slides | Validation sources | Test-only slides | Train patches | Validation patches | Test patches |
|---|---|---|---|---:|---:|---:|
| A | kidney_s0; liver_s0; tonsil_s0 | kidney_s0; liver_s0; tonsil_s0 | kidney_s1; liver_s1; tonsil_s1 | 24,283 | 2,052 | 23,494 |
| B | kidney_s1; liver_s1; tonsil_s1 | kidney_s1; liver_s1; tonsil_s1 | kidney_s0; liver_s0; tonsil_s0 | 20,893 | 2,601 | 26,335 |

For both folds: train/test disjointness passed; validation/test disjointness passed;
patch identifiers are unique across splits; configured and observed slide sets match;
and Fold B is the exact train/test reversal of Fold A. The preprocessing jobs reused
the immutable packed `images.zip` and `labels.zip` via links and generated only fold
metadata.

## Pre-submission validation

- all 10 new YAML files parsed;
- every referenced input path existed and no Frozen output directory pre-existed;
- canonical scientific settings matched for all six training configs;
- tiny dataloader smoke passed for train/validation/test in both folds;
- model construction and a tiny forward passed for LP, Selected PEFT, and FullFT;
- trainable counts were LP 650 / 699,736,523; Selected PEFT 7,908,779 /
  707,644,395 (1.11762052%); FullFT 699,736,523 / 699,736,523;
- Frozen checkpoint SHA256 was
  `b324c10fddb0f80f5ab03a0459453a4c4848866934daf63435b46749a6b278cf`,
  strict loading matched all keys, trainable parameters were zero, and a CPU forward
  passed without head replacement;
- atomic efficiency JSON writer smoke passed;
- Python compilation, SLURM `bash -n`, and `git diff --check` passed.

No seed 43/44 or tissue-specific slide-independent jobs were submitted.

## NT-header1 PEFT extension (2026-08-19 11:50 CEST)

These two jobs reuse the already materialized and validated Fold A/Fold B
datasets. They have no preprocessing dependency. Both use the same A100
training wrapper and resource policy as the current campaign (one `gpua100`
GPU, eight CPUs, 64 GiB RAM, 24 hours), with `ruche-gpu16` and
`ruche-gpu17` excluded. The scheduler accepted both jobs and left them queued
normally under `QOSMaxGRESPerUser`.

| Fold | Method | Type | Job ID | State at submission | Trainable parameters | Config | Efficiency metrics path |
|---|---|---|---:|---|---:|---|---|
| A | NT-header1 PEFT | training + inference | 1465912 | PENDING (`QOSMaxGRESPerUser`) | 7,945,835 / 707,644,395 (1.122857%) | `configs/slide_exp/training/training_sthelar40x_klt_5class_slideind_foldA_lora_adaptformer_nt_header1_np_hv_heads_seed42.yaml` | `run/sthelar40x_klt_5class_slideind_foldA_lora_adaptformer_r8_a8_red16_nt_header1_np_hv_heads_e10_seed42/log/<timestamped-run>/efficiency_metrics.json` |
| B | NT-header1 PEFT | training + inference | 1465913 | PENDING (`QOSMaxGRESPerUser`) | 7,945,835 / 707,644,395 (1.122857%) | `configs/slide_exp/training/training_sthelar40x_klt_5class_slideind_foldB_lora_adaptformer_nt_header1_np_hv_heads_seed42.yaml` | `run/sthelar40x_klt_5class_slideind_foldB_lora_adaptformer_r8_a8_red16_nt_header1_np_hv_heads_e10_seed42/log/<timestamped-run>/efficiency_metrics.json` |

Checkpoint retention was verified before submission: `save_every: 1` writes an
epoch checkpoint, while `keep_last_n: 1` and
`delete_intermediate_checkpoints: true` prune the previous epoch checkpoint.
A normal completed run therefore retains at most `model_best.pth` and the
latest epoch checkpoint (normally `checkpoint_10.pth`), never all ten epoch
checkpoints. If best and latest coincide, the final deduplication uses a hard
link so both names share one payload. A Selected-PEFT-like checkpoint is about
2.895 GB (2.70 GiB), making the expected completed run footprint about 2.7 GiB
when deduplicated and at most about 5.4 GiB when best and latest differ.

## Fold B infrastructure retries with one-checkpoint policy (2026-08-20)

Accounting confirmed that Fold B Selected PEFT job 1465821 and Fold B
NT-header1 PEFT job 1465913 failed before the first training epoch during
`model.to(device)` on `ruche-gpu14`. Neither produced a checkpoint, completed
efficiency record, or scientific result, and no replacement existed before the
following submissions. These are scientific no-change infrastructure retries.

| Fold | Method | Seed | New job ID | Replaces | State at 11:27 CEST | Config | GPU / partition | Excluded nodes | Submitted (Europe/Paris) | Checkpoint retention |
|---|---|---:|---:|---:|---|---|---|---|---|---|
| B | Selected PEFT | 42 | 1468272 | 1465821 | RUNNING on `ruche-gpu19` | `configs/slide_exp/training/training_sthelar40x_klt_5class_slideind_foldB_lora_adaptformer_heads_seed42.yaml` | 1 A100, `gpua100`; 8 CPUs; 64 GiB; 24 h | `ruche-gpu14,ruche-gpu16,ruche-gpu17` | 2026-08-20 11:23:33 | After successful checkpoint-10 inference, retain only `checkpoint_10.pth`; metadata in `<run>/checkpoint_retention_metadata.json`. |
| B | NT-header1 PEFT | 42 | 1468273 | 1465913 | PENDING (`Resources`) | `configs/slide_exp/training/training_sthelar40x_klt_5class_slideind_foldB_lora_adaptformer_nt_header1_np_hv_heads_seed42.yaml` | 1 A100, `gpua100`; 8 CPUs; 64 GiB; 24 h | `ruche-gpu14,ruche-gpu16,ruche-gpu17` | 2026-08-20 11:23:35 | After successful checkpoint-10 inference, retain only `checkpoint_10.pth`; metadata in `<run>/checkpoint_retention_metadata.json`. |

Training efficiency metrics remain at each timestamped run's
`efficiency_metrics.json`. The corresponding expected roots are:

- Selected PEFT: `run/sthelar40x_klt_5class_slideind_foldB_lora_adaptformer_r8_a8_red16_heads_e10_seed42/log/<timestamped-run>/efficiency_metrics.json`;
- NT-header1 PEFT: `run/sthelar40x_klt_5class_slideind_foldB_lora_adaptformer_r8_a8_red16_nt_header1_np_hv_heads_e10_seed42/log/<timestamped-run>/efficiency_metrics.json`.

Both retries have no preprocessing dependency because the Fold B metadata and
linked immutable payload were revalidated before submission.

## Final scientific reconciliation — 2026-08-27

All ten seed-42 SAM-H conditions are now **COMPLETED VALID**. Failed earlier
attempts remain infrastructure provenance and are SUPERSEDED; they are not
additional experimental replicates. Canonical trainable inference used
`checkpoint_10.pth`; Frozen used the untouched official pretrained model and
reports class-agnostic metrics only.

| Fold | Condition | Canonical job | Final status | Canonical artifacts |
|---|---|---:|---|---|
| A | Frozen | 1465774 | COMPLETED VALID | `run/sthelar40x_klt_5class_slideind_foldA_frozen_seed42/eval/` |
| B | Frozen | 1465815 | COMPLETED VALID | `run/sthelar40x_klt_5class_slideind_foldB_frozen_seed42/eval/` |
| A | LP | 1465814 | COMPLETED VALID | `run/sthelar40x_klt_5class_slideind_foldA_lp_final_heads_e10_seed42/log/2026-08-19T114405_sthelar40x_klt_5class_slideind_foldA_lp_final_heads_e10_seed42/` |
| B | LP | 1465816 | COMPLETED VALID | `run/sthelar40x_klt_5class_slideind_foldB_lp_final_heads_e10_seed42/log/2026-08-19T123449_sthelar40x_klt_5class_slideind_foldB_lp_final_heads_e10_seed42/` |
| A | Selected PEFT | 1465776 | COMPLETED VALID | `run/sthelar40x_klt_5class_slideind_foldA_lora_adaptformer_r8_a8_red16_heads_e10_seed42/log/2026-08-19T105648_sthelar40x_klt_5class_slideind_foldA_lora_adaptformer_r8_a8_red16_heads_e10_seed42/` |
| B | Selected PEFT | 1468272 | COMPLETED VALID | `run/sthelar40x_klt_5class_slideind_foldB_lora_adaptformer_r8_a8_red16_heads_e10_seed42/log/2026-08-20T112527_sthelar40x_klt_5class_slideind_foldB_lora_adaptformer_r8_a8_red16_heads_e10_seed42/` |
| A | NT-header1 | 1465912 | COMPLETED VALID | `run/sthelar40x_klt_5class_slideind_foldA_lora_adaptformer_r8_a8_red16_nt_header1_np_hv_heads_e10_seed42/log/2026-08-19T183125_sthelar40x_klt_5class_slideind_foldA_lora_adaptformer_r8_a8_red16_nt_header1_np_hv_heads_e10_seed42/` |
| B | NT-header1 | 1468273 | COMPLETED VALID | `run/sthelar40x_klt_5class_slideind_foldB_lora_adaptformer_r8_a8_red16_nt_header1_np_hv_heads_e10_seed42/log/2026-08-20T114259_sthelar40x_klt_5class_slideind_foldB_lora_adaptformer_r8_a8_red16_nt_header1_np_hv_heads_e10_seed42/` |
| A | FullFT | 1465777 | COMPLETED VALID | `run/sthelar40x_klt_5class_slideind_foldA_fullft_lr1e-5_e10_seed42/log/2026-08-19T110413_sthelar40x_klt_5class_slideind_foldA_fullft_lr1e-5_e10_seed42/` |
| B | FullFT | 1465822 | COMPLETED VALID | `run/sthelar40x_klt_5class_slideind_foldB_fullft_lr1e-5_e10_seed42/log/2026-08-19T181004_sthelar40x_klt_5class_slideind_foldB_fullft_lr1e-5_e10_seed42/` |

Each trainable directory above contains the retained epoch-10 checkpoint,
canonical inference JSON, completed training/inference efficiency JSONs, and
retention metadata. Full-precision metrics and per-class/per-slide provenance
are consolidated in `reports/neurips2026_paper_results/`. Scientific role:
Frozen/LP/Selected PEFT/FullFT belong in the MAIN PAPER; NT-header1 is a
SUPPORTING ANALYSIS typing ablation.
