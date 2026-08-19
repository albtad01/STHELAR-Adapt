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
