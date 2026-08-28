# Tissue-specific held-out-slide Fold B — six-domain campaign

Preparation/submission date: 2026-08-26 (Europe/Paris).

## Scope

This campaign is the reciprocal slide direction for the six non-KLT tissue
domains in the expanded tissue-specific study.  It uses the frozen primary
SAM-H Selected PEFT method and seed 42.  Kidney/Liver/Tonsil are not duplicated:
Tonsil Fold B job 1478428 is already complete, whereas the old Kidney Fold B
(1478433) and Liver Fold B (1478432) attempts failed before scientific
training and remain separate missing conditions outside this explicitly
six-job submission.

The model scope is unchanged: LoRA Q/V rank 8 alpha 8, AdaptFormer GELU
reduction 16, final NP/HV/NT heads trainable, base encoder and decoder bodies
frozen.  Completed matched Fold-A runs verify **7,908,779 / 707,644,395**
trainable/total parameters (**1.11762052%**).

## Metadata-only Fold B datasets

No image or label was regenerated or copied.  Each Fold-B directory contains
new split metadata and symlinks `images.zip`, `labels.zip`, `types.csv`, and
`dataset_config.yaml` to the immutable within-slide packed dataset.

| Tissue | Train/validation source | Whole-slide test | Train | Valid | Test | Slide disjoint | Patch-ID disjoint | Payload links/read smoke |
|---|---|---|---:|---:|---:|---|---|---|
| Breast | `breast_s1` | `breast_s0` | 42,794 | 6,775 | 45,773 | PASS | PASS | PASS |
| Pancreatic | `pancreatic_s1` | `pancreatic_s0` | 12,348 | 299 | 25,526 | PASS | PASS | PASS |
| Ovary | `ovary_s1` | `ovary_s0` | 23,387 | 2,111 | 10,039 | PASS | PASS | PASS |
| Colon | `colon_s2` | `colon_s1` | 10,690 | 1,213 | 18,464 | PASS | PASS | PASS |
| Lung | `lung_s3` | `lung_s1` | 19,547 | 1,074 | 10,938 | PASS | PASS | PASS |
| Skin | `skin_s2` | `skin_s1` | 11,308 | 1,482 | 11,209 | PASS | PASS | PASS |

All saved manifests report `all_checks_passed: true`; train/test and
validation/test slide intersections are empty, every packed patch identifier
occurs in exactly one split, and one image/label pair from every split was read
successfully from the linked ZIP payloads.

## Configuration validation

For every tissue, the Fold-B YAML is scientifically identical to its validated
Fold-A YAML except for the reciprocal dataset path and fold/run provenance.
Seed 42, 10 epochs, AdamW 5e-5, exponential scheduler, sampling, augmentation,
loss defaults, batch size 4, AMP, adapter scope, post-processing, and efficiency
instrumentation are unchanged.  Primary inference loads `checkpoint_10.pth`.
After successful inference, the wrapper atomically preserves retention
metadata and keeps only that checkpoint.

Preprocessing configs:

- `configs/slide_exp/preprocessing/preprocessing_sthelar40x_breast_5class_slideind_foldB_margin128_cap50000.yaml`
- `configs/slide_exp/preprocessing/preprocessing_sthelar40x_pancreatic_5class_slideind_foldB_margin128_cap50000.yaml`
- `configs/slide_exp/preprocessing/preprocessing_sthelar40x_ovary_5class_slideind_foldB_margin128.yaml`
- `configs/slide_exp/preprocessing/preprocessing_sthelar40x_colon_5class_slideind_foldB_margin128_cap50000.yaml`
- `configs/slide_exp/preprocessing/preprocessing_sthelar40x_lung_5class_slideind_foldB_margin128_cap50000.yaml`
- `configs/slide_exp/preprocessing/preprocessing_sthelar40x_skin_5class_slideind_foldB_margin128_cap50000.yaml`

Training configs and Slurm provenance:

| Tissue | Job ID | Initial state | Dependency | Config |
|---|---:|---|---|---|
| Breast | 1492582 | RUNNING on `ruche-gpu17` | none | `configs/slide_exp/training/training_sthelar40x_breast_5class_slideind_foldB_lora_adaptformer_heads_seed42.yaml` |
| Pancreatic | 1492583 | RUNNING on `ruche-gpu19` | none | `configs/slide_exp/training/training_sthelar40x_pancreatic_5class_slideind_foldB_lora_adaptformer_heads_seed42.yaml` |
| Ovary | 1492584 | RUNNING on `ruche-gpu19` | none | `configs/slide_exp/training/training_sthelar40x_ovary_5class_slideind_foldB_lora_adaptformer_heads_seed42.yaml` |
| Colon | 1492585 | PENDING (`QOSMaxGRESPerUser`) | none | `configs/slide_exp/training/training_sthelar40x_colon_5class_slideind_foldB_lora_adaptformer_heads_seed42.yaml` |
| Lung | 1492586 | PENDING (`QOSMaxGRESPerUser`) | none | `configs/slide_exp/training/training_sthelar40x_lung_5class_slideind_foldB_lora_adaptformer_heads_seed42.yaml` |
| Skin | 1492587 | PENDING (`QOSMaxGRESPerUser`) | none | `configs/slide_exp/training/training_sthelar40x_skin_5class_slideind_foldB_lora_adaptformer_heads_seed42.yaml` |

All six jobs were submitted at 2026-08-26 22:55:16--17 Europe/Paris to
`gpua100`, requesting one GPU, eight CPUs, 64 GiB and 24 hours.  They have no
Slurm dependency, requested node, or exclusion list.  The three pending jobs
are correctly throttled by the four-GPU per-user QOS and must not be
cancelled/resubmitted merely for that reason.

Logs are written to `logs/<job-name>_<job-id>.out/.err`; run artifacts are
under each config's `logging.log_dir`.  The planned persistent outputs are
`checkpoint_10.pth`, `inference_results.json`, `efficiency_metrics.json`,
`inference_efficiency_metrics.json`, and
`checkpoint_retention_metadata.json`.

## Infrastructure retries, 2026-08-27

The initial Ovary, Colon, and Lung jobs (1492584--1492586) failed before any
training epoch during `model.to(device)` with `CUDA-capable device(s) is/are
busy or unavailable` on `ruche-gpu19`. They produced no checkpoint, inference
result, or efficiency measurement and therefore are not scientific results.

The same unchanged configs were resubmitted with a temporary submission-level
safe-node policy:

| Tissue | Failed ID | Retry ID | State after startup | Assigned node | Excluded nodes |
|---|---:|---:|---|---|---|
| Ovary | 1492584 | 1501898 | RUNNING; training started | `ruche-gpu13` | `gpu11,gpu12,gpu14,gpu16,gpu18,gpu19` |
| Colon | 1492585 | 1501899 | RUNNING; training started | `ruche-gpu13` | `gpu11,gpu12,gpu14,gpu16,gpu18,gpu19` |
| Lung | 1492586 | 1501900 | FAILED during `model.to(device)` | `ruche-gpu13` | `gpu11,gpu12,gpu14,gpu16,gpu18,gpu19` |
| Lung | 1501900 | 1501917 | RUNNING; training started | `ruche-gpu15` | `gpu11,gpu12,gpu13,gpu14,gpu16,gpu18,gpu19` |

The Lung failure on one allocated gpu13 device, while Ovary and Colon began
training on other devices of the same node, is additional evidence of a
device-level infrastructure fault rather than a dataset or method failure.
No scientific configuration, checkpoint policy, dataset, seed, or training
setting changed in any retry. There are no dependencies and no job is pinned
to a specific node.

## Read-only status reconciliation — 2026-08-27

| Tissue | Canonical/retry job | Current classification | Canonical artifacts |
|---|---:|---|---|
| Breast | 1492582 | COMPLETED VALID | checkpoint 10, inference, both efficiency JSONs, retention metadata |
| Pancreatic | 1492583 | COMPLETED VALID | same |
| Skin | 1492587 | COMPLETED VALID | same |
| Ovary | 1501898 | RUNNING | incomplete efficiency record only; no final checkpoint/inference yet |
| Colon | 1501899 | RUNNING | incomplete efficiency record only; no final checkpoint/inference yet |
| Lung | 1501917 | RUNNING | incomplete efficiency record only; no final checkpoint/inference yet |

The original Ovary/Colon/Lung jobs 1492584--1492586 and Lung retry 1501900 are
FAILED INFRASTRUCTURE and SUPERSEDED. They produced no canonical inference and
must not enter scientific tables. Running rows must remain classified RUNNING
until their wrapper has produced all five required final artifacts; validation
or partial efficiency output is not a test result.

Outside this six-domain campaign, Tonsil Fold B job 1478428 is COMPLETED
VALID. Kidney Fold B job 1478433 and Liver Fold B job 1478432 are FAILED
INFRASTRUCTURE with no valid replacement and therefore remain MISSING
scientific reciprocal conditions. These missing Fold-B rows do not invalidate
the preregistered nine-tissue one-direction Fold-A study.

## Final read-only reconciliation — 2026-08-28

Slurm accounting reports jobs 1501898 (Ovary), 1501899 (Colon), and 1501917
(Lung) as `COMPLETED` with exit code `0:0`. Each canonical timestamped run was
also checked locally and contains `checkpoint_10.pth`, final
`inference_results.json`, completed training and inference efficiency JSONs,
and checkpoint-retention metadata. They are therefore COMPLETED VALID rather
than RUNNING. Together with Breast, Pancreatic, Skin, and Tonsil, seven
reciprocal Fold-B tissues are complete. Kidney and Liver remain the only
missing reciprocal tissue conditions; no replacement job was launched during
this reconciliation.
