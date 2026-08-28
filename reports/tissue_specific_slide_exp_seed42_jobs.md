# True slide-independent tissue-specific Selected PEFT campaign (seed 42)

> 2026-08-25 revision: the primary tissue-coverage campaign is now six tissues
> x Fold A only. Historical rows below are preserved unchanged. Current
> protocol, validation, storage calculation, and job provenance are recorded in
> `reports/tissue_specific_foldA_six_tissue_campaign.md`.

Audit and preparation date: 2026-08-25 (Europe/Paris).

## Scientific protocol and correspondence to COMPAYL

The new experiment is a protocol correction, not a PEFT search. It uses the
canonical CellViT-SAM-H x40 Selected PEFT scope: LoRA on attention Q/V
(rank 8, alpha 8, dropout 0), AdaptFormer (GELU, reduction 16), and only the
final NP/HV/NT heads plus the tissue classifier trainable. Encoder base weights
and all decoder bodies remain frozen. The verified scope is 7,908,779 trainable
parameters out of 707,644,395 (1.11762052%).

The canonical tissue-specific COMPAYL experiments used 40x, the STHELAR
five-class mapping (six outputs including background), batch size 4, AMP,
10 epochs, AdamW at 5e-5 with betas 0.85/0.85, exponential scheduling,
cell-aware sampling with gamma 0.85, and the same augmentation/loss defaults as
the new configs. The old preprocessing split every available slide spatially
along x into 70/15/15 train/valid/test regions with a 128-pixel exclusion
margin. Consequently, both slides of each tissue contributed to all three
splits:

| Tissue | Old seed | Cap | Old train | Old valid | Old test | Old test composition |
|---|---:|---|---:|---:|---:|---|
| Kidney | 42 | none | 8,805 | 1,428 | 939 | kidney_s0: 402; kidney_s1: 537 |
| Liver | 42 | none | 21,301 | 5,653 | 2,639 | liver_s0: 1,647; liver_s1: 992 |
| Tonsil | 43 | 50,000/slide preprocessing cap | 32,932 | 6,900 | 4,318 | tonsil_s0: 2,066; tonsil_s1: 2,252 |

There is no completed canonical five-class Tonsil Selected-PEFT seed-42
within-slide run. Therefore the pre-registered new Tonsil seed-42 folds can be
compared with the published/released within-slide Tonsil seed-43 result only
with an explicit seed caveat. Kidney and Liver have matched seed-42 old runs.
The old seed-42 configs refer to `latest_checkpoint.pth` (the final checkpoint
alias), while the new campaign explicitly evaluates `checkpoint_10.pth` and
retains it as the only persistent `.pth`. No test metric is used for model or
checkpoint selection.

For reference, the old within-slide Selected-PEFT aggregate test metrics were:

| Tissue | Seed | Dice | Jaccard | bPQ | mPQ | F1 detection |
|---|---:|---:|---:|---:|---:|---:|
| Kidney | 42 | 0.726961 | 0.605610 | 0.466722 | 0.230417 | 0.827133 |
| Liver | 42 | 0.818957 | 0.714624 | 0.588138 | 0.392534 | 0.891612 |
| Tonsil | 43 | 0.738147 | 0.613590 | 0.439763 | 0.234562 | 0.822613 |

These are reference values, not evidence about the new held-out slides.

## Materialized metadata-only folds

No image or label payload was regenerated. Each derived directory contains new
split CSV/manifest metadata and links its `images.zip`, `labels.zip`,
`types.csv`, and `dataset_config.yaml` to the corresponding immutable
within-slide packed dataset.

| Tissue | Fold | Train/valid source | Test-only slide | Train | Valid | Test | Slide disjoint | Patch-ID disjoint | Payload links |
|---|---|---|---|---:|---:|---:|---|---|---|
| Kidney | A | kidney_s0 | kidney_s1 | 6,321 | 402 | 4,449 | PASS | PASS | PASS |
| Kidney | B | kidney_s1 | kidney_s0 | 3,912 | 537 | 6,723 | PASS | PASS | PASS |
| Liver | A | liver_s0 | liver_s1 | 18,780 | 1,647 | 9,166 | PASS | PASS | PASS |
| Liver | B | liver_s1 | liver_s0 | 8,174 | 992 | 20,427 | PASS | PASS | PASS |
| Tonsil | A | tonsil_s0 | tonsil_s1 | 21,001 | 2,066 | 21,083 | PASS | PASS | PASS |
| Tonsil | B | tonsil_s1 | tonsil_s0 | 18,831 | 2,252 | 23,067 | PASS | PASS | PASS |

Fold A and Fold B are exact slide reversals for every tissue. Validation is a
spatially separated 15% region of the training slide only; the held-out slide
contributes zero patches to training or validation. All 18 train/valid/test
dataset constructions and first-sample reads passed.

## Configs

- `configs/slide_exp/training/training_sthelar40x_kidney_5class_slideind_foldA_lora_adaptformer_heads_seed42.yaml`
- `configs/slide_exp/training/training_sthelar40x_kidney_5class_slideind_foldB_lora_adaptformer_heads_seed42.yaml`
- `configs/slide_exp/training/training_sthelar40x_liver_5class_slideind_foldA_lora_adaptformer_heads_seed42.yaml`
- `configs/slide_exp/training/training_sthelar40x_liver_5class_slideind_foldB_lora_adaptformer_heads_seed42.yaml`
- `configs/slide_exp/training/training_sthelar40x_tonsil_5class_slideind_foldA_lora_adaptformer_heads_seed42.yaml`
- `configs/slide_exp/training/training_sthelar40x_tonsil_5class_slideind_foldB_lora_adaptformer_heads_seed42.yaml`

The six YAMLs parse and match the corresponding canonical tissue-specific
method/model/training/augmentation settings. The only scientific protocol
changes are the dataset/fold identity and the pre-registered seed 42 for
Tonsil; run provenance, efficiency fields, explicit epoch-10 evaluation, and
safe checkpoint retention are operational additions.

## Validation

- Model construction and the model summary's 256x256 forward pass: PASS.
- Trainable count: 7,908,779 / 707,644,395 (1.11762052%): PASS.
- Encoder LoRA Q/V and AdaptFormer trainable; decoder bodies frozen; final
  NP/HV/NT heads and tissue classifier trainable: PASS.
- YAML/path checks, Python compilation, Bash syntax, atomic efficiency writer,
  tissue-aware efficiency aggregation, and `git diff --check`: PASS.
- Primary inference checkpoint is explicitly `checkpoint_10.pth`: PASS.
- The retention finalizer requires completed training and inference JSONs,
  documents that inference loaded epoch 10, atomically stores best-validation
  metadata and SHA256, and only then removes `model_best.pth`/epochs 1-9: PASS.

## Storage safety ledger

Quota before the cleanup audit was 484,507,200 KiB / 524,288,000 KiB
(approximately 462.06 / 500 GiB; GPFS also reported 655,360 KiB in doubt).
`run/` was approximately 329 GiB while three valid jobs were active.

No files were deleted (0 bytes / 0 GiB reclaimed). The audit found no new high-confidence redundant `.pth`
artifact: every completed slide-independent run already retained one
`checkpoint_10.pth`; the sole directory with two `.pth` files was the active
SAM-H Fold-B FullFT seed-43 run and was therefore protected. Historical
within-slide/release `model_best.pth` files were treated as canonical
provenance and protected. Failed retry directories contained no large
checkpoints, and provenance logs were preserved.

Selected-PEFT checkpoints are approximately 2.696 GiB each. Six final
checkpoints add approximately 16.18 GiB. With the four-GPU QOS, the conservative
worst stage is either three tissue jobs at a two-checkpoint transient peak plus
three completed one-checkpoint runs while the existing FullFT job remains
active (nine new checkpoint equivalents, approximately 24.27 GiB), or four
active plus two completed after that FullFT retention has reclaimed one large
checkpoint (ten equivalents, approximately 26.96 GiB on a lower baseline).
Including the small CellViT-256 PEFT final checkpoints, projected high-water is
approximately 487 GiB, leaving about 13 GiB below the 500-GiB quota. Expected
final usage is approximately 471 GiB after all retention wrappers complete.
The post-submission quota observation was 484,881,920 KiB (462.42 GiB); the
0.36-GiB increase since the pre-audit observation came from concurrently active
scientific jobs, not from cleanup or the pending tissue jobs.

## SLURM provenance

Dependency for every row: `afterok:1478320:1478322` only. No job is pinned to a
node and no historical node blacklist is applied. `1475942` is intentionally
not a dependency.

| Tissue | Fold | Method | Seed | Job ID | State at submission | Dependency | Config |
|---|---|---|---:|---:|---|---|---|
| Kidney | A | Selected PEFT | 42 | 1478434 | PENDING (`Dependency`) | `afterok:1478320:1478322` | `configs/slide_exp/training/training_sthelar40x_kidney_5class_slideind_foldA_lora_adaptformer_heads_seed42.yaml` |
| Kidney | B | Selected PEFT | 42 | 1478433 | PENDING (`Dependency`) | `afterok:1478320:1478322` | `configs/slide_exp/training/training_sthelar40x_kidney_5class_slideind_foldB_lora_adaptformer_heads_seed42.yaml` |
| Liver | A | Selected PEFT | 42 | 1478430 | PENDING (`Dependency`) | `afterok:1478320:1478322` | `configs/slide_exp/training/training_sthelar40x_liver_5class_slideind_foldA_lora_adaptformer_heads_seed42.yaml` |
| Liver | B | Selected PEFT | 42 | 1478432 | PENDING (`Dependency`) | `afterok:1478320:1478322` | `configs/slide_exp/training/training_sthelar40x_liver_5class_slideind_foldB_lora_adaptformer_heads_seed42.yaml` |
| Tonsil | A | Selected PEFT | 42 | 1478431 | PENDING (`Dependency`) | `afterok:1478320:1478322` | `configs/slide_exp/training/training_sthelar40x_tonsil_5class_slideind_foldA_lora_adaptformer_heads_seed42.yaml` |
| Tonsil | B | Selected PEFT | 42 | 1478428 | PENDING (`Dependency`) | `afterok:1478320:1478322` | `configs/slide_exp/training/training_sthelar40x_tonsil_5class_slideind_foldB_lora_adaptformer_heads_seed42.yaml` |

Submission timestamps were 2026-08-25 12:18:19--12:18:22 Europe/Paris.
Each job requests `gpua100`, one GPU, eight CPUs, 64 GiB RAM, and 24 hours;
`ReqNodeList` and `ExcNodeList` are both null.

## Analysis plan and caveats

Each run's `inference_results.json` preserves whole-slide Dice, Jaccard, bPQ,
mPQ, DQ/SQ, detection F1/precision/recall, per-class detection/PQ summaries,
and per-patch paired confusion plus unmatched true/predicted counts. Those
sufficient statistics support held-out-slide F1 type, per-class F1/supports,
and confusion matrices without rerunning inference. Operational metrics are
stored in `efficiency_metrics.json` and
`inference_efficiency_metrics.json`; the aggregation key now includes tissue,
so same-fold/same-seed tissue runs cannot collide.

The final comparison must be reported separately by tissue and fold. Patches
must not be treated as independent biological replicates: the experimental
unit is the held-out slide (two directional folds per tissue). The old test set
was a 15% spatial region from both slides, whereas the new test set is one
complete unseen slide and training uses only the other slide. Thus changes
reflect both removal of slide leakage and slide-to-slide domain shift/data-size
asymmetry. Tonsil additionally has the unavoidable old-seed43/new-seed42
caveat described above.

## Recovery submission, 2026-08-26

Kidney 1480494 completed training but failed on the singleton final inference
batch.  Its existing `checkpoint_10.pth` is recovered by inference-only job
1482332 after the batch-dimension fix; it is not retrained.  The downstream
jobs tied to the permanently failed `afterok:1480494` condition were cancelled
while still pending and replaced without dependencies.

| Tissue | Superseded/cancelled ID | Replacement ID | Type | Dependency |
|---|---:|---:|---|---|
| Kidney A | 1480494 | 1482332 | inference + retention only | none |
| Breast A | 1480496 | 1482333 | training | none |
| Pancreatic A | 1480497 | 1482334 | training | none |
| Ovary A | 1480498 | 1482335 | training | none |
| Colon A | 1480503 | 1482336 | training | none |
| Lung A | 1480504 | 1482337 | training | none |
| Skin A | 1480505 | 1482338 | training | none |

All seven replacement jobs were initially PENDING for normal scheduler
Priority, not dependencies.  Submission timestamp:
2026-08-26T10:53:19+02:00.

## Six-domain Fold-B expansion, 2026-08-26

Six reciprocal Fold-B conditions were submitted without dependencies after an
owner-approved 37.211865-GiB checkpoint cleanup. These are Breast, Pancreatic,
Ovary, Colon, Lung, and Skin; they do not duplicate the historical KLT Fold-B
jobs. Full configs, split counts, validations, storage projection, and job
provenance are recorded in
`reports/tissue_specific_foldB_six_tissue_campaign.md`.

| Tissue | Fold | Job ID | Initial state |
|---|---|---:|---|
| Breast | B | 1492582 | RUNNING |
| Pancreatic | B | 1492583 | RUNNING |
| Ovary | B | 1492584 | RUNNING |
| Colon | B | 1492585 | PENDING (`QOSMaxGRESPerUser`) |
| Lung | B | 1492586 | PENDING (`QOSMaxGRESPerUser`) |
| Skin | B | 1492587 | PENDING (`QOSMaxGRESPerUser`) |

Retry provenance for the three gpu19 infrastructure failures and the subsequent
device-level gpu13 Lung failure is recorded in
`reports/tissue_specific_foldB_six_tissue_campaign.md`. Current replacement
IDs are Ovary `1501898`, Colon `1501899`, and Lung `1501917`; interim Lung job
`1501900` failed before training and does not count as a scientific run.

## Paper-registry status — 2026-08-27

- The required nine-tissue, one-direction Fold-A campaign is COMPLETED VALID
  for Breast, Colon, Kidney, Liver, Lung, Ovary, Pancreatic, Skin, and Tonsil.
- Reciprocal Fold B is supplementary: Breast, Pancreatic, Skin, and Tonsil are
  COMPLETED VALID; Ovary, Colon, and Lung are RUNNING; Kidney and Liver are
  MISSING after FAILED INFRASTRUCTURE attempts.
- Every completed row uses primary `checkpoint_10.pth`. A run folder or YAML
  without canonical inference does not count as an experiment.
- Historical failed, cancelled, dependency-never-satisfied, singleton-inference
  and CUDA attempts remain provenance only. Kidney Fold A is valid because its
  completed epoch-10 checkpoint was recovered by inference job 1482332 without
  retraining.
- MAIN PAPER: nine Fold-A tissue-specialization rows. SUPPORTING ANALYSIS:
  reciprocal Fold-B rows and generalist-versus-specialist comparisons.
