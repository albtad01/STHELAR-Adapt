# Fold-A slide-independent Selected-PEFT tissue campaign

Prepared and submitted on 2026-08-25 (Europe/Paris). This revision replaces
the planned reciprocal three-tissue study as the primary tissue-coverage
experiment. It does not cancel or alter the already running Tonsil Fold-B job.
The initial literal `s0 -> s1` subset contained six tissues; after confirming
that the remaining domains merely use different slide numbering, the study was
expanded to all nine canonical tissue domains using one ordered direction.

## Exact tissue set

The nine canonical tissue-specific packed datasets were audited from their
actual `patch_info_with_split.csv` files. Exactly six have the literal slide
pair required by the preregistered `tissue_s0 -> tissue_s1` direction:

| Tissue | Train/validation source | Whole untouched test slide |
|---|---|---|
| Breast | `breast_s0` | `breast_s1` |
| Kidney | `kidney_s0` | `kidney_s1` |
| Liver | `liver_s0` | `liver_s1` |
| Ovary | `ovary_s0` | `ovary_s1` |
| Pancreatic | `pancreatic_s0` | `pancreatic_s1` |
| Tonsil | `tonsil_s0` | `tonsil_s1` |

Colon (`s1/s2`), Lung (`s1/s3`), and Skin (`s1/s2`) each have two slides but
do not satisfy the literal `s0 -> s1` spelling. After explicit approval, they
were added without relabelling as Colon `s1 -> s2`, Lung `s1 -> s3`, and Skin
`s1 -> s2`; the scientific rule remains one whole slide for train/validation
and a distinct whole slide for test.

## Expansion to all nine tissue domains

| Tissue | Train/validation source | Whole untouched test | Train | Valid | Test | Job ID | Dependency |
|---|---|---|---:|---:|---:|---:|---|
| Colon | `colon_s1` | `colon_s2` | 16,435 | 2,029 | 11,903 | 1480503 | `afterok:1480496:1480497` |
| Lung | `lung_s1` | `lung_s3` | 9,544 | 1,394 | 20,621 | 1480504 | `afterok:1480496:1480497:1480498` |
| Skin | `skin_s1` | `skin_s2` | 10,561 | 648 | 12,790 | 1480505 | `afterok:1480496:1480497:1480503:1480504` |

All three metadata datasets are metadata-only and passed exact slide identity,
train/test and validation/test disjointness, patch-ID uniqueness, payload-link,
YAML/path, scientific-setting identity, and train/validation/test first-sample
checks. Their model/training configuration is identical to the already
validated Selected-PEFT scope.

The dependencies all include Breast/Pancreatic directly and add storage-safe
throttling around the already queued Ovary job. Immediately launching Ovary,
Colon, Lung, and Skin together after Breast/Pancreatic would project roughly
492.6 GiB at the simultaneous two-checkpoint peak. The staged graph instead
keeps at most two of these runs active and projects a conservative maximum of
about 484.5 GiB before small JSON outputs, with final usage about 481.8 GiB.
No existing job or dependency was modified.

The immediate post-expansion quota observation was 494,116,368 KiB = 471.226
GiB. The increase relative to the pre-submission snapshot coincides with the
active Kidney/Liver jobs beginning to write temporary checkpoints and is
already included in the staged high-water reasoning; it is not duplicated
payload data.

## Scientific correspondence and caveats

All new conditions use CellViT-SAM-H x40 Selected PEFT: LoRA Q/V rank 8,
alpha 8 and dropout 0; AdaptFormer GELU reduction 16; final NP/HV/NT heads and
the existing tissue classifier trainable; encoder base and decoder bodies
frozen. The verified scope is 7,908,779 trainable parameters out of
707,644,395 (1.11762052%). Training is seed 42, batch size 4, AMP, 10 epochs,
AdamW at 5e-5 with betas 0.85/0.85, exponential scheduling, cell-aware
sampling gamma 0.85, and the canonical augmentation/loss/post-processing
policy. Primary test inference is fixed to `checkpoint_10.pth` and uses no
test-based selection.

The old within-slide datasets used the same 40x five-class payloads and a
128-pixel spatial margin, but train/valid/test regions from both slides. The
new validation region is carved spatially only from `s0`, and the complete
`s1` payload becomes test. Therefore deltas measure removal of slide leakage
plus slide-to-slide domain shift and training/test-size asymmetry; patches are
not biological replicates.

| Tissue | Old Selected seed | Old train | Old valid | Old test | Old primary checkpoint | New seed |
|---|---:|---:|---:|---:|---|---:|
| Breast | 42 | 71,875 | 14,024 | 9,443 | final epoch alias | 42 |
| Kidney | 42 | 8,805 | 1,428 | 939 | final epoch alias | 42 |
| Liver | 42 | 21,301 | 5,653 | 2,639 | final epoch alias | 42 |
| Ovary | 42 | 28,131 | 5,026 | 2,380 | final epoch alias | 42 |
| Pancreatic | 43 | 30,787 | 4,780 | 2,606 | validation-selected `model_best` | 42 |
| Tonsil | 43 | 32,932 | 6,900 | 4,318 | validation-selected `model_best` | 42 |

There is no canonical completed five-class Selected-PEFT within-slide seed-42
run for Pancreatic or Tonsil. Their direct comparison must therefore retain
both an old-seed43/new-seed42 caveat and the old-model-best/new-epoch10 caveat;
these two rows must not be presented as perfectly seed/checkpoint matched.

## Metadata-only Fold A datasets and validation

No images or labels were regenerated. `images.zip`, `labels.zip`, `types.csv`,
and `dataset_config.yaml` are symlinks to each immutable within-slide packed
dataset. The existing repository slide-split helper applies the spatial-margin
principle within `s0`; because it partitions by spatial coordinate rather than
patch count, the realized validation percentage varies with tissue geometry.

| Tissue | Train | Valid | Test | Train/test slides | Valid/test slides | Patch IDs | Payload links |
|---|---:|---:|---:|---|---|---|---|
| Breast | 43,105 | 2,668 | 49,569 | PASS | PASS | PASS | PASS |
| Kidney | 6,321 | 402 | 4,449 | PASS | PASS | PASS | PASS |
| Liver | 18,780 | 1,647 | 9,166 | PASS | PASS | PASS | PASS |
| Ovary | 9,770 | 269 | 25,498 | PASS | PASS | PASS | PASS |
| Pancreatic | 23,219 | 2,307 | 12,647 | PASS | PASS | PASS | PASS |
| Tonsil | 21,001 | 2,066 | 21,083 | PASS | PASS | PASS | PASS |

All 18 train/validation/test dataset constructions and first-sample reads
passed. Every manifest reports the exact slide identities above; patch IDs are
globally unique across the three splits.

## Reuse, supersession, and SLURM provenance

Tonsil Fold A job 1478431 was already running the exact condition and is
reused. Failed Kidney/Liver Fold-A jobs 1478434 and 1478430 failed during CUDA
initialization before a completed epoch/checkpoint/inference result and are
superseded by 1480494 and 1480495. No duplicate of a valid condition was
submitted.

Temporary submission-level exclusion:
`ruche-gpu11,ruche-gpu12,ruche-gpu14,ruche-gpu16,ruche-gpu17,ruche-gpu18,ruche-gpu19`.
Only gpu13/gpu15 were eligible at submission. gpu13/gpu15 had multiple recent
long successful KLT jobs; gpu11/gpu19 had newer unresolved CUDA-device
failures; gpu12 had a recent NODE_FAIL and no completed post-recovery campaign
job; gpu14 lacked sufficiently recent clean campaign evidence; gpu16-18 were
DOWN/POWERED_DOWN with ResumeTimeout. This is not a permanent repository
blacklist and no job is pinned to a node.

| Tissue | Condition status | Job ID | Old failed ID | Dependency | State just after submission |
|---|---|---:|---:|---|---|
| Tonsil | reused exact run | 1478431 | - | none | RUNNING on gpu19 |
| Kidney | retry | 1480494 | 1478434 | none | RUNNING on gpu13 |
| Liver | retry | 1480495 | 1478430 | none | RUNNING on gpu13 |
| Breast | new | 1480496 | - | `afterok:1480494:1480495` | PENDING (Dependency) |
| Pancreatic | new | 1480497 | - | `afterok:1480494:1480495` | PENDING (Dependency) |
| Ovary | new | 1480498 | - | `afterok:1480496:1480497` | PENDING (Dependency) |

The throttling graph is:

```text
Kidney 1480494 --+
                 +-- afterok --> Breast 1480496 ----+
Liver  1480495 --+                                  +-- afterok --> Ovary 1480498
                 +-- afterok --> Pancreatic 1480497-+
```

Thus at most two newly submitted SAM-H tissue runs can be at the transient
two-checkpoint peak. The unrelated already-running Tonsil Fold B is neither a
dependency nor a member of this six-condition analysis.

## Storage audit

Quota at the cleanup audit was 488,232,736 KiB / 524,288,000 KiB =
465.615 / 500 GiB. No automatic deletion was performed: all completed
slide-independent runs already retained exactly one canonical
`checkpoint_10.pth`; the only two-checkpoint directories belonged to active
Tonsil jobs; failed Kidney/Liver directories contained only small provenance
logs/configs and no checkpoint. Reclaimed space is therefore exactly 0 bytes.
The post-metadata/post-submission observation was 488,275,424 KiB = 465.656
GiB; the small increase is active logs/metadata, not cleanup.

A Selected-PEFT checkpoint is about 2.696 GiB. From the audit snapshot, the
five missing final checkpoints plus retention of the two active Tonsil runs
project approximately 473.70 GiB final usage (less than 474.3 GiB allowing for
JSON outputs). The two-job wave policy projects approximately 476.40 GiB at
the worst checkpoint high-water (less than about 477 GiB with outputs),
leaving over 23 GiB below the 500-GiB quota.

## Output and comparison plan

Each successful wrapper run must leave one `checkpoint_10.pth` plus
`inference_results.json`, `efficiency_metrics.json`,
`inference_efficiency_metrics.json`, and
`checkpoint_retention_metadata.json`. The inference JSON stores sufficient
per-patch paired confusion and unmatched counts to derive held-out-slide Dice,
bPQ, mPQ, detection F1, present-class F1 type, per-class F1/support, and the
aggregate confusion matrix without treating patches as replicates.

The matched result scaffold is
`reports/tissue_specific_foldA_within_vs_slideind.csv`. After all six complete,
fill the new columns directly from epoch-10 inference and compute each delta as
`held_out_s1 - within_slide`. Interpret detection/segmentation, typing, and
tissue-dependent slide shift separately; report one held-out slide per tissue,
not patch-level confidence intervals.

Logs are in `logs/<job-name>_<job-id>.out/.err`; run-local scientific outputs
are under each config's `logging.log_dir` followed by its timestamped run
directory.

## Result status update (2026-08-26)

Eight of nine Fold-A tissue directions now have complete epoch-10 held-out
slide inference and one retained checkpoint.  Breast replacement job 1482333
is still running at epoch 10/10.  The machine-readable within-slide comparison
has been updated at `reports/tissue_specific_foldA_within_vs_slideind.csv`; the
metric interpretation, seed-44 priorities, and read-only storage candidate
audit are in `reports/tissue_specific_foldA_results_20260826.md`.

## 2026-08-26 singleton-inference recovery and dependency-free retries

Kidney retry 1480494 completed all ten training epochs and safely wrote
`checkpoint_10.pth`, but failed on the final one-patch inference batch.  The
inference code used an unrestricted `squeeze()` on the nuclei-type target;
with test count 4,449 and inference batch size 16, the last batch had size one
and lost its batch dimension before a four-dimensional `permute`.  No Kidney
training was repeated.  The fix removes only the singleton channel dimension
and passed synthetic `unpack_masks` checks at batch sizes one and two.

The six old pending jobs were cancelled without ever starting:
1480496, 1480497, 1480498, 1480503, 1480504, and 1480505.  Replacement jobs
were submitted independently, with no SLURM dependency.  Kidney is an
inference-and-retention-only recovery from the existing epoch-10 checkpoint.

| Tissue | Old ID | New ID | Job type | Dependency | State after submission |
|---|---:|---:|---|---|---|
| Kidney | 1480494 (failed after training) | 1482332 | epoch-10 inference + retention | none | PENDING (Priority) |
| Breast | 1480496 (cancelled pending) | 1482333 | training + epoch-10 inference + retention | none | PENDING (Priority) |
| Pancreatic | 1480497 (cancelled pending) | 1482334 | training + epoch-10 inference + retention | none | PENDING (Priority) |
| Ovary | 1480498 (cancelled pending) | 1482335 | training + epoch-10 inference + retention | none | PENDING (Priority) |
| Colon | 1480503 (cancelled pending) | 1482336 | training + epoch-10 inference + retention | none | PENDING (Priority) |
| Lung | 1480504 (cancelled pending) | 1482337 | training + epoch-10 inference + retention | none | PENDING (Priority) |
| Skin | 1480505 (cancelled pending) | 1482338 | training + epoch-10 inference + retention | none | PENDING (Priority) |

Submission time was 2026-08-26T10:53:19+02:00.  All jobs use `gpua100`, one
GPU, eight CPUs and 64 GiB.  The training jobs retain the established
checkpoint-10-only post-inference policy.  The temporary submission exclusion
was preserved as `ruche-gpu[11-12,14,16-19]`, leaving the recently validated
gpu13/gpu15 nodes eligible; no job is pinned to either node.  Observed quota
before submission was 469/500 GiB and `run/` was 335 GiB.  With a maximum of
four concurrently running GPUs, the conservative checkpoint high-water is
about 491 GiB; successful Kidney retention will additionally remove its
temporary `model_best.pth` copy.

## Final Fold-A registry — 2026-08-27

The expanded campaign is no longer an eight-of-nine snapshot: all nine ordered
tissue directions are **COMPLETED VALID** at seed 42. Every row has the retained
epoch-10 checkpoint, canonical whole-held-out-slide inference, training and
inference efficiency records, and retention metadata.

| Tissue | Direction | Canonical job | Test patches | Status | Scientific role |
|---|---|---:|---:|---|---|
| Breast | `breast_s0 -> breast_s1` | 1482333 | 49,569 | COMPLETED VALID | MAIN PAPER tissue coverage |
| Colon | `colon_s1 -> colon_s2` | 1482336 | 11,903 | COMPLETED VALID | MAIN PAPER tissue coverage |
| Kidney | `kidney_s0 -> kidney_s1` | 1480494 training + 1482332 inference recovery | 4,449 | COMPLETED VALID | MAIN PAPER; generalist/specialist matched |
| Liver | `liver_s0 -> liver_s1` | 1480495 | 9,166 | COMPLETED VALID | MAIN PAPER; generalist/specialist matched |
| Lung | `lung_s1 -> lung_s3` | 1482337 | 20,621 | COMPLETED VALID | MAIN PAPER tissue coverage |
| Ovary | `ovary_s0 -> ovary_s1` | 1482335 | 25,498 | COMPLETED VALID | MAIN PAPER tissue coverage |
| Pancreatic | `pancreatic_s0 -> pancreatic_s1` | 1482334 | 12,647 | COMPLETED VALID | MAIN PAPER with historical seed/checkpoint caveat |
| Skin | `skin_s1 -> skin_s2` | 1482338 | 12,790 | COMPLETED VALID | MAIN PAPER tissue coverage |
| Tonsil | `tonsil_s0 -> tonsil_s1` | 1478431 | 21,083 | COMPLETED VALID | MAIN PAPER with historical seed/checkpoint and KLT-coverage caveats |

The nine final test rows and within-slide deltas are summarized in
`reports/tissue_specific_foldA_results_20260826.md`. The existing CSV comparison
is an earlier snapshot and still lacks the final Breast values.

For generalist versus specialist analysis, Kidney and Liver share both slide
identity and exact patch identity with KLT Fold A. Tonsil shares `tonsil_s1`
but not its complete patch set: KLT evaluates 9,879 packed test patches, while
the specialist evaluates 21,083. A matched analysis should use the common
identifiers already present in both inference JSONs; no retraining is needed.

### Complete historical within-slide composition

All nine historical preprocessing configs used 40x, the five-class mapping,
spatial x-axis 70/15/15 regions and a 128-coordinate-unit exclusion margin.
Breast, Colon, Lung, Pancreatic, Skin and Tonsil capped source material at
50,000 patches per slide; Kidney, Liver and Ovary did not. Both selected slides
contributed to every split:

| Tissue | Slides | Train / valid / test | Test patches by slide |
|---|---|---:|---|
| Breast | `s0,s1` | 71,875 / 14,024 / 9,443 | s0 2,668; s1 6,775 |
| Colon | `s1,s2` | 23,126 / 3,999 / 3,242 | s1 2,029; s2 1,213 |
| Kidney | `s0,s1` | 8,805 / 1,428 / 939 | s0 402; s1 537 |
| Liver | `s0,s1` | 21,301 / 5,653 / 2,639 | s0 1,647; s1 992 |
| Lung | `s1,s3` | 24,929 / 4,162 / 2,468 | s1 1,394; s3 1,074 |
| Ovary | `s0,s1` | 28,131 / 5,026 / 2,380 | s0 269; s1 2,111 |
| Pancreatic | `s0,s1` | 30,787 / 4,780 / 2,606 | s0 2,307; s1 299 |
| Skin | `s1,s2` | 18,114 / 3,755 / 2,130 | s1 648; s2 1,482 |
| Tonsil | `s0,s1` | 32,932 / 6,900 / 4,318 | s0 2,066; s1 2,252 |

The held-out-slide protocol retains seed 42, 10 epochs, optimizer/scheduler,
losses, augmentations, class-aware sampling, batch size 4, AMP, magnification
and post-processing. Unavoidable differences are: one slide supplies all
training/validation patches; the other supplies the complete packed test set;
validation is an approximately 85/15 spatial split inside the training slide;
and the realized patch counts differ substantially by direction and tissue.
