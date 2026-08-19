# Reviewer-response experiment and audit plan

Audit date: 2026-08-18. This is a read-only audit: no dataset, run, checkpoint, adapter, release artifact, or SLURM job was changed or launched.

## Executive conclusion

The submitted benchmark is a **within-slide spatial evaluation**, not a slide-independent evaluation. In every released KLT and tissue-specific dataset, each selected slide contributes train, validation, and test patches. The 128-pixel exclusion margin prevents direct boundary overlap, but it does not make the test distribution independent at slide level.

The smallest direct reviewer response is a two-fold KLT experiment in which each slide is tested exactly once and never participates in training, validation, checkpoint selection, threshold selection, or tuning for that fold. Run the raw CellViT-SAM-H x40 model, final-head linear probing, the selected LoRA Q/V + AdaptFormer + NP/HV/NT-head method, and FullFT. Start with seed 42 on Fold A, validate the split and evaluation, complete Fold B, and then add seeds 43 and 44 for the three trainable methods. The complete defensible KLT design is 18 training runs plus two deterministic raw-model evaluations; the seed-42 sanity stage is only three training runs plus one raw evaluation.

The audit used the release preprocessing/training configs, generated split manifests and patch tables, `run/` configs/logs/results, both adapter manifests, `utils/collect_inference_results.py`, `utils/collect_sthelar_run_tables.py`, `utils/collect_sthelar_run_tables_v2.py`, the type-analysis utilities, and the model/inference factories. The collection utilities recover epoch and aggregate inference metrics into `reports/runs_summary.csv` and `reports/epoch_metrics.csv`; they do not currently capture peak GPU memory or synchronized throughput.

## A. Existing experimental design

### Slides actually selected

The following IDs appear both in the release preprocessing YAML files under `configs/release/compayl2026/preprocessing/` and in the generated `patch_info_with_split.csv` files under `Datasets/cellvit_ready`.

| Experiment | Selected slide IDs |
|---|---|
| Breast | `breast_s0`, `breast_s1` |
| Colon | `colon_s1`, `colon_s2` |
| Kidney | `kidney_s0`, `kidney_s1` |
| Liver | `liver_s0`, `liver_s1` |
| Lung | `lung_s1`, `lung_s3` |
| Ovary | `ovary_s0`, `ovary_s1` |
| Pancreatic | `pancreatic_s0`, `pancreatic_s1` |
| Skin | `skin_s1`, `skin_s2` |
| Tonsil | `tonsil_s0`, `tonsil_s1` |
| KLT | `kidney_s0`, `kidney_s1`, `liver_s0`, `liver_s1`, `tonsil_s0`, `tonsil_s1` |

The source `STHELAR_40x/cell_metadata/index.csv` contains additional slides that were not selected for the nine canonical two-slide experiments: Breast has `s0,s1,s3,s6`; Pancreatic has `s0,s1,s2`; Skin has `s1,s2,s3,s4`. Colon, Kidney, Liver, Lung, Ovary, and Tonsil have the two listed slides. Thus “available in the source data” and “included in the submitted experiment” must be distinguished.

### How the current split is generated

The generated `split_manifest.yaml` files report `requested_strategy: spatial` and `actual_strategy: spatial`. The implementation is `assign_spatial_split_single_slide()` in `preprocessing/sthelar/convert_hf_to_cellvit.py`:

1. Compute patch centers from `xmin,ymin,xmax,ymax`.
2. Group patches by `slide_id`; splitting is performed independently inside every slide.
3. Along the x axis, place boundaries at 70% and 85% of that slide's center-coordinate span.
4. Assign train at or left of `b1 - 128`, validation between `b1 + 128` and `b2 - 128`, and test at or right of `b2 + 128`.
5. Boundary-band patches are discarded before the final manifest is written.

The generated metadata confirms that every selected slide has nonzero train, validation, and test rows. KLT caps the sampled input at 10,000 patches per slide; several tissue-specific datasets cap it at 50,000. This is sound spatial de-overlap, but it allows slide-specific staining, scanner, preparation, and morphology characteristics to occur on both sides of the split.

### Is image/label regeneration required for slide-independent folds?

No. The existing `cellvit_ready` products contain sufficient metadata and packed data to create new fold definitions without regenerating pixels or labels:

- `patch_info_with_split.csv` has `slide_id`, `file_name`, packed image/label names, `xmin,ymin,xmax,ymax`, `x_center,y_center`, and the current split.
- `images.zip` and `labels.zip` contain the packed patches and labels.
- `types.csv` stores the class taxonomy.
- `cell_count_train.csv`, `cell_count_valid.csv`, and `cell_count_test.csv` are the fold-specific index files consumed by `cell_segmentation/datasets/pannuke.py`.
- `dataset_config.yaml` and `split_manifest.yaml` record construction and split provenance.
- Upstream `STHELAR_40x/patches_overview_sthelar40x.parquet` and `STHELAR_40x/cell_metadata/index.csv` retain source slide/patch provenance.

A future implementation should create **new fold dataset directories**, reuse the immutable ZIP/type data, and generate new fold-specific CSV/manifests. Existing directories must not be edited. Before training, assert that the sets of slide IDs in train, validation, and test are disjoint and save the assertion result with the run.

## B. Slide-independent evaluation

### Recommended KLT two-fold protocol

| Fold | Train/validation source slides | Test-only slides |
|---|---|---|
| A | `kidney_s0`, `liver_s0`, `tonsil_s0` | `kidney_s1`, `liver_s1`, `tonsil_s1` |
| B | `kidney_s1`, `liver_s1`, `tonsil_s1` | `kidney_s0`, `liver_s0`, `tonsil_s0` |

Within each training slide, form train/validation regions with the existing spatial-margin method (recommended 85/15 inside the training slide, as already supported by `assign_train_valid_inside_train_slides()`). Test slides are assigned wholly to test. They must not be used for early stopping, checkpoint selection, threshold selection, normalization fitting, qualitative patch selection, or any hyperparameter decision.

Use the same patch cap policy, label mapping, augmentation, epochs, optimizer, and fixed post-processing settings across methods and folds. If a detection threshold must be selected, select it independently on training-slide validation data in each fold. For the raw model, prefer a fixed published/default threshold; otherwise tune only on training-slide validation data and freeze it before test inference.

### Comparison with the submitted split

Keep the current within-slide results as the in-domain reference and add the slide-independent results as a paired robustness test. Report, for each method:

- pooled metrics for continuity with the paper;
- per-test-slide and per-tissue metrics;
- the within-slide to held-out-slide delta;
- mean and interval across seeds for trainable methods;
- a paired analysis whose independent units are slides (or tissues), not patches.

With only two slides per KLT tissue, both folds are needed: they prevent the result from depending on whether `s0` or `s1` happened to be selected as test. A patch-level significance test would overstate sample size because patches from one slide are correlated.

### Tissues with more than two source slides

For Breast (4 source slides), Pancreatic (3), and Skin (4), the most defensible expanded-cohort design is leave-one-slide-out cross-validation. If compute requires two folds, assign whole slides to balanced, predeclared folds and rotate all slides through test; do not silently omit extra slides while claiming all-slide generalization. A cheaper alternative is to retain the canonical two-slide paired protocol for comparability and use the additional slides once as a completely external test set, with no tuning on them.

### Methods to rerun

1. **Raw CellViT-SAM-H x40:** inference only, no STHELAR training. This directly supplies the missing pretrained baseline.
2. **Final-head linear probing:** establishes the value of changing only the STHELAR output taxonomy/heads.
3. **Selected PEFT:** LoRA Q/V + AdaptFormer + final NP/HV/NT heads, matching the released method.
4. **FullFT:** the upper-capacity reference under exactly the same held-out slides.

### Staged schedule and seeds

1. **Seed-42 sanity:** Fold A only; three training runs (LP, PEFT, FullFT) and one raw-model inference evaluation. Verify manifests, slide disjointness, label counts, checkpoint selection, threshold provenance, and metric parity on a tiny dry-run subset before the full evaluation.
2. **Complete seed 42:** only after the sanity checks pass, run Fold B: three more training runs and one raw-model evaluation.
3. **Replication:** run seeds 43 and 44 for LP, PEFT, and FullFT on both folds: 12 additional training runs. The raw checkpoint need not be repeated unless the inference pipeline is stochastic.

Three independent seeds (42, 43, 44) are the minimum defensible choice for trainable methods. Report all seed/fold points rather than only mean and standard deviation. With six KLT slides, uncertainty should also be estimated at slide level, for example by a paired slide bootstrap, while acknowledging that six slides remain a small biological sample.

### Should PEFT and FullFT be repeated slide-independently for all nine tissues?

Not as the first reviewer-response experiment. Two folds × nine tissues × two methods equals **36 training runs per seed**, or **108 training runs for three seeds**, before raw/LP baselines. Existing complete tissue jobs range from roughly 3 to 24 hours of SLURM elapsed time depending on tissue and GPU; the full three-seed campaign is feasible as a scheduled cluster campaign but disproportionate to the smallest scientific question and vulnerable to hardware confounding.

First establish the result on KLT. If expansion is warranted, run seed-42 PEFT on all tissue folds, then add FullFT and replication only for a predeclared heterogeneous subset (for example Kidney, Breast, and Pancreatic/Skin) or where the KLT result reveals the largest generalization gap. A full nine-tissue campaign becomes justified only if the revised paper intends to make tissue-by-tissue slide-generalization claims.

## C. Baseline audit

### Repository support and engineering cost

The model factory in `cell_segmentation/experiments/experiment_cellvit_pannuke.py` already supports `CellViT256` and SAM-B/L/H variants. The inference factory in `cell_segmentation/inference/inference_cellvit_experiment_pannuke.py` supports `CellViT`, `CellViT256`, and `CellViTSAM`, and the adapter utilities operate over the encoder block structure. `models/pretrained/CellViT-256-x40.pth` is present (187,224,155 bytes). A completed ten-epoch ViT-256 liver log/config also exists, although it used an older pre-margin128 liver split and is not a paper-ready comparison.

| Baseline | Scientific value | Engineering cost | Audit conclusion |
|---|---:|---:|---|
| Raw CellViT-SAM-H x40 | Very high | Very low | Add an inference-only path/config and prevent accidental STHELAR head training. The current frozen run is useful evidence but is not a cleanly labeled raw-taxonomy evaluation. |
| Raw CellViT-256 x40 | High | Low | Checkpoint and loader already exist; add the same inference-only evaluation and shape/load assertions. |
| CellViT-256 LP/selected PEFT/FullFT | High | Low–moderate | Generic model/adapters already work and a ViT-256 adapted run completed. Work is chiefly release-grade configs, KLT split wiring, and smoke/metric validation. |
| HoVer-Net | Moderate–high conventional anchor | High | No full model/checkpoint/evaluation integration is present; only derived post-processing code exists. Requires environment, checkpoint, normalization, output conversion, and evaluator validation. |
| CellViT++ | Potentially high modern external anchor | High–very high | Not implemented locally and may introduce pretraining/task/taxonomy differences. Use only after the internal raw/256 baselines. |

Priority is therefore: raw SAM-H, raw CellViT-256, adapted CellViT-256, HoVer-Net, then CellViT++ (unless an official compatible inference package materially lowers the last two costs).

### Taxonomy constraint for raw pretrained models

PanNuke's semantic classes and STHELAR's classes are not equivalent. The fact that both checkpoints/configurations can have the same number of output channels does not establish semantic compatibility. No label mapping should be inferred from tensor shape.

Valid raw-model comparisons are:

- class-agnostic detection F1;
- bPQ / class-agnostic PQ, and class-agnostic DQ/SQ if reported;
- binary nuclear segmentation metrics such as Dice/Jaccard where computed consistently.

Not directly comparable without an independently justified mapping are:

- STHELAR F1 type, type accuracy, or per-STHELAR-class F1;
- class-aware mPQ or per-class PQ.

Final-head LP, PEFT, and FullFT are trained against the declared STHELAR mapping, so their STHELAR type metrics are valid.

### Type-error analysis to add

The existing KLT type summaries already identify a failure mode hidden by aggregate detection F1. At seed 42, selected PEFT and FullFT have similar Epithelial F1 (0.8774 vs 0.8779) and PEFT slightly exceeds FullFT on Immune (0.8411 vs 0.8378), but PEFT final-head performance is lower on Stromal (0.5243 vs 0.5470) and nearly collapses on Other (0.0021 vs 0.4097). Melanocyte has zero support in this KLT test result and must not enter an unqualified macro average. Decoder last-stage adaptation substantially recovers Other in the existing ablation, suggesting the gap is localized to type representation/capacity rather than detection alone.

For the revision, add per-slide confusion matrices, per-class precision/recall/F1 with support, present-class macro averages, matched-detection type accuracy, and false-negative/false-positive breakdowns. Qualitative overlays should be selected by predeclared error strata, not by favorable appearance, and should focus on Stromal/Other confusion and rare/unsupported classes.

## D. Computational-efficiency audit

### What can be reconstructed now

| Quantity | Retrospective status | Evidence/limitation |
|---|---|---|
| SLURM job ID | Yes for submitted jobs | Encoded in `logs/*.out`/`*.err`; `sacct` records remain available for sampled IDs. |
| SLURM elapsed time | Yes | Includes setup and often export/inference, so it is not pure training time. |
| Training wall time and seconds/epoch | Yes, approximately | `logs.log` timestamps at each epoch and `Finished run`; excludes/combines some initialization and validation details. |
| GPU partition/type | Partial | Submission scripts identify A100 partition or generic/V100 jobs; exact device name is not logged for every training run. Fisher/inference logs explicitly show NVIDIA A100-SXM4-40GB. |
| Batch size | Yes | Canonical release configs use batch size 4. |
| Mixed precision | Yes | Canonical configs set `mixed_precision: true`; the trainer uses float16 autocast/GradScaler. |
| Trainable/full parameters | Yes | Existing logs record both. |
| Checkpoint/adapter storage | Yes | Can be measured exactly with `stat`; checkpoint contents include optimizer/training state and should not be equated with deployable adapter size. |
| Peak CUDA memory | No | No peak allocated/reserved measurement is logged. |
| Inference throughput | No | No stable image/patch count per synchronized inference interval is logged. |

Representative current evidence:

- KLT FullFT seed 42, job `1151141`: SLURM elapsed 14:17:20; log-derived ten-epoch training interval 14.06 h, mean 84.39 min/epoch.
- KLT selected PEFT seed 42, job `1151148`: SLURM elapsed 12:03:05; log-derived training interval 11.85 h, mean 71.07 min/epoch.
- KLT final-head LP seed 42, job `1209325` on the V100 submission path: SLURM elapsed 17:16:19; log-derived interval 16.95 h, mean 101.72 min/epoch. It must not be used as a same-hardware speed comparison with the A100 runs.
- Sample tissue SLURM elapsed times include Kidney FullFT/PEFT 3:04:27/2:47:28 (`1172319`/`1173368`), Breast FullFT/PEFT 23:52:34/21:09:03 (`1186380`/`1186381`), and a seed-43 Breast V100 timeout at 24:00:10 (`1193122`). Tonsil FullFT job `1208016` was cancelled after 16:19:50 and resumed in `1209326` for 10:59:05; these segments cannot be presented as one clean timed run.
- Typical current full training checkpoints are about 8.398 GB decimal for FullFT, 2.895 GB for selected PEFT, and 2.799 GB for frozen/final-head runs. The released selected adapters are about 31.9 MB, illustrating why deployable adapter size must be reported separately from resumable training checkpoints.
- Logged SAM-H counts are approximately 699,736,523 total parameters; selected PEFT has 707,644,395 total and 7,908,779 trainable (1.1176%); final-head LP has 650 trainable; FullFT trains 699,736,523.

Elapsed-time evidence is heterogeneous across A100 and V100 jobs and across fresh/resumed jobs. It supports retrospective descriptive reporting, not a clean causal efficiency comparison.

### Minimal instrumentation for future matched runs

Without changing the scientific training procedure, future matched runs should write one machine-readable JSON/CSV sidecar containing:

- hardware/device name, node, CUDA/PyTorch versions, batch size, workers, AMP mode, seed, fold, and numbers of train/validation/test patches;
- `torch.cuda.reset_peak_memory_stats()` after initialization, followed by `max_memory_allocated()` and `max_memory_reserved()` for training; repeat separately for inference;
- synchronized `perf_counter()` timestamps for total training and each epoch, reporting seconds/epoch;
- synchronized inference timing after a short warm-up, processed patch/image count, patches/s, images/s, batch size, and whether post-processing/data loading are included;
- exact trainable and full parameter counts using `numel`, already largely logged;
- byte sizes for best checkpoint, resumable checkpoint, exported adapter, and any prediction bundle.

Run PEFT and FullFT back-to-back on the same GPU class with identical batch size, data workers, validation cadence, and data manifest. Report torch allocated and reserved peaks explicitly; neither is the same as `nvidia-smi` process memory.

## Prioritized experiment matrix

Counts below treat training runs separately from inference-only evaluations and are incremental by row.

| Priority | Experiment | Scientific question | Reviewer criticism addressed | New runs | Compute relative to existing runs | Expected value |
|---:|---|---|---|---:|---|---|
| 1 | KLT Fold A seed-42 sanity: raw + LP + selected PEFT + FullFT | Does the pipeline enforce true held-out-slide testing, and does the ranking survive once? | 1, 2 | 3 train + 1 inference | About one LP + one PEFT + one FullFT | Essential validity gate; highest immediate value |
| 2 | KLT Fold B seed 42 | Is the result robust to reversing which slide is held out? | 1 | 3 train + 1 inference | Same as priority 1 | Completes the direct two-fold reviewer response |
| 3 | KLT seeds 43/44, both folds, LP/PEFT/FullFT | Are method differences stable across initialization? | 1, 4 | 12 train | Two additional seeds × six trainable fold/method combinations | Minimum defensible replicated slide-independent result |
| 4 | Raw CellViT-256 x40 on both KLT folds | Does backbone size explain performance without adaptation? | 2 | 2 inference | Much less than one training run | High-value, low-cost missing baseline |
| 5 | CellViT-256 selected PEFT, seed 42, both folds | Does PEFT remain effective on a smaller backbone? | 2 | 2 train | Expected below SAM-H per run; measure rather than assume | Strong architecture/efficiency evidence |
| 6 | Matched instrumented SAM-H LP/PEFT/FullFT efficiency runs | What are memory, time, throughput, and storage trade-offs on identical hardware? | 3 | 3 train plus inference timing | One matched run per method; may overlap priority 1 if instrumentation is ready | Converts the PEFT efficiency claim from parameter-only to operational evidence |
| 7 | Per-class/per-slide error and qualitative analysis | Why does detection remain competitive while F1 type lags? | 5 | 0 train | Reuse preserved predictions/results; targeted re-inference only if records are insufficient | High interpretive value at low compute cost |
| 8 | Seed-42 all-nine tissue PEFT/FullFT two-fold campaign | Are slide-generalization effects tissue-dependent? | 1, 4 | 36 train | Roughly twice the current 18 tissue method runs per seed | Valuable supplement, but defer until KLT validates the design |
| 9 | Three-seed all-nine campaign | Are tissue-specific conclusions statistically replicated? | 1, 4 | 108 train total | Large multi-week GPU campaign depending on concurrency | Highest coverage, low priority-to-cost ratio |
| 10 | HoVer-Net or CellViT++ external baseline | Does the result hold against an external architecture/training ecosystem? | 2 | Undetermined | High engineering plus training/inference | Optional after internal baselines; useful only with rigorously matched evaluation |

## Recommended paper claim after completion

Present the original spatial-margin benchmark as within-slide performance and the two-fold KLT experiment as the primary slide-independent generalization result. Do not imply patient/slide independence from the 128-pixel margin. Keep raw-model semantic claims class-agnostic, report operational efficiency only from matched instrumented hardware, and frame the remaining tissue-wide/statistical limitations explicitly if the deferred campaigns are not run.
