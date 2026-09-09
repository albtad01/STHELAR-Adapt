# NeurIPS compute-resource audit (read-only)

Audit cutoff: **2026-09-03 16:16:01 CEST**. This audit made no change to an experiment, scheduler state, canonical scientific artifact, or manuscript file. The only written artifact is this report.

## Executive result

| Quantity | GPU-hours | Interpretation |
|---|---:|---|
| A. Final paper-facing training allocations | **802.701** | 114 unique one-GPU training allocations; includes setup, validation, and inline final inference because Slurm accounted them in the same allocation |
| B. Separate paper-facing evaluation/diagnostic allocations | **3.172** | Four Frozen evaluations, one Kidney-A inference recovery, and one Fisher diagnostic; additive to A |
| Timed inference inside paper-facing training allocations | **57.487** | Measured for 95 of 113 jobs that performed inference inline; already contained in A and **not** added again |
| C. Non-paper/preliminary project allocations to cutoff | **971.362** | 968.071 terminal-job GPU-hours plus 3.291 GPU-hours consumed by active noncanonical timing job 1614903 at the cutoff |
| D. All observed project GPU allocations to cutoff | **1,777.235** | A + B + C; conservative project lower bound |

Rounded paper-safe values are therefore **about 803 GPU-hours of final training**, **about 3.2 additional GPU-hours of standalone final evaluation**, **about 971 GPU-hours of preliminary/other GPU allocation**, and **about 1,777 GPU-hours for the project through the audit cutoff**. The 57.487 hours of measured inline inference must not be added to these totals because it occurred inside the 802.701 training-job allocations.

All final held-out-slide training/evaluation allocations used the `gpua100` partition and saved metadata identifies **NVIDIA A100-SXM4-40GB**. The historical KLT selection ablation is the exception: jobs **1209325** and **1227083** used the generic/V100 submission path (37.593 GPU-hours combined). Repository provenance identifies that path as 32-GB V100, but the exact NVIDIA device string was not recorded, so this report does not invent a more specific model. All other paper-facing allocations (768.280 GPU-hours, including the Fisher diagnostic) were A100 allocations.

## Scope, evidence, and accounting rule

Paper-facing membership was reconstructed from `reports/workshop_master_results.csv`, `reports/workshop_latex_tables.tex`, `reports/workshop_ablation_seed_audit.csv`, `reports/klt_typing_peft_strategy_analysis.md`, the timestamped canonical result/config paths, and the campaign/retention manifests under `reports/`. Slurm fields were recovered with read-only `sacct` queries (`JobID`, `State`, `ElapsedRaw`, `AllocTRES`, `Partition`, `NodeList`, `Start`, `End`, and, where needed, `SubmitLine`).

GPU-hours follow the requested rule exactly:

> allocated GPUs x elapsed Slurm wall-clock hours.

Every paper-facing job allocated one GPU. Array task IDs are reported in Slurm's displayed form (for example, `1596567_0`); the corresponding raw accounting IDs differ but were not counted a second time. Batch/extern steps were excluded. A training wrapper normally performs validation and final checkpoint-10 inference before exiting, so its full Slurm allocation is classified as a training allocation. Inline inference is also reported as a measured sub-phase, but never added again.

The repository does not contain the current complete manuscript source. The accessible main-text fragment still contains a superseded within-slide tissue table, while the canonical final registry contains the reciprocal held-out-slide matrices requested for this audit. The primary total above treats the canonical final registry as the current paper. If the legacy within-slide tissue table is still present in the submitted PDF, see the conditional adjustment below.

## 1. Paper-facing compute

### Totals by experiment family

| Family | Training jobs | Slurm `COMPLETED` | Slurm `FAILED` after usable result | GPU-hours |
|---|---:|---:|---:|---:|
| Historical within-slide KLT PEFT-selection ablation | 18 | 18 | 0 | 238.005 |
| Reciprocal KLT, SAM-H: LP/PEFT/FullFT, 3 seeds x 2 folds | 18 | 16 | 2 | 136.410 |
| Reciprocal KLT, CellViT-256: LP/PEFT/FullFT, 3 seeds x 2 folds | 18 | 16 | 2 | 113.495 |
| Nine-tissue reciprocal SAM-H Selected-PEFT, seed 42 | 18 | 15 | 3 | 99.905 |
| Nine-tissue reciprocal SAM-H FullFT, seed 42 | 18 | 18 | 0 | 110.446 |
| Nine-tissue reciprocal CellViT-256 Selected-PEFT, seed 42 | 18 | 18 | 0 | 82.010 |
| Tissue-specific CellViT-256 FullFT controls (K/L/T), seed 42 | 6 | 6 | 0 | 22.431 |
| **Total training** | **114** | **107** | **7** | **802.701** |

The seven scheduler-`FAILED` allocations are scientifically complete, not failed training results. Jobs 1524187, 1524189, 1524192, 1524193, 1524196, and 1524197 had already completed training and canonical inference before a later adapter-export step failed. Job 1480494 completed ten training epochs but failed on the singleton final-inference batch; inference-only job 1482332 recovered its checkpoint without retraining. Thus **114/114 training allocations produced the numerical result assigned to them**, although only 107 ended in Slurm state `COMPLETED`.

No additional unique GPU allocation could be mapped to a retained numerical row in the canonical main/appendix result registry. CPU-only consolidation, cleanup, and table-generation jobs are outside this GPU audit.

There is no missing Slurm elapsed time or GPU count in the 114-job ledger, so the paper-facing training minimum and maximum are both 802.701 GPU-hours, subject only to the manuscript-membership ambiguity described next.

### Conditional legacy-table adjustment

The stale accessible main-text fragment contains a nine-tissue **within-slide spatial** PEFT/FullFT table that the final held-out-slide registry supersedes. Its 18 conditions used 19 allocations because the Tonsil FullFT run was resumed: **217.349 GPU-hours** (201.019 completed + 16.331 cancelled/resumed; 116.415 A100-partition + 100.935 V100-path). These hours are in C, not A.

If that old numerical table remains in the actual submitted paper, reclassify those 217.349 hours as paper-facing: A becomes **1,020.051 GPU-hours**, C becomes **754.013 GPU-hours**, and D is unchanged. This is a document-scope adjustment, not timing uncertainty.

### Separate inference/evaluation allocations

| Experiment/config identifier | Job ID | GPU | GPUs | Elapsed | GPU-hours | Slurm status | Paper result? |
|---|---:|---|---:|---:|---:|---|:---:|
| SAM-H Frozen KLT Fold A | 1465774 | A100-SXM4-40GB | 1 | 00:40:55 | 0.682 | COMPLETED | yes |
| SAM-H Frozen KLT Fold B | 1465815 | A100-SXM4-40GB | 1 | 00:50:48 | 0.847 | COMPLETED | yes |
| CellViT-256 Frozen KLT Fold A | 1475937 | A100-SXM4-40GB | 1 | 00:38:12 | 0.637 | COMPLETED | yes |
| CellViT-256 Frozen KLT Fold B | 1475938 | A100-SXM4-40GB | 1 | 00:47:31 | 0.792 | COMPLETED | yes |
| SAM-H tissue Kidney-A checkpoint-10 inference recovery | 1482332 | A100-SXM4-40GB | 1 | 00:07:39 | 0.128 | COMPLETED | yes |
| KLT FullFT seed-43 Fisher diagnostic | 1170754 | A100 partition; log identifies A100-SXM4-40GB | 1 | 00:05:13 | 0.087 | COMPLETED | yes |
| **Standalone evaluation/diagnostic total** |  |  |  |  | **3.172** |  |  |

Canonical `inference_efficiency_metrics.json` files provide another **57.583 GPU-hours** of direct inference timing for all 96 final-registry trainable conditions. Of this, 0.095 hours belongs to the separate recovery job above and **57.487 hours** occurred inline in 95 training allocations. Historical-ablation inference sub-times were not instrumented. At the observed 8--10 patches/s and 4,653 test patches, their 18 inline tests would add only roughly 2--3 hours of inference activity, still already contained in A. A fair inference statement is therefore: **at least 60.7 GPU-hours of paper-result inference/diagnostics are reconstructable, approximately 63 GPU-hours including the legacy-ablation estimate, but only 3.172 GPU-hours are additive allocations outside A.**

## 2. Exploratory / preliminary compute

After subtracting the 120 paper-facing allocations above from every GPU allocation whose Slurm `WorkDir` is this repository, 236 non-paper allocations remain.

| Non-paper scheduler status | Allocations | GPU-hours to cutoff | Examples/interpretation |
|---|---:|---:|---|
| COMPLETED | 118 | 799.667 | exploratory ablations, superseded/model-best-era training, preprocessing, extra variants/seeds, and evaluations not retained in final tables |
| FAILED | 94 | 75.141 | debugging, configuration failures, CUDA/device failures, and superseded partial attempts |
| NODE_FAIL | 8 | 2.120 | infrastructure failures |
| CANCELLED | 12 | 19.130 | includes meaningful partial runs as well as near-zero queued cancellations |
| TIMEOUT | 3 | 72.013 | long incomplete exploratory runs |
| RUNNING at cutoff | 1 | 3.291 | noncanonical timing job 1614903; four GPUs allocated, so this grows by 4 GPU-hours per wall hour |
| **Total non-paper to cutoff** | **236** | **971.362** |  |

The identifiable superseded nine-tissue within-slide/model-best-era campaign alone accounts for **217.349 GPU-hours** as described above. Other identifiable preliminary families include May prototypes, June adapter/decoder searches and preprocessing, V100 tissue variants, extra seeds and model-best experiments, August infrastructure retries that did not yield the retained result, and the September noncanonical same-node timing audit. The non-paper remainder by project phase is:

| Start month | Non-paper GPU-hours | Main character of work |
|---|---:|---|
| 2026-05 | 47.247 | early prototypes, preprocessing, short failures/debugging |
| 2026-06 | 708.146 | broad preliminary ablations, tissue/model variants, preprocessing, failed and timed-out runs |
| 2026-07 | 169.548 | tissue/model-best-era completions, V100 fallbacks/resume, extra adapter variants |
| 2026-08 | 43.095 | held-out-slide infrastructure failures, superseded retries, extra nonfinal work |
| 2026-09 through cutoff | 3.326 | cancelled and active noncanonical four-GPU timing allocations |
| **Total** | **971.362** |  |

This 971.362-hour figure is a **conservative lower bound and the best Slurm-based estimate**, not a claim of globally exhaustive lifetime energy use. It is exact for allocations recorded since 2026-01-01 with this repository as `WorkDir`, but cannot capture local interactive GPU use, jobs launched from another working directory, deleted/expired accounting outside the queried interval, or compute performed by collaborators under other accounts. No defensible finite maximum can be inferred from the available evidence. Reporting a narrow upper range would imply false precision.

## 3. Total project compute and hardware conclusion

At the cutoff, Slurm recorded 355 terminal GPU allocations totaling **1,773.944 GPU-hours**, plus active job 1614903 at **3.291 GPU-hours**, for **1,777.235 GPU-hours** observed. The project-total paper-safe estimate is therefore **approximately 1,777 GPU-hours through 2026-09-03 16:16 CEST**, explicitly a lower bound and a time-stamped snapshot.

Not all project jobs used A100-SXM4-40GB. Paper-facing held-out-slide work did; two reported historical-ablation jobs used the V100 submission path. Preliminary work used both `gpua100` and generic/V100 paths: after paper-facing subtraction, **521.569 GPU-hours** were on `gpua100` and **449.793 GPU-hours** were on the generic/V100 path through the cutoff. Exact device strings are absent for many early generic-partition jobs.

## 4. Checklist / paper-safe wording

### NeurIPS checklist justification (1--2 sentences)

> **Yes.** Appendix X reports the worker type and memory, per-run wall-clock GPU usage, and aggregate compute for every reported experiment. It also distinguishes the approximately 803 GPU-hours of final training (plus 3.2 GPU-hours of separate evaluation) from approximately 971 GPU-hours of preliminary, failed, superseded, preprocessing, and other non-paper allocations, for approximately 1,777 project GPU-hours through the stated audit cutoff.

### Compact appendix paragraph

> Experiments were run on an internal Slurm cluster, normally with one NVIDIA A100-SXM4-40GB GPU, 8 CPU cores, and 64 GiB host memory per job; two historical KLT selection runs used the cluster's 32-GB V100 submission path (the exact device string was not retained). Using allocated GPUs multiplied by Slurm elapsed wall time, the final reported training runs consumed approximately 803 GPU-hours, with 3.2 additional GPU-hours in standalone evaluation/diagnostic jobs; preliminary, failed, superseded, preprocessing, and other non-paper work consumed approximately 971 additional GPU-hours, giving approximately 1,777 GPU-hours for the project through 3 September 2026. Per-run times are provided in Appendix Table X; retained model states range from approximately 0.17 to 7.82 GiB and verified SAM-H/CellViT-256 adapters are approximately 30.4/1.55 MiB.

If the legacy within-slide tissue table remains, replace “803” with “1,020” and “971” with “754”; the 1,777 project total stays the same.

### Is `\answerYes{}` defensible?

**Yes, conditionally after the disclosure is actually added to the paper/appendix and the per-run ledger (or an equivalent compact per-run table) is included or referenced.** The official 2026 checklist asks for worker type plus relevant memory/storage, compute for individual experimental runs, an estimated total, and disclosure that the full project consumed more compute than the final reported experiments ([official NeurIPS 2026 checklist, item 8](https://neurips.cc/public/guides/PaperChecklist)). The wording and ledger in this audit meet those points. Merely adding the aggregate one-paragraph wording while omitting per-run times would not fully meet the literal individual-run guidance. The manuscript should also correct its blanket A100 statement or explicitly carve out the two historical V100-path jobs.

## Appendix A: deduplicated paper-facing training-job ledger

All rows below produced a result used in the canonical paper registry. `FAILED*` means Slurm reported failure only after usable training output (and, except for 1480494, canonical inference) existed; see the note above. GPU-hours equal elapsed hours because every listed training job allocated one GPU.

### Historical KLT PEFT-selection ablation

| Experiment/config identifier | Job ID | GPU | GPUs | Elapsed | GPU-hours | Slurm status | Paper result? |
|---|---:|---|---:|---:|---:|---|:---:|
| KLT FullFT, seed 42 | 1151141 | A100-SXM4-40GB | 1 | 14:17:20 | 14.289 | COMPLETED | yes |
| KLT decoder conv only, seed 42 | 1151142 | A100-SXM4-40GB | 1 | 10:54:50 | 10.914 | COMPLETED | yes |
| KLT LoRA only, seed 42 | 1151143 | A100-SXM4-40GB | 1 | 12:07:33 | 12.126 | COMPLETED | yes |
| KLT AdaptFormer only, seed 42 | 1151144 | A100-SXM4-40GB | 1 | 11:18:22 | 11.306 | COMPLETED | yes |
| KLT VeRA only, seed 42 | 1151145 | A100-SXM4-40GB | 1 | 11:42:20 | 11.706 | COMPLETED | yes |
| KLT LoRA + AdaptFormer conv, seed 42 | 1151146 | A100-SXM4-40GB | 1 | 13:14:07 | 13.235 | COMPLETED | yes |
| KLT VeRA + AdaptFormer conv, seed 42 | 1151147 | A100-SXM4-40GB | 1 | 13:37:25 | 13.624 | COMPLETED | yes |
| KLT selected LoRA + AdaptFormer heads, seed 42 | 1151148 | A100-SXM4-40GB | 1 | 12:03:05 | 12.051 | COMPLETED | yes |
| KLT LoRA + AdaptFormer last stage, seed 42 | 1151149 | A100-SXM4-40GB | 1 | 12:16:38 | 12.277 | COMPLETED | yes |
| KLT FullFT, seed 43 | 1169677 | A100-SXM4-40GB | 1 | 13:45:56 | 13.766 | COMPLETED | yes |
| KLT LoRA + AdaptFormer conv, seed 43 | 1169678 | A100-SXM4-40GB | 1 | 13:25:36 | 13.427 | COMPLETED | yes |
| KLT selected LoRA + AdaptFormer heads, seed 43 | 1169679 | A100-SXM4-40GB | 1 | 12:21:10 | 12.353 | COMPLETED | yes |
| KLT LoRA + AdaptFormer last stage, seed 43 | 1172353 | A100-SXM4-40GB | 1 | 12:24:43 | 12.412 | COMPLETED | yes |
| KLT VeRA + AdaptFormer conv, seed 43 | 1172354 | A100-SXM4-40GB | 1 | 13:30:54 | 13.515 | COMPLETED | yes |
| KLT VeRA + AdaptFormer heads, seed 43 | 1172355 | A100-SXM4-40GB | 1 | 12:45:31 | 12.759 | COMPLETED | yes |
| KLT final-head LP, seed 42 | 1209325 | V100 path (32 GB) | 1 | 17:16:19 | 17.272 | COMPLETED | yes |
| KLT AdaptFormer + heads, seed 42 | 1212870 | A100-SXM4-40GB | 1 | 10:39:13 | 10.654 | COMPLETED | yes |
| KLT AdaptFormer + heads, seed 43 | 1227083 | V100 path (32 GB) | 1 | 20:19:16 | 20.321 | COMPLETED | yes |

### Reciprocal KLT: SAM-H

| Experiment/config identifier | Job ID | GPU | GPUs | Elapsed | GPU-hours | Slurm status | Paper result? |
|---|---:|---|---:|---:|---:|---|:---:|
| KLT SAM-H PEFT, fold A, seed 42 | 1465776 | A100-SXM4-40GB | 1 | 08:37:46 | 8.629 | COMPLETED | yes |
| KLT SAM-H FullFT, fold A, seed 42 | 1465777 | A100-SXM4-40GB | 1 | 09:35:28 | 9.591 | COMPLETED | yes |
| KLT SAM-H LP, fold A, seed 42 | 1465814 | A100-SXM4-40GB | 1 | 06:22:15 | 6.371 | COMPLETED | yes |
| KLT SAM-H LP, fold B, seed 42 | 1465816 | A100-SXM4-40GB | 1 | 05:55:29 | 5.925 | COMPLETED | yes |
| KLT SAM-H FullFT, fold B, seed 42 | 1465822 | A100-SXM4-40GB | 1 | 08:04:04 | 8.068 | COMPLETED | yes |
| KLT SAM-H PEFT, fold B, seed 42 | 1468272 | A100-SXM4-40GB | 1 | 07:56:26 | 7.941 | COMPLETED | yes |
| KLT SAM-H LP, fold A, seed 43 | 1468274 | A100-SXM4-40GB | 1 | 06:28:37 | 6.477 | COMPLETED | yes |
| KLT SAM-H LP, fold B, seed 43 | 1468275 | A100-SXM4-40GB | 1 | 05:57:02 | 5.951 | COMPLETED | yes |
| KLT SAM-H PEFT, fold A, seed 43 | 1475933 | A100-SXM4-40GB | 1 | 07:26:59 | 7.450 | COMPLETED | yes |
| KLT SAM-H PEFT, fold B, seed 43 | 1475934 | A100-SXM4-40GB | 1 | 06:43:31 | 6.725 | COMPLETED | yes |
| KLT SAM-H FullFT, fold A, seed 43 | 1475941 | A100-SXM4-40GB | 1 | 09:34:24 | 9.573 | COMPLETED | yes |
| KLT SAM-H FullFT, fold B, seed 43 | 1475942 | A100-SXM4-40GB | 1 | 08:29:36 | 8.493 | COMPLETED | yes |
| KLT SAM-H PEFT, fold A, seed 44 | 1524187 | A100-SXM4-40GB | 1 | 07:58:27 | 7.974 | FAILED* | yes |
| KLT SAM-H PEFT, fold B, seed 44 | 1524189 | A100-SXM4-40GB | 1 | 07:13:46 | 7.229 | FAILED* | yes |
| KLT SAM-H FullFT, fold A, seed 44 | 1524190 | A100-SXM4-40GB | 1 | 09:14:47 | 9.246 | COMPLETED | yes |
| KLT SAM-H FullFT, fold B, seed 44 | 1524191 | A100-SXM4-40GB | 1 | 08:09:16 | 8.154 | COMPLETED | yes |
| KLT SAM-H LP, fold A, seed 44 | 1570493 | A100-SXM4-40GB | 1 | 06:18:59 | 6.316 | COMPLETED | yes |
| KLT SAM-H LP, fold B, seed 44 | 1570494 | A100-SXM4-40GB | 1 | 06:17:43 | 6.295 | COMPLETED | yes |

### Reciprocal KLT: CellViT-256

| Experiment/config identifier | Job ID | GPU | GPUs | Elapsed | GPU-hours | Slurm status | Paper result? |
|---|---:|---|---:|---:|---:|---|:---:|
| KLT CellViT-256 FullFT, fold A, seed 42 | 1473865 | A100-SXM4-40GB | 1 | 06:46:55 | 6.782 | COMPLETED | yes |
| KLT CellViT-256 FullFT, fold B, seed 42 | 1473866 | A100-SXM4-40GB | 1 | 06:06:40 | 6.111 | COMPLETED | yes |
| KLT CellViT-256 LP, fold A, seed 42 | 1475939 | A100-SXM4-40GB | 1 | 05:18:42 | 5.312 | COMPLETED | yes |
| KLT CellViT-256 LP, fold B, seed 42 | 1475940 | A100-SXM4-40GB | 1 | 04:50:32 | 4.842 | COMPLETED | yes |
| KLT CellViT-256 PEFT, fold A, seed 42 | 1478320 | A100-SXM4-40GB | 1 | 07:37:52 | 7.631 | COMPLETED | yes |
| KLT CellViT-256 PEFT, fold B, seed 42 | 1478322 | A100-SXM4-40GB | 1 | 06:01:49 | 6.030 | COMPLETED | yes |
| KLT CellViT-256 PEFT, fold A, seed 43 | 1524192 | A100-SXM4-40GB | 1 | 06:25:57 | 6.433 | FAILED* | yes |
| KLT CellViT-256 PEFT, fold B, seed 43 | 1524193 | A100-SXM4-40GB | 1 | 05:51:28 | 5.858 | FAILED* | yes |
| KLT CellViT-256 FullFT, fold A, seed 43 | 1524194 | A100-SXM4-40GB | 1 | 06:51:23 | 6.856 | COMPLETED | yes |
| KLT CellViT-256 FullFT, fold B, seed 43 | 1524195 | A100-SXM4-40GB | 1 | 06:09:21 | 6.156 | COMPLETED | yes |
| KLT CellViT-256 PEFT, fold A, seed 44 | 1558251 | A100-SXM4-40GB | 1 | 06:49:09 | 6.819 | COMPLETED | yes |
| KLT CellViT-256 PEFT, fold B, seed 44 | 1558640 | A100-SXM4-40GB | 1 | 07:12:06 | 7.202 | COMPLETED | yes |
| KLT CellViT-256 FullFT, fold A, seed 44 | 1558641 | A100-SXM4-40GB | 1 | 08:23:28 | 8.391 | COMPLETED | yes |
| KLT CellViT-256 FullFT, fold B, seed 44 | 1559201 | A100-SXM4-40GB | 1 | 06:14:33 | 6.242 | COMPLETED | yes |
| KLT CellViT-256 LP, fold A, seed 43 | 1570495 | A100-SXM4-40GB | 1 | 05:27:24 | 5.457 | COMPLETED | yes |
| KLT CellViT-256 LP, fold B, seed 43 | 1570496 | A100-SXM4-40GB | 1 | 05:31:56 | 5.532 | COMPLETED | yes |
| KLT CellViT-256 LP, fold A, seed 44 | 1570497 | A100-SXM4-40GB | 1 | 06:27:53 | 6.465 | COMPLETED | yes |
| KLT CellViT-256 LP, fold B, seed 44 | 1570498 | A100-SXM4-40GB | 1 | 05:22:33 | 5.376 | COMPLETED | yes |

### Nine-tissue reciprocal SAM-H Selected-PEFT

| Experiment/config identifier | Job ID | GPU | GPUs | Elapsed | GPU-hours | Slurm status | Paper result? |
|---|---:|---|---:|---:|---:|---|:---:|
| Tonsil SAM-H PEFT, fold B, seed 42 | 1478428 | A100-SXM4-40GB | 1 | 07:45:56 | 7.766 | COMPLETED | yes |
| Tonsil SAM-H PEFT, fold A, seed 42 | 1478431 | A100-SXM4-40GB | 1 | 09:09:14 | 9.154 | COMPLETED | yes |
| Kidney SAM-H PEFT, fold A, seed 42 | 1480494 | A100-SXM4-40GB | 1 | 01:52:04 | 1.868 | FAILED* | yes |
| Liver SAM-H PEFT, fold A, seed 42 | 1480495 | A100-SXM4-40GB | 1 | 05:33:12 | 5.553 | COMPLETED | yes |
| Breast SAM-H PEFT, fold A, seed 42 | 1482333 | A100-SXM4-40GB | 1 | 12:19:43 | 12.329 | COMPLETED | yes |
| Pancreatic SAM-H PEFT, fold A, seed 42 | 1482334 | A100-SXM4-40GB | 1 | 06:26:33 | 6.442 | COMPLETED | yes |
| Ovary SAM-H PEFT, fold A, seed 42 | 1482335 | A100-SXM4-40GB | 1 | 03:19:50 | 3.331 | COMPLETED | yes |
| Colon SAM-H PEFT, fold A, seed 42 | 1482336 | A100-SXM4-40GB | 1 | 05:35:27 | 5.591 | COMPLETED | yes |
| Lung SAM-H PEFT, fold A, seed 42 | 1482337 | A100-SXM4-40GB | 1 | 03:22:48 | 3.380 | COMPLETED | yes |
| Skin SAM-H PEFT, fold A, seed 42 | 1482338 | A100-SXM4-40GB | 1 | 03:06:24 | 3.107 | COMPLETED | yes |
| Breast SAM-H PEFT, fold B, seed 42 | 1492582 | A100-SXM4-40GB | 1 | 12:46:57 | 12.783 | COMPLETED | yes |
| Pancreatic SAM-H PEFT, fold B, seed 42 | 1492583 | A100-SXM4-40GB | 1 | 03:59:33 | 3.993 | COMPLETED | yes |
| Skin SAM-H PEFT, fold B, seed 42 | 1492587 | A100-SXM4-40GB | 1 | 03:26:07 | 3.435 | COMPLETED | yes |
| Ovary SAM-H PEFT, fold B, seed 42 | 1501898 | A100-SXM4-40GB | 1 | 07:03:22 | 7.056 | COMPLETED | yes |
| Colon SAM-H PEFT, fold B, seed 42 | 1501899 | A100-SXM4-40GB | 1 | 04:22:50 | 4.381 | COMPLETED | yes |
| Lung SAM-H PEFT, fold B, seed 42 | 1501917 | A100-SXM4-40GB | 1 | 05:26:55 | 5.449 | COMPLETED | yes |
| Kidney SAM-H PEFT, fold B, seed 42 | 1524196 | A100-SXM4-40GB | 1 | 01:21:10 | 1.353 | FAILED* | yes |
| Liver SAM-H PEFT, fold B, seed 42 | 1524197 | A100-SXM4-40GB | 1 | 02:56:14 | 2.937 | FAILED* | yes |

### Nine-tissue reciprocal CellViT-256 Selected-PEFT

| Experiment/config identifier | Job ID | GPU | GPUs | Elapsed | GPU-hours | Slurm status | Paper result? |
|---|---:|---|---:|---:|---:|---|:---:|
| Kidney CellViT-256 PEFT, fold B, seed 42 | 1558644 | A100-SXM4-40GB | 1 | 01:01:45 | 1.029 | COMPLETED | yes |
| Liver CellViT-256 PEFT, fold A, seed 42 | 1558645 | A100-SXM4-40GB | 1 | 04:11:50 | 4.197 | COMPLETED | yes |
| Liver CellViT-256 PEFT, fold B, seed 42 | 1558646 | A100-SXM4-40GB | 1 | 02:26:44 | 2.446 | COMPLETED | yes |
| Tonsil CellViT-256 PEFT, fold A, seed 42 | 1558698 | A100-SXM4-40GB | 1 | 08:12:01 | 8.200 | COMPLETED | yes |
| Tonsil CellViT-256 PEFT, fold B, seed 42 | 1558796 | A100-SXM4-40GB | 1 | 06:32:26 | 6.541 | COMPLETED | yes |
| Kidney CellViT-256 PEFT, fold A, seed 42 | 1559486 | A100-SXM4-40GB | 1 | 01:35:44 | 1.596 | COMPLETED | yes |
| Lung CellViT-256 PEFT, fold B, seed 42 | 1570250 | A100-SXM4-40GB | 1 | 05:28:02 | 5.467 | COMPLETED | yes |
| Ovary CellViT-256 PEFT, fold A, seed 42 | 1570251 | A100-SXM4-40GB | 1 | 03:43:08 | 3.719 | COMPLETED | yes |
| Pancreatic CellViT-256 PEFT, fold B, seed 42 | 1570254 | A100-SXM4-40GB | 1 | 03:08:21 | 3.139 | COMPLETED | yes |
| Breast CellViT-256 PEFT, fold A, seed 42 | 1570484 | A100-SXM4-40GB | 1 | 09:02:14 | 9.037 | COMPLETED | yes |
| Breast CellViT-256 PEFT, fold B, seed 42 | 1570485 | A100-SXM4-40GB | 1 | 09:45:23 | 9.756 | COMPLETED | yes |
| Colon CellViT-256 PEFT, fold A, seed 42 | 1570486 | A100-SXM4-40GB | 1 | 04:33:38 | 4.561 | COMPLETED | yes |
| Colon CellViT-256 PEFT, fold B, seed 42 | 1570487 | A100-SXM4-40GB | 1 | 02:59:11 | 2.986 | COMPLETED | yes |
| Lung CellViT-256 PEFT, fold A, seed 42 | 1570488 | A100-SXM4-40GB | 1 | 02:59:35 | 2.993 | COMPLETED | yes |
| Ovary CellViT-256 PEFT, fold B, seed 42 | 1570489 | A100-SXM4-40GB | 1 | 05:19:53 | 5.331 | COMPLETED | yes |
| Pancreatic CellViT-256 PEFT, fold A, seed 42 | 1570490 | A100-SXM4-40GB | 1 | 05:51:14 | 5.854 | COMPLETED | yes |
| Skin CellViT-256 PEFT, fold A, seed 42 | 1570491 | A100-SXM4-40GB | 1 | 02:28:53 | 2.481 | COMPLETED | yes |
| Skin CellViT-256 PEFT, fold B, seed 42 | 1570492 | A100-SXM4-40GB | 1 | 02:40:33 | 2.676 | COMPLETED | yes |

### Nine-tissue reciprocal SAM-H FullFT

| Experiment/config identifier | Job ID | GPU | GPUs | Elapsed | GPU-hours | Slurm status | Paper result? |
|---|---:|---|---:|---:|---:|---|:---:|
| Kidney SAM-H FullFT, fold A, seed 42 | 1586683 | A100-SXM4-40GB | 1 | 02:16:25 | 2.274 | COMPLETED | yes |
| Kidney SAM-H FullFT, fold B, seed 42 | 1586684 | A100-SXM4-40GB | 1 | 01:32:50 | 1.547 | COMPLETED | yes |
| Liver SAM-H FullFT, fold A, seed 42 | 1586685 | A100-SXM4-40GB | 1 | 06:01:34 | 6.026 | COMPLETED | yes |
| Liver SAM-H FullFT, fold B, seed 42 | 1586686 | A100-SXM4-40GB | 1 | 03:06:50 | 3.114 | COMPLETED | yes |
| Tonsil SAM-H FullFT, fold A, seed 42 | 1586687 | A100-SXM4-40GB | 1 | 10:14:25 | 10.240 | COMPLETED | yes |
| Tonsil SAM-H FullFT, fold B, seed 42 | 1586688 | A100-SXM4-40GB | 1 | 08:25:54 | 8.432 | COMPLETED | yes |
| Breast SAM-H FullFT, fold A, seed 42 | 1596567_0 | A100-SXM4-40GB | 1 | 13:24:09 | 13.402 | COMPLETED | yes |
| Breast SAM-H FullFT, fold B, seed 42 | 1596567_1 | A100-SXM4-40GB | 1 | 13:58:54 | 13.982 | COMPLETED | yes |
| Colon SAM-H FullFT, fold A, seed 42 | 1596567_2 | A100-SXM4-40GB | 1 | 06:14:54 | 6.248 | COMPLETED | yes |
| Colon SAM-H FullFT, fold B, seed 42 | 1596567_3 | A100-SXM4-40GB | 1 | 04:30:23 | 4.506 | COMPLETED | yes |
| Lung SAM-H FullFT, fold A, seed 42 | 1596567_4 | A100-SXM4-40GB | 1 | 03:39:14 | 3.654 | COMPLETED | yes |
| Lung SAM-H FullFT, fold B, seed 42 | 1596567_5 | A100-SXM4-40GB | 1 | 06:12:10 | 6.203 | COMPLETED | yes |
| Ovary SAM-H FullFT, fold A, seed 42 | 1596567_6 | A100-SXM4-40GB | 1 | 03:52:38 | 3.877 | COMPLETED | yes |
| Ovary SAM-H FullFT, fold B, seed 42 | 1596567_7 | A100-SXM4-40GB | 1 | 07:43:12 | 7.720 | COMPLETED | yes |
| Pancreatic SAM-H FullFT, fold A, seed 42 | 1596567_8 | A100-SXM4-40GB | 1 | 07:17:14 | 7.287 | COMPLETED | yes |
| Pancreatic SAM-H FullFT, fold B, seed 42 | 1596567_9 | A100-SXM4-40GB | 1 | 04:19:27 | 4.324 | COMPLETED | yes |
| Skin SAM-H FullFT, fold A, seed 42 | 1596567_10 | A100-SXM4-40GB | 1 | 03:40:16 | 3.671 | COMPLETED | yes |
| Skin SAM-H FullFT, fold B, seed 42 | 1596567_11 | A100-SXM4-40GB | 1 | 03:56:15 | 3.938 | COMPLETED | yes |

### Tissue-specific CellViT-256 FullFT controls

| Experiment/config identifier | Job ID | GPU | GPUs | Elapsed | GPU-hours | Slurm status | Paper result? |
|---|---:|---|---:|---:|---:|---|:---:|
| Kidney CellViT-256 FullFT, fold A, seed 42 | 1586689 | A100-SXM4-40GB | 1 | 01:30:15 | 1.504 | COMPLETED | yes |
| Kidney CellViT-256 FullFT, fold B, seed 42 | 1586690 | A100-SXM4-40GB | 1 | 01:05:02 | 1.084 | COMPLETED | yes |
| Liver CellViT-256 FullFT, fold A, seed 42 | 1586691 | A100-SXM4-40GB | 1 | 04:07:17 | 4.121 | COMPLETED | yes |
| Liver CellViT-256 FullFT, fold B, seed 42 | 1586692 | A100-SXM4-40GB | 1 | 02:12:12 | 2.203 | COMPLETED | yes |
| Tonsil CellViT-256 FullFT, fold A, seed 42 | 1586693 | A100-SXM4-40GB | 1 | 06:49:47 | 6.830 | COMPLETED | yes |
| Tonsil CellViT-256 FullFT, fold B, seed 42 | 1586694 | A100-SXM4-40GB | 1 | 06:41:20 | 6.689 | COMPLETED | yes |

## Appendix B: deduplicated non-paper GPU-allocation ledger

This is the explicit inventory behind C; **every row has `Paper result? = no`**. Classification is conservative: a terminal allocation is non-paper unless its job ID is in Appendix A or the standalone evaluation table. “A100 path” and “generic/V100 path” are scheduler-path labels; exact device strings were unavailable for many preliminary jobs. Job 1614903 is frozen at the audit cutoff rather than at this report-writing pass.

| Job ID | Slurm job name | Classification | GPU path | GPUs | State | Elapsed | GPU-hours |
|---:|---|---|---|---:|---|---:|---:|
| 1032445 | train_sthelar | failed/incomplete preliminary | generic/V100 path | 1 | FAILED | 00:00:04 | 0.001 |
| 1032491 | train_sthelar | failed/incomplete preliminary | generic/V100 path | 1 | FAILED | 00:00:02 | 0.001 |
| 1032509 | train_sthelar | failed/incomplete preliminary | generic/V100 path | 1 | FAILED | 00:00:03 | 0.001 |
| 1032517 | train_sthelar | failed/incomplete preliminary | generic/V100 path | 1 | FAILED | 00:00:03 | 0.001 |
| 1032524 | train_sthelar | exploratory/superseded training | generic/V100 path | 1 | COMPLETED | 00:43:14 | 0.721 |
| 1033422 | train_sthelar | exploratory/superseded training | generic/V100 path | 1 | COMPLETED | 00:44:30 | 0.742 |
| 1033423 | train_sthelar | exploratory/superseded training | generic/V100 path | 1 | COMPLETED | 00:49:40 | 0.828 |
| 1033424 | train_sthelar | failed/incomplete preliminary | generic/V100 path | 1 | FAILED | 00:07:07 | 0.119 |
| 1033654 | train_sthelar | exploratory/superseded training | generic/V100 path | 1 | COMPLETED | 00:18:27 | 0.307 |
| 1040826 | train_sthelar | exploratory/superseded training | generic/V100 path | 1 | COMPLETED | 00:45:14 | 0.754 |
| 1040827 | train_sthelar | exploratory/superseded training | generic/V100 path | 1 | COMPLETED | 00:44:06 | 0.735 |
| 1040828 | train_sthelar | exploratory/superseded training | generic/V100 path | 1 | COMPLETED | 00:45:41 | 0.761 |
| 1040829 | train_sthelar | exploratory/superseded training | generic/V100 path | 1 | COMPLETED | 00:37:12 | 0.620 |
| 1040830 | train_sthelar | exploratory/superseded training | generic/V100 path | 1 | COMPLETED | 00:40:28 | 0.674 |
| 1040831 | train_sthelar | exploratory/superseded training | generic/V100 path | 1 | COMPLETED | 00:40:22 | 0.673 |
| 1044992 | train_sthelar | exploratory/superseded training | generic/V100 path | 1 | COMPLETED | 00:50:41 | 0.845 |
| 1044993 | train_sthelar | exploratory/superseded training | generic/V100 path | 1 | COMPLETED | 00:51:09 | 0.853 |
| 1044994 | train_sthelar | exploratory/superseded training | generic/V100 path | 1 | COMPLETED | 00:49:40 | 0.828 |
| 1063746 | train_sthelar | failed/incomplete preliminary | generic/V100 path | 1 | FAILED | 00:05:32 | 0.092 |
| 1065625 | train_sthelar | exploratory/superseded training | generic/V100 path | 1 | COMPLETED | 00:52:18 | 0.872 |
| 1065626 | train_sthelar | exploratory/superseded training | generic/V100 path | 1 | COMPLETED | 00:18:29 | 0.308 |
| 1065627 | train_sthelar | exploratory/superseded training | generic/V100 path | 1 | COMPLETED | 00:38:35 | 0.643 |
| 1065628 | train_sthelar | exploratory/superseded training | generic/V100 path | 1 | COMPLETED | 00:47:23 | 0.790 |
| 1065629 | train_sthelar | exploratory/superseded training | generic/V100 path | 1 | COMPLETED | 00:47:23 | 0.790 |
| 1065630 | train_sthelar | exploratory/superseded training | generic/V100 path | 1 | COMPLETED | 00:47:24 | 0.790 |
| 1065631 | train_sthelar | exploratory/superseded training | generic/V100 path | 1 | COMPLETED | 00:44:11 | 0.736 |
| 1065632 | train_sthelar | exploratory/superseded training | generic/V100 path | 1 | COMPLETED | 00:46:09 | 0.769 |
| 1065633 | train_sthelar | exploratory/superseded training | generic/V100 path | 1 | COMPLETED | 00:45:22 | 0.756 |
| 1065634 | train_sthelar | exploratory/superseded training | generic/V100 path | 1 | COMPLETED | 00:45:36 | 0.760 |
| 1068318 | prep_sthelar | preprocessing | generic/V100 path | 1 | FAILED | 00:00:04 | 0.001 |
| 1068331 | prep_sthelar | preprocessing | generic/V100 path | 1 | FAILED | 00:00:07 | 0.002 |
| 1068389 | prep_sthelar | preprocessing | generic/V100 path | 1 | COMPLETED | 01:40:28 | 1.674 |
| 1068423 | train_sthelar | failed/incomplete preliminary | generic/V100 path | 1 | CANCELLED | 00:04:53 | 0.081 |
| 1069067 | train_sthelar | exploratory/superseded training | generic/V100 path | 1 | COMPLETED | 09:43:40 | 9.728 |
| 1069068 | train_sthelar | exploratory/superseded training | generic/V100 path | 1 | COMPLETED | 09:53:55 | 9.899 |
| 1069069 | train_sthelar | exploratory/superseded training | generic/V100 path | 1 | COMPLETED | 09:05:38 | 9.094 |
| 1080763 | train_sthelar | exploratory/superseded training | generic/V100 path | 1 | COMPLETED | 15:39:33 | 15.659 |
| 1080780 | train_sthelar | failed/incomplete preliminary | generic/V100 path | 1 | CANCELLED | 02:36:49 | 2.614 |
| 1082172 | train_sthelar | failed/incomplete preliminary | generic/V100 path | 1 | FAILED | 21:05:32 | 21.092 |
| 1088175 | train_sthelar | failed/incomplete preliminary | generic/V100 path | 1 | FAILED | 00:00:05 | 0.001 |
| 1088183 | train_sthelar | failed/incomplete preliminary | generic/V100 path | 1 | FAILED | 00:00:03 | 0.001 |
| 1107655 | train_sthelar | failed/incomplete preliminary | generic/V100 path | 1 | TIMEOUT | 24:00:19 | 24.005 |
| 1107656 | train_sthelar | failed/incomplete preliminary | generic/V100 path | 1 | FAILED | 00:02:49 | 0.047 |
| 1107657 | train_sthelar | failed/incomplete preliminary | generic/V100 path | 1 | TIMEOUT | 24:00:19 | 24.005 |
| 1113889 | train_sthelar | exploratory/superseded training | A100 path | 1 | COMPLETED | 09:51:11 | 9.853 |
| 1113899 | train_sthelar | exploratory/superseded training | A100 path | 1 | COMPLETED | 10:46:35 | 10.776 |
| 1113900 | train_sthelar | exploratory/superseded training | A100 path | 1 | COMPLETED | 10:24:36 | 10.410 |
| 1119921 | prep_sthelar | preprocessing | generic/V100 path | 1 | COMPLETED | 01:38:27 | 1.641 |
| 1119928 | prep_sthelar | preprocessing | generic/V100 path | 1 | FAILED | 00:00:03 | 0.001 |
| 1119940 | train_sthelar | failed/incomplete preliminary | A100 path | 1 | FAILED | 00:01:22 | 0.023 |
| 1119941 | train_sthelar | failed/incomplete preliminary | A100 path | 1 | FAILED | 00:00:01 | 0.000 |
| 1119942 | train_sthelar | failed/incomplete preliminary | A100 path | 1 | FAILED | 00:00:01 | 0.000 |
| 1119945 | train_sthelar | exploratory/superseded training | A100 path | 1 | COMPLETED | 09:35:29 | 9.591 |
| 1125683 | train_sthelar | exploratory/superseded training | A100 path | 1 | COMPLETED | 06:03:47 | 6.063 |
| 1125684 | train_sthelar | exploratory/superseded training | A100 path | 1 | COMPLETED | 06:20:15 | 6.338 |
| 1125685 | train_sthelar | failed/incomplete preliminary | A100 path | 1 | FAILED | 00:00:25 | 0.007 |
| 1125696 | train_sthelar | exploratory/superseded training | A100 path | 1 | COMPLETED | 05:49:57 | 5.832 |
| 1127503 | prep_sthelar | preprocessing | generic/V100 path | 1 | COMPLETED | 00:53:18 | 0.888 |
| 1127504 | train_sthelar | exploratory/superseded training | A100 path | 1 | COMPLETED | 07:16:11 | 7.270 |
| 1127505 | train_sthelar | exploratory/superseded training | A100 path | 1 | COMPLETED | 07:29:57 | 7.499 |
| 1127507 | train_sthelar | failed/incomplete preliminary | A100 path | 1 | FAILED | 01:48:32 | 1.809 |
| 1127508 | train_sthelar | failed/incomplete preliminary | A100 path | 1 | FAILED | 01:48:23 | 1.806 |
| 1127509 | train_sthelar | exploratory/superseded training | A100 path | 1 | COMPLETED | 01:54:22 | 1.906 |
| 1128690 | train_sthelar | exploratory/superseded training | A100 path | 1 | COMPLETED | 09:45:07 | 9.752 |
| 1130686 | prep_sthelar | preprocessing | generic/V100 path | 1 | COMPLETED | 01:28:39 | 1.478 |
| 1130687 | train_sthelar | exploratory/superseded training | A100 path | 1 | COMPLETED | 05:03:19 | 5.055 |
| 1130703 | train_sthelar | exploratory/superseded training | A100 path | 1 | COMPLETED | 01:58:17 | 1.971 |
| 1130742 | train_sthelar | exploratory/superseded training | A100 path | 1 | COMPLETED | 08:22:50 | 8.381 |
| 1130743 | train_sthelar | failed/incomplete preliminary | A100 path | 1 | FAILED | 08:22:04 | 8.368 |
| 1130755 | train_sthelar | failed/incomplete preliminary | A100 path | 1 | FAILED | 06:27:34 | 6.459 |
| 1131917 | infer_sth | exploratory evaluation | A100 path | 1 | FAILED | 00:00:02 | 0.001 |
| 1131918 | infer_sth | exploratory evaluation | A100 path | 1 | FAILED | 00:00:02 | 0.001 |
| 1131924 | infer_sth | exploratory evaluation | A100 path | 1 | FAILED | 00:00:18 | 0.005 |
| 1131925 | infer_sth | exploratory evaluation | A100 path | 1 | FAILED | 00:00:13 | 0.004 |
| 1131930 | infer_sthelar | exploratory evaluation | A100 path | 1 | FAILED | 00:03:19 | 0.055 |
| 1131931 | infer_sthelar | exploratory evaluation | A100 path | 1 | FAILED | 00:03:14 | 0.054 |
| 1131945 | infer_sthelar | exploratory evaluation | A100 path | 1 | FAILED | 00:03:54 | 0.065 |
| 1131946 | infer_sthelar | exploratory evaluation | A100 path | 1 | FAILED | 00:03:50 | 0.064 |
| 1131981 | infer_sthelar | exploratory evaluation | A100 path | 1 | COMPLETED | 00:39:51 | 0.664 |
| 1131982 | infer_sthelar | exploratory evaluation | A100 path | 1 | COMPLETED | 00:06:17 | 0.105 |
| 1132029 | train_sthelar | failed/incomplete preliminary | A100 path | 1 | CANCELLED | 00:00:33 | 0.009 |
| 1132031 | train_sthelar | failed/incomplete preliminary | A100 path | 1 | CANCELLED | 00:00:10 | 0.003 |
| 1132034 | train_sthelar | failed/incomplete preliminary | A100 path | 1 | CANCELLED | 00:00:10 | 0.003 |
| 1132035 | train_sthelar | exploratory/superseded training | A100 path | 1 | COMPLETED | 09:48:18 | 9.805 |
| 1132123 | train_sthelar | exploratory/superseded training | A100 path | 1 | COMPLETED | 07:24:35 | 7.410 |
| 1132648 | train_sthelar | exploratory/superseded training | A100 path | 1 | COMPLETED | 09:24:54 | 9.415 |
| 1132649 | train_sthelar | exploratory/superseded training | A100 path | 1 | COMPLETED | 06:53:18 | 6.888 |
| 1135395 | train_sthelar | exploratory/superseded training | A100 path | 1 | COMPLETED | 10:28:51 | 10.481 |
| 1135397 | train_sthelar | exploratory/superseded training | A100 path | 1 | COMPLETED | 07:06:08 | 7.102 |
| 1136164 | train_sthelar | failed/incomplete preliminary | A100 path | 1 | CANCELLED | 00:00:48 | 0.013 |
| 1136166 | train_sthelar | exploratory/superseded training | A100 path | 1 | COMPLETED | 07:36:10 | 7.603 |
| 1136167 | train_sthelar | failed/incomplete preliminary | A100 path | 1 | FAILED | 06:02:05 | 6.035 |
| 1138770 | train_sthelar | failed/incomplete preliminary | A100 path | 1 | FAILED | 00:04:23 | 0.073 |
| 1138772 | train_sthelar | failed/incomplete preliminary | A100 path | 1 | FAILED | 00:04:17 | 0.071 |
| 1138773 | train_sthelar | failed/incomplete preliminary | A100 path | 1 | FAILED | 00:04:14 | 0.071 |
| 1138774 | train_sthelar | failed/incomplete preliminary | A100 path | 1 | FAILED | 00:04:11 | 0.070 |
| 1138775 | train_sthelar | exploratory/superseded training | A100 path | 1 | COMPLETED | 07:25:31 | 7.425 |
| 1139845 | train_sthelar | exploratory/superseded training | A100 path | 1 | COMPLETED | 07:16:29 | 7.275 |
| 1139846 | train_sthelar | exploratory/superseded training | A100 path | 1 | COMPLETED | 07:33:08 | 7.552 |
| 1139851 | train_sthelar | exploratory/superseded training | A100 path | 1 | COMPLETED | 07:11:58 | 7.199 |
| 1139852 | train_sthelar | exploratory/superseded training | A100 path | 1 | COMPLETED | 07:26:33 | 7.442 |
| 1139863 | train_sthelar | exploratory/superseded training | A100 path | 1 | COMPLETED | 10:13:55 | 10.232 |
| 1140484 | train_sthelar | exploratory/superseded training | A100 path | 1 | COMPLETED | 08:56:57 | 8.949 |
| 1140485 | train_sthelar | exploratory/superseded training | A100 path | 1 | COMPLETED | 09:45:12 | 9.753 |
| 1140486 | train_sthelar | exploratory/superseded training | A100 path | 1 | COMPLETED | 07:52:17 | 7.871 |
| 1140487 | train_sthelar | exploratory/superseded training | A100 path | 1 | COMPLETED | 10:46:00 | 10.767 |
| 1141405 | train_sthelar | failed/incomplete preliminary | A100 path | 1 | CANCELLED | 00:00:43 | 0.012 |
| 1141406 | train_sthelar | failed/incomplete preliminary | A100 path | 1 | CANCELLED | 00:00:43 | 0.012 |
| 1141407 | train_sthelar | failed/incomplete preliminary | A100 path | 1 | CANCELLED | 00:00:43 | 0.012 |
| 1141410 | train_sthelar | exploratory/superseded training | A100 path | 1 | COMPLETED | 01:20:14 | 1.337 |
| 1141411 | train_sthelar | exploratory/superseded training | A100 path | 1 | COMPLETED | 01:17:41 | 1.295 |
| 1141412 | train_sthelar | exploratory/superseded training | A100 path | 1 | COMPLETED | 01:16:29 | 1.275 |
| 1141413 | train_sthelar | exploratory/superseded training | A100 path | 1 | COMPLETED | 01:22:45 | 1.379 |
| 1141414 | train_sthelar | exploratory/superseded training | A100 path | 1 | COMPLETED | 01:26:29 | 1.441 |
| 1142897 | train_sthelar | exploratory/superseded training | A100 path | 1 | COMPLETED | 01:07:35 | 1.126 |
| 1142898 | train_sthelar | exploratory/superseded training | A100 path | 1 | COMPLETED | 01:18:17 | 1.305 |
| 1142899 | train_sthelar | exploratory/superseded training | A100 path | 1 | COMPLETED | 01:24:38 | 1.411 |
| 1142900 | train_sthelar | exploratory/superseded training | A100 path | 1 | COMPLETED | 01:21:51 | 1.364 |
| 1145669 | train_sthelar | exploratory/superseded training | A100 path | 1 | COMPLETED | 10:17:16 | 10.288 |
| 1145670 | train_sthelar | exploratory/superseded training | A100 path | 1 | COMPLETED | 10:25:04 | 10.418 |
| 1145671 | train_sthelar | exploratory/superseded training | A100 path | 1 | COMPLETED | 07:12:22 | 7.206 |
| 1145672 | train_sthelar | exploratory/superseded training | A100 path | 1 | COMPLETED | 07:42:59 | 7.716 |
| 1151107 | train_sthelar | exploratory/superseded training | A100 path | 1 | COMPLETED | 07:59:23 | 7.990 |
| 1151111 | prep_sthelar | preprocessing | generic/V100 path | 1 | COMPLETED | 01:53:59 | 1.900 |
| 1151140 | train_sthelar | exploratory/superseded training | A100 path | 1 | COMPLETED | 02:12:29 | 2.208 |
| 1153890 | fisher_liver_fullft | exploratory evaluation | A100 path | 1 | COMPLETED | 00:04:18 | 0.072 |
| 1169680 | train_sthelar | failed/incomplete preliminary | A100 path | 1 | FAILED | 11:14:35 | 11.243 |
| 1169681 | train_sthelar | failed/incomplete preliminary | A100 path | 1 | FAILED | 04:02:59 | 4.050 |
| 1169682 | train_sthelar | failed/incomplete preliminary | A100 path | 1 | FAILED | 03:51:43 | 3.862 |
| 1172318 | train_sthelar | exploratory/superseded training | A100 path | 1 | COMPLETED | 07:22:10 | 7.369 |
| 1172319 | train_sthelar | exploratory/superseded training | A100 path | 1 | COMPLETED | 03:04:27 | 3.074 |
| 1172320 | train_sthelar | exploratory/superseded training | A100 path | 1 | COMPLETED | 09:33:10 | 9.553 |
| 1173367 | train_sthelar | exploratory/superseded training | A100 path | 1 | COMPLETED | 06:59:31 | 6.992 |
| 1173368 | train_sthelar | exploratory/superseded training | A100 path | 1 | COMPLETED | 02:47:28 | 2.791 |
| 1173369 | train_sthelar | exploratory/superseded training | A100 path | 1 | COMPLETED | 09:00:48 | 9.013 |
| 1186380 | train_sthelar | exploratory/superseded training | A100 path | 1 | COMPLETED | 23:52:34 | 23.876 |
| 1186381 | train_sthelar | exploratory/superseded training | A100 path | 1 | COMPLETED | 21:09:03 | 21.151 |
| 1186382 | train_sthelar | exploratory/superseded training | A100 path | 1 | COMPLETED | 08:50:18 | 8.838 |
| 1186383 | train_sthelar | exploratory/superseded training | A100 path | 1 | COMPLETED | 08:09:29 | 8.158 |
| 1186384 | train_sthelar | exploratory/superseded training | A100 path | 1 | COMPLETED | 08:22:19 | 8.372 |
| 1186385 | train_sthelar | exploratory/superseded training | A100 path | 1 | COMPLETED | 07:13:36 | 7.227 |
| 1187867 | sthelar_g3_colon_peft_v100 | exploratory/superseded training | generic/V100 path | 1 | COMPLETED | 14:31:52 | 14.531 |
| 1187868 | sthelar_g3_lung_peft_v100 | exploratory/superseded training | generic/V100 path | 1 | COMPLETED | 12:04:36 | 12.077 |
| 1187869 | sthelar_g3_skin_peft_v100 | exploratory/superseded training | generic/V100 path | 1 | COMPLETED | 08:24:51 | 8.414 |
| 1187880 | sthelar_liver_peft_v100 | exploratory/superseded training | generic/V100 path | 1 | COMPLETED | 11:00:27 | 11.008 |
| 1193119 | peft43_liver_v100 | exploratory/superseded training | generic/V100 path | 1 | COMPLETED | 10:54:04 | 10.901 |
| 1193120 | peft43_kidney_v100 | exploratory/superseded training | generic/V100 path | 1 | COMPLETED | 04:40:17 | 4.671 |
| 1193121 | peft43_tonsil_v100 | exploratory/superseded training | generic/V100 path | 1 | COMPLETED | 23:36:14 | 23.604 |
| 1193122 | peft43_breast_v100 | failed/incomplete preliminary | generic/V100 path | 1 | TIMEOUT | 24:00:10 | 24.003 |
| 1193123 | peft43_colon_v100 | exploratory/superseded training | generic/V100 path | 1 | COMPLETED | 14:22:30 | 14.375 |
| 1193124 | peft43_lung_v100 | exploratory/superseded training | generic/V100 path | 1 | COMPLETED | 12:47:42 | 12.795 |
| 1193125 | peft43_ovary_v100 | exploratory/superseded training | generic/V100 path | 1 | COMPLETED | 15:38:19 | 15.639 |
| 1193126 | peft43_pancreatic_v100 | exploratory/superseded training | generic/V100 path | 1 | COMPLETED | 15:15:14 | 15.254 |
| 1193127 | peft43_skin_v100 | exploratory/superseded training | generic/V100 path | 1 | COMPLETED | 09:06:18 | 9.105 |
| 1208016 | fullft_tonsil_v100 | failed/incomplete preliminary | generic/V100 path | 1 | CANCELLED | 16:19:50 | 16.331 |
| 1208017 | fullft_pancreatic_v100 | exploratory/superseded training | generic/V100 path | 1 | COMPLETED | 16:02:39 | 16.044 |
| 1208018 | fullft_skin_v100 | exploratory/superseded training | generic/V100 path | 1 | COMPLETED | 10:18:12 | 10.303 |
| 1208019 | vera_af_heads43_v100 | exploratory/superseded training | generic/V100 path | 1 | COMPLETED | 12:22:53 | 12.381 |
| 1209326 | resume_tonsil_fullft_v100 | exploratory/superseded training | generic/V100 path | 1 | COMPLETED | 10:59:05 | 10.985 |
| 1212868 | train_sthelar | exploratory/superseded training | A100 path | 1 | COMPLETED | 12:00:02 | 12.001 |
| 1213033 | train_sthelar | exploratory/superseded training | generic/V100 path | 1 | COMPLETED | 18:25:23 | 18.423 |
| 1213034 | train_sthelar | exploratory/superseded training | generic/V100 path | 1 | COMPLETED | 21:03:02 | 21.051 |
| 1214255 | train_sthelar | exploratory/superseded training | generic/V100 path | 1 | COMPLETED | 22:31:58 | 22.533 |
| 1227082 | train_sthelar | failed/incomplete preliminary | A100 path | 1 | CANCELLED | 00:00:22 | 0.006 |
| 1227952 | train_sthelar | exploratory/superseded training | generic/V100 path | 1 | COMPLETED | 04:47:15 | 4.787 |
| 1465775 | train_sthelar | failed/incomplete preliminary | A100 path | 1 | FAILED | 00:03:09 | 0.052 |
| 1465778 | eval_klt_frozen | exploratory evaluation | A100 path | 1 | FAILED | 00:01:53 | 0.031 |
| 1465779 | train_sthelar | failed/incomplete preliminary | A100 path | 1 | FAILED | 00:03:02 | 0.051 |
| 1465780 | train_sthelar | failed/incomplete preliminary | A100 path | 1 | FAILED | 00:02:52 | 0.048 |
| 1465781 | train_sthelar | failed/incomplete preliminary | A100 path | 1 | FAILED | 00:02:59 | 0.050 |
| 1465821 | kltB_peft_s42_retry | failed/incomplete preliminary | A100 path | 1 | FAILED | 00:03:30 | 0.058 |
| 1465912 | kltA_ntheader1_s42 | exploratory/superseded training | A100 path | 1 | COMPLETED | 08:17:45 | 8.296 |
| 1465913 | kltB_ntheader1_s42 | failed/incomplete preliminary | A100 path | 1 | FAILED | 00:03:53 | 0.065 |
| 1468273 | kltB_nthead_s42_retry2 | exploratory/superseded training | A100 path | 1 | COMPLETED | 07:59:07 | 7.985 |
| 1468276 | kltA_peft_s43 | failed/incomplete preliminary | A100 path | 1 | FAILED | 00:15:30 | 0.258 |
| 1468277 | kltB_peft_s43 | failed/incomplete preliminary | A100 path | 1 | FAILED | 00:15:31 | 0.259 |
| 1468278 | kltA_nthead_s43 | failed/incomplete preliminary | A100 path | 1 | FAILED | 00:02:55 | 0.049 |
| 1468279 | kltB_nthead_s43 | failed/incomplete preliminary | A100 path | 1 | FAILED | 00:02:55 | 0.049 |
| 1468280 | kltA_fullft_s43 | failed/incomplete preliminary | A100 path | 1 | FAILED | 00:02:43 | 0.045 |
| 1468281 | kltB_fullft_s43 | failed/incomplete preliminary | A100 path | 1 | FAILED | 00:02:43 | 0.045 |
| 1469655 | kltA_peft_s43_retry | failed/incomplete preliminary | A100 path | 1 | FAILED | 06:40:22 | 6.673 |
| 1469656 | kltB_peft_s43_retry | failed/incomplete preliminary | A100 path | 1 | FAILED | 00:05:33 | 0.092 |
| 1469657 | kltA_nthead_s43_retry | failed/incomplete preliminary | A100 path | 1 | FAILED | 00:05:05 | 0.085 |
| 1469658 | kltB_nthead_s43_retry | failed/incomplete preliminary | A100 path | 1 | FAILED | 00:04:59 | 0.083 |
| 1469778 | cv256_A_frozen_s42 | failed/incomplete preliminary | A100 path | 1 | FAILED | 00:03:12 | 0.053 |
| 1469779 | cv256_B_frozen_s42 | failed/incomplete preliminary | A100 path | 1 | FAILED | 00:04:37 | 0.077 |
| 1469780 | cv256_A_lp_s42 | failed/incomplete preliminary | A100 path | 1 | FAILED | 00:06:15 | 0.104 |
| 1469781 | cv256_B_lp_s42 | failed/incomplete preliminary | A100 path | 1 | FAILED | 00:05:01 | 0.084 |
| 1469782 | cv256_A_fullft_s42 | failed/incomplete preliminary | A100 path | 1 | FAILED | 00:04:58 | 0.083 |
| 1469783 | cv256_B_fullft_s42 | failed/incomplete preliminary | A100 path | 1 | FAILED | 00:04:39 | 0.077 |
| 1473855 | kltA_peft_s43_retry3 | failed/incomplete preliminary | A100 path | 1 | NODE_FAIL | 00:16:46 | 0.279 |
| 1473856 | kltB_peft_s43_retry3 | failed/incomplete preliminary | A100 path | 1 | NODE_FAIL | 00:16:46 | 0.279 |
| 1473857 | kltA_nthead_s43_retry3 | failed/incomplete preliminary | A100 path | 1 | NODE_FAIL | 00:16:46 | 0.279 |
| 1473858 | kltB_nthead_s43_retry3 | failed/incomplete preliminary | A100 path | 1 | NODE_FAIL | 00:16:46 | 0.279 |
| 1473861 | cv256_A_frozen_s42_retry2 | failed/incomplete preliminary | A100 path | 1 | NODE_FAIL | 00:15:02 | 0.251 |
| 1473862 | cv256_B_frozen_s42_retry2 | failed/incomplete preliminary | A100 path | 1 | NODE_FAIL | 00:15:02 | 0.251 |
| 1473863 | cv256_A_lp_s42_retry2 | failed/incomplete preliminary | A100 path | 1 | NODE_FAIL | 00:15:02 | 0.251 |
| 1473864 | cv256_B_lp_s42_retry2 | failed/incomplete preliminary | A100 path | 1 | NODE_FAIL | 00:15:02 | 0.251 |
| 1475935 | kltA_nthead_s43_retry4 | exploratory/superseded training | A100 path | 1 | COMPLETED | 07:55:38 | 7.927 |
| 1475936 | kltB_nthead_s43_retry4 | exploratory/superseded training | A100 path | 1 | COMPLETED | 07:10:49 | 7.180 |
| 1478430 | livA_peft_s42 | failed/incomplete preliminary | A100 path | 1 | FAILED | 00:03:15 | 0.054 |
| 1478432 | livB_peft_s42 | failed/incomplete preliminary | A100 path | 1 | FAILED | 00:03:08 | 0.052 |
| 1478433 | kidB_peft_s42 | failed/incomplete preliminary | A100 path | 1 | FAILED | 00:03:15 | 0.054 |
| 1478434 | kidA_peft_s42 | failed/incomplete preliminary | A100 path | 1 | FAILED | 00:02:50 | 0.047 |
| 1492584 | ovaB_peft_s42 | failed/incomplete preliminary | A100 path | 1 | FAILED | 00:03:12 | 0.053 |
| 1492585 | colB_peft_s42 | failed/incomplete preliminary | A100 path | 1 | FAILED | 00:02:44 | 0.046 |
| 1492586 | lunB_peft_s42 | failed/incomplete preliminary | A100 path | 1 | FAILED | 00:03:18 | 0.055 |
| 1501900 | lunB_peft_s42_retry | failed/incomplete preliminary | A100 path | 1 | FAILED | 00:03:20 | 0.056 |
| 1558252 | cv256B_peft_s44 | failed/incomplete preliminary | A100 path | 1 | FAILED | 00:02:41 | 0.045 |
| 1558253 | cv256A_full_s44 | failed/incomplete preliminary | A100 path | 1 | FAILED | 00:02:41 | 0.045 |
| 1558254 | cv256B_full_s44 | failed/incomplete preliminary | A100 path | 1 | FAILED | 00:02:11 | 0.036 |
| 1558255 | c256kidA_p42 | failed/incomplete preliminary | A100 path | 1 | FAILED | 00:02:24 | 0.040 |
| 1558256 | c256kidB_p42 | failed/incomplete preliminary | A100 path | 1 | FAILED | 00:02:24 | 0.040 |
| 1558257 | c256livA_p42 | failed/incomplete preliminary | A100 path | 1 | FAILED | 00:02:24 | 0.040 |
| 1558258 | c256livB_p42 | failed/incomplete preliminary | A100 path | 1 | FAILED | 00:02:02 | 0.034 |
| 1558259 | c256tonA_p42 | failed/incomplete preliminary | A100 path | 1 | FAILED | 00:02:02 | 0.034 |
| 1558260 | c256tonB_p42 | failed/incomplete preliminary | A100 path | 1 | FAILED | 00:02:02 | 0.034 |
| 1558642 | cv256B_full_s44_r1 | failed/incomplete preliminary | A100 path | 1 | FAILED | 00:02:56 | 0.049 |
| 1558643 | c256kidA_p42_r1 | failed/incomplete preliminary | A100 path | 1 | FAILED | 00:02:41 | 0.045 |
| 1570245 | c256brA_p42 | failed/incomplete preliminary | A100 path | 1 | FAILED | 00:01:19 | 0.022 |
| 1570246 | c256brB_p42 | failed/incomplete preliminary | A100 path | 1 | FAILED | 00:01:19 | 0.022 |
| 1570247 | c256colA_p42 | failed/incomplete preliminary | A100 path | 1 | FAILED | 00:01:19 | 0.022 |
| 1570248 | c256colB_p42 | failed/incomplete preliminary | A100 path | 1 | FAILED | 00:01:19 | 0.022 |
| 1570249 | c256lunA_p42 | failed/incomplete preliminary | A100 path | 1 | FAILED | 00:01:09 | 0.019 |
| 1570252 | c256ovaB_p42 | failed/incomplete preliminary | A100 path | 1 | FAILED | 00:02:02 | 0.034 |
| 1570253 | c256panA_p42 | failed/incomplete preliminary | A100 path | 1 | FAILED | 00:00:53 | 0.015 |
| 1570255 | c256skiA_p42 | failed/incomplete preliminary | A100 path | 1 | FAILED | 00:01:05 | 0.018 |
| 1570256 | c256skiB_p42 | failed/incomplete preliminary | A100 path | 1 | FAILED | 00:00:43 | 0.012 |
| 1570257 | c256brA_p42_r1 | failed/incomplete preliminary | A100 path | 1 | FAILED | 00:00:44 | 0.012 |
| 1570258 | c256brB_p42_r1 | failed/incomplete preliminary | A100 path | 1 | FAILED | 00:00:42 | 0.012 |
| 1570259 | c256colA_p42_r1 | failed/incomplete preliminary | A100 path | 1 | FAILED | 00:00:43 | 0.012 |
| 1570260 | c256colB_p42_r1 | failed/incomplete preliminary | A100 path | 1 | FAILED | 00:00:44 | 0.012 |
| 1570261 | c256lunA_p42_r1 | failed/incomplete preliminary | A100 path | 1 | FAILED | 00:00:43 | 0.012 |
| 1570262 | c256panA_p42_r1 | failed/incomplete preliminary | A100 path | 1 | FAILED | 00:00:42 | 0.012 |
| 1614901 | cv256A_timing_samegpu | noncanonical timing | A100 path | 4 | CANCELLED | 00:00:31 | 0.034 |
| 1614903 | cv256A_timing_samegpu | noncanonical timing | A100 path | 4 | RUNNING | 00:49:22 | 3.291 |
| **Total (236 allocations)** |  |  |  |  |  |  | **971.362** |
