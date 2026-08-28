# CellViT-256 x40 Slide-Independent Seed42 Jobs

Submission timestamp: 2026-08-21 11:58:18-19 CEST (Europe/Paris).

All jobs request one GPU in `gpua100`, 8 CPUs, and 64 GiB. The submission-level exclusion is `ruche-gpu11,ruche-gpu14,ruche-gpu16,ruche-gpu17` (`scontrol`: `ruche-gpu[11,14,16-17]`). No job has a dependency, requested node, or assigned node at the state snapshot. Frozen requests 8 hours; LP and FullFT request 24 hours.

| Fold | Backbone | Condition | Seed | Trainable / total params | Config | Job ID | State at 2026-08-21 11:58:59 CEST | Node | GPU | Checkpoint | Dependency | Submitted |
|---|---|---|---:|---:|---|---:|---|---|---|---|---|---|
| A | CellViT-256 x40 | Frozen | deterministic seed42-equivalent | 0 / 46,750,349 | `configs/slide_exp/training/evaluation_sthelar40x_klt_5class_slideind_foldA_cellvit256_frozen_seed42.yaml` | 1469778 | PENDING (Priority) | unassigned | 1 GPU, `gpua100` | untouched `${PROJECT_ROOT}/models/pretrained/CellViT-256-x40.pth` | none | 2026-08-21 11:58:18 CEST |
| B | CellViT-256 x40 | Frozen | deterministic seed42-equivalent | 0 / 46,750,349 | `configs/slide_exp/training/evaluation_sthelar40x_klt_5class_slideind_foldB_cellvit256_frozen_seed42.yaml` | 1469779 | PENDING (Priority) | unassigned | 1 GPU, `gpua100` | untouched `${PROJECT_ROOT}/models/pretrained/CellViT-256-x40.pth` | none | 2026-08-21 11:58:18 CEST |
| A | CellViT-256 x40 | LP | 42 | 650 / 46,743,419 | `configs/slide_exp/training/training_sthelar40x_klt_5class_slideind_foldA_cellvit256_lp_seed42.yaml` | 1469780 | PENDING (Priority) | unassigned | 1 GPU, `gpua100` | `<timestamped-run>/checkpoints/checkpoint_10.pth` | none | 2026-08-21 11:58:19 CEST |
| B | CellViT-256 x40 | LP | 42 | 650 / 46,743,419 | `configs/slide_exp/training/training_sthelar40x_klt_5class_slideind_foldB_cellvit256_lp_seed42.yaml` | 1469781 | PENDING (Priority) | unassigned | 1 GPU, `gpua100` | `<timestamped-run>/checkpoints/checkpoint_10.pth` | none | 2026-08-21 11:58:19 CEST |
| A | CellViT-256 x40 | FullFT | 42 | 46,743,419 / 46,743,419 | `configs/slide_exp/training/training_sthelar40x_klt_5class_slideind_foldA_cellvit256_fullft_seed42.yaml` | 1469782 | PENDING (Priority) | unassigned | 1 GPU, `gpua100` | `<timestamped-run>/checkpoints/checkpoint_10.pth` | none | 2026-08-21 11:58:19 CEST |
| B | CellViT-256 x40 | FullFT | 42 | 46,743,419 / 46,743,419 | `configs/slide_exp/training/training_sthelar40x_klt_5class_slideind_foldB_cellvit256_fullft_seed42.yaml` | 1469783 | PENDING (Priority) | unassigned | 1 GPU, `gpua100` | `<timestamped-run>/checkpoints/checkpoint_10.pth` | none | 2026-08-21 11:58:19 CEST |

## Checkpoint and instrumentation policy

- Trainable runs evaluate `checkpoint_10.pth`, not validation-best or latest-by-test behavior.
- The campaign wrapper invokes retention only after successful checkpoint-10 inference and completed result/efficiency artifacts.
- Retention metadata preserve all ten validation scores, best validation epoch/score, and final validation score before `model_best.pth` and checkpoints 1-9 are removed.
- Exactly one full `.pth`, `checkpoint_10.pth`, remains after successful retention.
- Training and inference efficiency JSONs use the existing slide-independent recorder and identify the backbone as `CellViT-256 x40`.
- Frozen keeps the untouched pretrained checkpoint via a symlink in the evaluation run and creates no trained checkpoint.

## Scheduler coexistence

The post-submission `squeue --me` snapshot also showed the six pre-existing SAM-H seed43 retries unchanged:

- 1469655-1469658: PENDING (Priority)
- 1469659-1469660: PENDING (Dependency)

No existing job was cancelled, updated, reprioritized, or otherwise modified.

## Scope confirmation

Submitted here: exactly two CellViT-256 Frozen inference jobs, two CellViT-256 LP training jobs, and two CellViT-256 FullFT training jobs. No CellViT-256 PEFT, HoVer-Net, CellViT++, seed43, tissue-specific, brain, preprocessing, or patch-generation job was submitted.

## GPU-19 infrastructure failure and retries — 2026-08-22

All six original jobs started on `ruche-gpu19` after that node had suffered a
CUDA failure. Jobs 1469778--1469783 failed before model inference/training at
`model.to(device)` with `CUDA driver initialization failed`; therefore they
produced no scientific results or trained checkpoints. Slurm subsequently put
the node in `DRAIN+POWERED_DOWN` with reason `Kill task failed`.

The six scientifically identical replacements below were submitted at
18:30 CEST. They keep the same configs, seeds, resources, checkpoint policy,
and evaluation definitions. `ruche-gpu19` was added to the existing exclusions,
and no replacement has a requested node.

| Fold | Backbone | Condition | Seed | Trainable / total params | Original job | Replacement job | State at 2026-08-22 18:31 CEST | Dependency |
|---|---|---|---:|---:|---:|---:|---|---|
| A | CellViT-256 x40 | Frozen | deterministic seed42-equivalent | 0 / 46,750,349 | 1469778 | 1472761 | PENDING (`ReqNodeNotAvail`) | none |
| B | CellViT-256 x40 | Frozen | deterministic seed42-equivalent | 0 / 46,750,349 | 1469779 | 1472762 | PENDING (`ReqNodeNotAvail`) | none |
| A | CellViT-256 x40 | LP | 42 | 650 / 46,743,419 | 1469780 | 1472763 | PENDING (`ReqNodeNotAvail`) | none |
| B | CellViT-256 x40 | LP | 42 | 650 / 46,743,419 | 1469781 | 1472764 | PENDING (`ReqNodeNotAvail`) | none |
| A | CellViT-256 x40 | FullFT | 42 | 46,743,419 / 46,743,419 | 1469782 | 1472765 | PENDING (`ReqNodeNotAvail`) | none |
| B | CellViT-256 x40 | FullFT | 42 | 46,743,419 / 46,743,419 | 1469783 | 1472766 | PENDING (`ReqNodeNotAvail`) | none |

Replacement exclusion list: `ruche-gpu11,ruche-gpu14,ruche-gpu16,ruche-gpu17,ruche-gpu19`.

## Hardware-matched A100 resubmission — 2026-08-23

The six pending retry jobs 1472761--1472766 were verified to have elapsed zero
seconds, no assigned node, and no new result before cancellation. They were
replaced exactly once and remain on `gpua100` so runtime, VRAM, inference
throughput, and storage measurements are hardware-matched to CellViT-SAM-H.
No requested node or V100 fallback is used.

| Fold | Condition | Seed | Backbone | Old pending ID | Cancellation | New ID | State at submission | Walltime | Config |
|---|---|---:|---|---:|---|---:|---|---|---|
| A | Frozen | deterministic seed42-equivalent | CellViT-256 x40 | 1472761 | `CANCELLED`, elapsed 0 | 1473861 | `PENDING (ReqNodeNotAvail)` | 8 h | `configs/slide_exp/training/evaluation_sthelar40x_klt_5class_slideind_foldA_cellvit256_frozen_seed42.yaml` |
| B | Frozen | deterministic seed42-equivalent | CellViT-256 x40 | 1472762 | `CANCELLED`, elapsed 0 | 1473862 | `PENDING (ReqNodeNotAvail)` | 8 h | `configs/slide_exp/training/evaluation_sthelar40x_klt_5class_slideind_foldB_cellvit256_frozen_seed42.yaml` |
| A | LP | 42 | CellViT-256 x40 | 1472763 | `CANCELLED`, elapsed 0 | 1473863 | `PENDING (ReqNodeNotAvail)` | 24 h | `configs/slide_exp/training/training_sthelar40x_klt_5class_slideind_foldA_cellvit256_lp_seed42.yaml` |
| B | LP | 42 | CellViT-256 x40 | 1472764 | `CANCELLED`, elapsed 0 | 1473864 | `PENDING (ReqNodeNotAvail)` | 24 h | `configs/slide_exp/training/training_sthelar40x_klt_5class_slideind_foldB_cellvit256_lp_seed42.yaml` |
| A | FullFT | 42 | CellViT-256 x40 | 1472765 | `CANCELLED`, elapsed 0 | 1473865 | `PENDING (ReqNodeNotAvail)` | 24 h | `configs/slide_exp/training/training_sthelar40x_klt_5class_slideind_foldA_cellvit256_fullft_seed42.yaml` |
| B | FullFT | 42 | CellViT-256 x40 | 1472766 | `CANCELLED`, elapsed 0 | 1473866 | `PENDING (ReqNodeNotAvail)` | 24 h | `configs/slide_exp/training/training_sthelar40x_klt_5class_slideind_foldB_cellvit256_fullft_seed42.yaml` |

All six were submitted at 2026-08-23 10:12:36 CEST with one A100 in
`gpua100`, eight CPUs, and 64 GiB RAM. The exclusion list is
`ruche-gpu11,ruche-gpu14,ruche-gpu16,ruche-gpu17,ruche-gpu19` for the same
unresolved CUDA/diagnostic reasons documented in
`reports/slide_exp_seed43_jobs.md`; currently down nodes 12/13/15/18 remain
allowed so an administrative restoration can release the jobs automatically.
CellViT-256 training keeps the existing 24-hour limit because no completed
matched KLT runtime is yet available. No CellViT-256 job has a dependency.

Validated parameter scopes remain Frozen 0 / 46,750,349, LP 650 /
46,743,419, and FullFT 46,743,419 / 46,743,419. A fresh CPU Frozen smoke
strictly matched every checkpoint key, trained zero parameters, retained the
PanNuke taxonomy, and emitted only class-agnostic metrics. Trainable runs keep
`checkpoint_10.pth` as the primary and sole persistent `.pth` after successful
final inference and atomic retention. Efficiency JSONs explicitly record A100
`gpu_model`, `partition`, and `CellViT-256 x40` backbone identity.

## Restored A100 nodes and retry 3 — 2026-08-24

The retry-2 Frozen and LP jobs failed with `NODE_FAIL` on `ruche-gpu12` before
scientific execution. In contrast, retry-2 FullFT jobs 1473865 and 1473866
started on restored `ruche-gpu13` at 11:49:17 CEST and entered real CUDA
training; they were retained and not duplicated.

The four missing conditions were submitted again with no requested node and
with exclusions `ruche-gpu11,ruche-gpu12,ruche-gpu14,ruche-gpu16,ruche-gpu17,ruche-gpu18,ruche-gpu19`,
allowing only the currently restored gpu13/gpu15 pair.

| Fold | Condition | Seed | Failed job | Retry-3 job | State at 2026-08-24 12:25 CEST | Dependency |
|---|---|---:|---:|---:|---|---|
| A | Frozen | deterministic seed42-equivalent | 1473861 | 1475937 | PENDING (`QOSMaxGRESPerUser`) | none |
| B | Frozen | deterministic seed42-equivalent | 1473862 | 1475938 | PENDING (`QOSMaxGRESPerUser`) | none |
| A | LP | 42 | 1473863 | 1475939 | PENDING (`QOSMaxGRESPerUser`) | none |
| B | LP | 42 | 1473864 | 1475940 | PENDING (`QOSMaxGRESPerUser`) | none |
| A | FullFT | 42 | n/a | 1473865 | RUNNING on `ruche-gpu13` | none |
| B | FullFT | 42 | n/a | 1473866 | RUNNING on `ruche-gpu13` | none |

The four retry-3 jobs were submitted at 12:24:58--12:25:08 CEST. Pending due
to `QOSMaxGRESPerUser` is expected while the account is using its four-A100
concurrency allowance; these jobs must not be duplicated merely for remaining
pending.

## Selected PEFT Fold A/B — 2026-08-25

Two additional seed-42 training jobs fill the missing CellViT-256 Selected
PEFT comparison. They reuse the validated materialized folds and make no
changes to the completed Frozen, LP, or FullFT results.

| Fold | Method | Seed | Job ID | State immediately after submission | Node | Config | Efficiency metrics |
|---|---|---:|---:|---|---|---|---|
| A | Selected PEFT | 42 | 1478320 | RUNNING | `ruche-gpu19` | `configs/slide_exp/training/training_sthelar40x_klt_5class_slideind_foldA_cellvit256_lora_adaptformer_heads_seed42.yaml` | `<timestamped-run>/efficiency_metrics.json` and `<timestamped-run>/inference_efficiency_metrics.json` |
| B | Selected PEFT | 42 | 1478322 | RUNNING | `ruche-gpu11` | `configs/slide_exp/training/training_sthelar40x_klt_5class_slideind_foldB_cellvit256_lora_adaptformer_heads_seed42.yaml` | `<timestamped-run>/efficiency_metrics.json` and `<timestamped-run>/inference_efficiency_metrics.json` |

Submission times were 2026-08-25 11:40:31 and 11:40:35 CEST. Both jobs use
`gpua100`, one A100, eight CPUs, 64 GiB RAM, batch size 4, AMP, and a 24-hour
limit. Neither job has a dependency, requested node, or exclusion list. At
submission, nodes 11, 12, 13, 14, 15, and 19 were returned to service with no
Slurm reason; nodes 16, 17, and 18 remained down with `ResumeTimeout`. Historical
blacklists were therefore not reused after the reported infrastructure repair,
and Slurm selected among currently eligible A100 nodes.

### Architecture and trainability audit

CellViT-256 has 12 ViT blocks with embedding width 384. Each block represents
Q/K/V with one `Linear(384, 1152)` tensor, so the established `LoRAQKV` wrapper
can apply independent rank-8, alpha-8 deltas to the exact Q and V slices without
altering K. The existing AdaptFormer wrapper is compatible with each block's
`norm2` plus `MLP(384, 1536, 384)` residual path; reduction 16 gives a 24-wide
bottleneck with GELU. No approximate or substituted adaptation mechanism is
used.

The strict factory now permits this one CellViT-256 PEFT mode only when
`decoder_train_scope: heads_only`; it fails closed for other decoder scopes.
The validated scope is:

- 48 LoRA tensors: for each block 0--11,
  `attn.qkv.adapter_{q,v}_{down,up}.weight`;
- 60 AdaptFormer tensors: for each block 0--11, `mlp.adapter_alpha`,
  `mlp.adapter_downsample.{weight,bias}`, and
  `mlp.adapter_upsample.{weight,bias}`;
- six final-head tensors:
  `nuclei_binary_map_decoder.decoder0_header.2.{weight,bias}`,
  `hv_map_decoder.decoder0_header.2.{weight,bias}`, and
  `nuclei_type_maps_decoder.decoder0_header.2.{weight,bias}`.

This is exactly 114 trainable tensors, 374,198 trainable parameters, and
47,116,967 total parameters (0.7941894902%). Every encoder base tensor and every
NP/HV/NT decoder-body tensor is frozen. The per-run training efficiency JSON
also stores the expanded exact `trainable_parameter_names` and deduplicated
`trainable_module_names`; `logs.log` records the full parameter-by-parameter
trainability report.

Strict pretrained loading accepted all compatible CellViT-256 x40 tensors and
only the explicitly declared one-class STHELAR tissue-classifier shape change.
A real CPU forward/backward smoke passed for NP, HV, and six-channel NT outputs.
A separate checkpoint smoke reconstructed both adapter types through the actual
inference class, loaded `checkpoint_10.pth` with all keys matched, and created
the held-out-test dataloader successfully.

Fold validation was repeated without regeneration: A has 24,283 / 2,052 /
23,494 train/validation/test patches; B has 20,893 / 2,601 / 26,335. Slide and
patch disjointness, exact fold reversal, and immutable ZIP links all passed.

### Storage and checkpoint policy

At submission the workdir quota was 447 / 500 GiB and `run/` occupied 313 GiB.
Using the measured CellViT-256 LP checkpoint plus the exact added model and Adam
state tensors gives an estimated 191.7 MB (0.179 GiB) checkpoint per PEFT run.
The two-run temporary checkpoint high-water is approximately 0.71 GiB when
`model_best.pth` and `checkpoint_10.pth` coexist; final persistent storage is
approximately 0.36 GiB. The wrapper retains only `checkpoint_10.pth` after
successful epoch-10 inference and atomic retention metadata, while preserving
the four required JSON outputs.

## Final scientific reconciliation — 2026-08-27

All eight Fold-A/B CellViT-256 conditions are now COMPLETED VALID. Earlier
CUDA/NODE_FAIL attempts and zero-runtime cancellations are SUPERSEDED and do
not count as scientific replicates.

| Fold | Method | Canonical job | Final status | Canonical artifact root |
|---|---|---:|---|---|
| A | Frozen | 1475937 | COMPLETED VALID | `run/sthelar40x_klt_5class_slideind_foldA_cellvit256_frozen_seed42/eval/` |
| B | Frozen | 1475938 | COMPLETED VALID | `run/sthelar40x_klt_5class_slideind_foldB_cellvit256_frozen_seed42/eval/` |
| A | LP | 1475939 | COMPLETED VALID | `run/sthelar40x_klt_5class_slideind_foldA_cellvit256_lp_final_heads_e10_seed42/log/2026-08-24T195358_sthelar40x_klt_5class_slideind_foldA_cellvit256_lp_final_heads_e10_seed42/` |
| B | LP | 1475940 | COMPLETED VALID | `run/sthelar40x_klt_5class_slideind_foldB_cellvit256_lp_final_heads_e10_seed42/log/2026-08-24T203638_sthelar40x_klt_5class_slideind_foldB_cellvit256_lp_final_heads_e10_seed42/` |
| A | Selected PEFT | 1478320 | COMPLETED VALID | `run/sthelar40x_klt_5class_slideind_foldA_cellvit256_lora_adaptformer_r8_a8_red16_heads_e10_seed42/log/2026-08-25T114238_sthelar40x_klt_5class_slideind_foldA_cellvit256_lora_adaptformer_r8_a8_red16_heads_e10_seed42/` |
| B | Selected PEFT | 1478322 | COMPLETED VALID | `run/sthelar40x_klt_5class_slideind_foldB_cellvit256_lora_adaptformer_r8_a8_red16_heads_e10_seed42/log/2026-08-25T114238_sthelar40x_klt_5class_slideind_foldB_cellvit256_lora_adaptformer_r8_a8_red16_heads_e10_seed42/` |
| A | FullFT | 1473865 | COMPLETED VALID | `run/sthelar40x_klt_5class_slideind_foldA_cellvit256_fullft_lr1e-5_e10_seed42/log/2026-08-24T115109_sthelar40x_klt_5class_slideind_foldA_cellvit256_fullft_lr1e-5_e10_seed42/` |
| B | FullFT | 1473866 | COMPLETED VALID | `run/sthelar40x_klt_5class_slideind_foldB_cellvit256_fullft_lr1e-5_e10_seed42/log/2026-08-24T115109_sthelar40x_klt_5class_slideind_foldB_cellvit256_fullft_lr1e-5_e10_seed42/` |

Every trainable row contains its retained `checkpoint_10.pth`, canonical test
JSON, both efficiency JSONs, and retention metadata. Frozen has no training
checkpoint and reports only valid class-agnostic outputs. Full-precision
results are in `reports/neurips2026_paper_results/cellvit256_slideind.csv`;
the matched computational comparison and verified adapter-only reconstructions
are in `reports/efficiency_audit/`. Paper role: SUPPORTING ANALYSIS that
contextualizes the primary SAM-H PEFT claim across backbone scale.
