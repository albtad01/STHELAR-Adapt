# Slide-independent KLT seed-43 training jobs — 2026-08-20

Eight training jobs were submitted after revalidating the existing materialized
Fold A/B datasets. No preprocessing, Frozen, seed-44, tissue-specific, or brain
job was submitted. All jobs use the campaign-specific post-run retention wrapper
`ruche/slurm_train_checkpoint10_only.sh` and have no dependency.

| Fold | Method | Seed | Job ID | State at 11:27 CEST | Config | GPU / partition | Excluded nodes | Submitted (Europe/Paris) | Checkpoint retention |
|---|---|---:|---:|---|---|---|---|---|---|
| A | LP | 43 | 1468274 | PENDING (`Priority`) | `configs/slide_exp/training/training_sthelar40x_klt_5class_slideind_foldA_lp_final_heads_seed43.yaml` | 1 A100, `gpua100`; 8 CPUs; 64 GiB; 24 h | `ruche-gpu14,ruche-gpu16,ruche-gpu17` | 2026-08-20 11:23:38 | After successful checkpoint-10 inference, retain only `checkpoint_10.pth`; metadata in `<run>/checkpoint_retention_metadata.json`. |
| B | LP | 43 | 1468275 | PENDING (`Priority`) | `configs/slide_exp/training/training_sthelar40x_klt_5class_slideind_foldB_lp_final_heads_seed43.yaml` | 1 A100, `gpua100`; 8 CPUs; 64 GiB; 24 h | `ruche-gpu14,ruche-gpu16,ruche-gpu17` | 2026-08-20 11:23:41 | After successful checkpoint-10 inference, retain only `checkpoint_10.pth`; metadata in `<run>/checkpoint_retention_metadata.json`. |
| A | Selected PEFT | 43 | 1468276 | PENDING (`Priority`) | `configs/slide_exp/training/training_sthelar40x_klt_5class_slideind_foldA_lora_adaptformer_heads_seed43.yaml` | 1 A100, `gpua100`; 8 CPUs; 64 GiB; 24 h | `ruche-gpu14,ruche-gpu16,ruche-gpu17` | 2026-08-20 11:23:44 | After successful checkpoint-10 inference, retain only `checkpoint_10.pth`; metadata in `<run>/checkpoint_retention_metadata.json`. |
| B | Selected PEFT | 43 | 1468277 | PENDING (`Priority`) | `configs/slide_exp/training/training_sthelar40x_klt_5class_slideind_foldB_lora_adaptformer_heads_seed43.yaml` | 1 A100, `gpua100`; 8 CPUs; 64 GiB; 24 h | `ruche-gpu14,ruche-gpu16,ruche-gpu17` | 2026-08-20 11:23:47 | After successful checkpoint-10 inference, retain only `checkpoint_10.pth`; metadata in `<run>/checkpoint_retention_metadata.json`. |
| A | NT-header1 PEFT | 43 | 1468278 | PENDING (`Priority`) | `configs/slide_exp/training/training_sthelar40x_klt_5class_slideind_foldA_lora_adaptformer_nt_header1_np_hv_heads_seed43.yaml` | 1 A100, `gpua100`; 8 CPUs; 64 GiB; 24 h | `ruche-gpu14,ruche-gpu16,ruche-gpu17` | 2026-08-20 11:23:51 | After successful checkpoint-10 inference, retain only `checkpoint_10.pth`; metadata in `<run>/checkpoint_retention_metadata.json`. |
| B | NT-header1 PEFT | 43 | 1468279 | PENDING (`Priority`) | `configs/slide_exp/training/training_sthelar40x_klt_5class_slideind_foldB_lora_adaptformer_nt_header1_np_hv_heads_seed43.yaml` | 1 A100, `gpua100`; 8 CPUs; 64 GiB; 24 h | `ruche-gpu14,ruche-gpu16,ruche-gpu17` | 2026-08-20 11:23:54 | After successful checkpoint-10 inference, retain only `checkpoint_10.pth`; metadata in `<run>/checkpoint_retention_metadata.json`. |
| A | FullFT | 43 | 1468280 | PENDING (`Priority`) | `configs/slide_exp/training/training_sthelar40x_klt_5class_slideind_foldA_fullft_seed43.yaml` | 1 A100, `gpua100`; 8 CPUs; 64 GiB; 24 h | `ruche-gpu14,ruche-gpu16,ruche-gpu17` | 2026-08-20 11:23:57 | After successful checkpoint-10 inference, retain only `checkpoint_10.pth`; metadata in `<run>/checkpoint_retention_metadata.json`. |
| B | FullFT | 43 | 1468281 | PENDING (`Priority`) | `configs/slide_exp/training/training_sthelar40x_klt_5class_slideind_foldB_fullft_seed43.yaml` | 1 A100, `gpua100`; 8 CPUs; 64 GiB; 24 h | `ruche-gpu14,ruche-gpu16,ruche-gpu17` | 2026-08-20 11:24:01 | After successful checkpoint-10 inference, retain only `checkpoint_10.pth`; metadata in `<run>/checkpoint_retention_metadata.json`. |

## Validated parameter counts

| Method | Trainable | Total | Trainable percentage |
|---|---:|---:|---:|
| LP | 650 | 699,736,523 | 0.00009289% |
| Selected PEFT | 7,908,779 | 707,644,395 | 1.11762052% |
| NT-header1 PEFT | 7,945,835 | 707,644,395 | 1.12285705% |
| FullFT | 699,736,523 | 699,736,523 | 100.00000000% |

For NT-header1, the 300 trainable parameter names exactly matched the canonical
within-slide seed-42 log. The NP/HV decoder bodies remained frozen; only their
final heads were trainable.

## Efficiency output roots

Each timestamped run writes training metrics to
`<configured-log-dir>/<timestamped-run>/efficiency_metrics.json`, inference
metrics to `inference_efficiency_metrics.json`, and checkpoint-retention
provenance to `checkpoint_retention_metadata.json`. The training JSON is the
authoritative source for trainable and total parameter counts. The collector
`utils/collect_slide_exp_efficiency.py` recognizes LP, Selected PEFT,
NT-header1 PEFT, and FullFT as separate methods.

## Infrastructure retries and fixed checkpoint policy — 2026-08-21

The checkpoint policy was frozen before these submissions:

- primary evaluation checkpoint: `checkpoint_10.pth`;
- final persistent state: exactly one `.pth`, `checkpoint_10.pth`;
- `model_best.pth` is removed by the post-run wrapper only after epoch-10
  readability, successful epoch-10 inference, completed efficiency/result
  JSONs, and an atomic retention-metadata write have all been verified;
- best validation epoch and score remain in
  `checkpoint_retention_metadata.json`.

SLURM accounting confirmed that jobs 1468276--1468281 failed on
`ruche-gpu11` without a completed training epoch, checkpoint, inference, or
scientific result. `squeue --me` was empty immediately before submission, so no
replacement was duplicated. The two completed LP seed-43 jobs were not rerun.

All replacements request `gpua100`, one A100, 8 CPUs, 64 GiB RAM, and 24 hours.
They are constrained to the campaign-proven `ruche-gpu19` and also explicitly
exclude `ruche-gpu11,ruche-gpu14,ruche-gpu16,ruche-gpu17`.

| Fold | Method | Seed | Old failed job | Replacement job | State at 2026-08-21 11:17 CEST | Config | Dependency | Checkpoint policy |
|---|---|---:|---:|---:|---|---|---|---|
| A | Selected PEFT | 43 | 1468276 | 1469655 | PENDING (`Priority`) | `configs/slide_exp/training/training_sthelar40x_klt_5class_slideind_foldA_lora_adaptformer_heads_seed43.yaml` | none | fixed epoch 10; retain checkpoint 10 only |
| B | Selected PEFT | 43 | 1468277 | 1469656 | PENDING (`Priority`) | `configs/slide_exp/training/training_sthelar40x_klt_5class_slideind_foldB_lora_adaptformer_heads_seed43.yaml` | none | fixed epoch 10; retain checkpoint 10 only |
| A | NT-header1 PEFT | 43 | 1468278 | 1469657 | PENDING (`Priority`) | `configs/slide_exp/training/training_sthelar40x_klt_5class_slideind_foldA_lora_adaptformer_nt_header1_np_hv_heads_seed43.yaml` | none | fixed epoch 10; retain checkpoint 10 only |
| B | NT-header1 PEFT | 43 | 1468279 | 1469658 | PENDING (`Priority`) | `configs/slide_exp/training/training_sthelar40x_klt_5class_slideind_foldB_lora_adaptformer_nt_header1_np_hv_heads_seed43.yaml` | none | fixed epoch 10; retain checkpoint 10 only |
| A | FullFT | 43 | 1468280 | 1469659 | PENDING (`Dependency`) | `configs/slide_exp/training/training_sthelar40x_klt_5class_slideind_foldA_fullft_seed43.yaml` | `afterok:1469655:1469656:1469657:1469658` | fixed epoch 10; retain checkpoint 10 only |
| B | FullFT | 43 | 1468281 | 1469660 | PENDING (`Dependency`) | `configs/slide_exp/training/training_sthelar40x_klt_5class_slideind_foldB_fullft_seed43.yaml` | `afterok:1469659` | fixed epoch 10; retain checkpoint 10 only |

Submission times were 2026-08-21 11:15:33, 11:15:48, 11:16:00,
11:16:08, 11:16:17, and 11:16:30 CEST, respectively. Each timestamped
replacement run will write:

- `efficiency_metrics.json`;
- `inference_efficiency_metrics.json`;
- `inference_results.json`;
- `checkpoint_retention_metadata.json`.

## Disk preflight and safe cleanup

Before cleanup, `ruche-quota` reported 447 GiB / 500 GiB and `run/` occupied
314 GiB. Six `model_best.pth` names from completed slide-independent runs met
all fixed-epoch retention preconditions. Every deletion was recorded atomically
in its run-local `checkpoint_retention_metadata.json`; each affected checkpoint
directory now contains only `checkpoint_10.pth`.

| Removed path | Logical bytes | Physical bytes reclaimed |
|---|---:|---:|
| `run/sthelar40x_klt_5class_slideind_foldA_fullft_lr1e-5_e10_seed42/log/2026-08-19T110413_sthelar40x_klt_5class_slideind_foldA_fullft_lr1e-5_e10_seed42/checkpoints/model_best.pth` | 8,397,780,353 | 8,397,780,353 |
| `run/sthelar40x_klt_5class_slideind_foldA_lora_adaptformer_r8_a8_red16_heads_e10_seed42/log/2026-08-19T105648_sthelar40x_klt_5class_slideind_foldA_lora_adaptformer_r8_a8_red16_heads_e10_seed42/checkpoints/model_best.pth` | 2,894,602,378 | 0 (hard link to checkpoint 10) |
| `run/sthelar40x_klt_5class_slideind_foldA_lora_adaptformer_r8_a8_red16_nt_header1_np_hv_heads_e10_seed42/log/2026-08-19T183125_sthelar40x_klt_5class_slideind_foldA_lora_adaptformer_r8_a8_red16_nt_header1_np_hv_heads_e10_seed42/checkpoints/model_best.pth` | 2,894,896,652 | 2,894,896,652 |
| `run/sthelar40x_klt_5class_slideind_foldA_lp_final_heads_e10_seed42/log/2026-08-19T114405_sthelar40x_klt_5class_slideind_foldA_lp_final_heads_e10_seed42/checkpoints/model_best.pth` | 2,799,325,178 | 0 (hard link to checkpoint 10) |
| `run/sthelar40x_klt_5class_slideind_foldB_fullft_lr1e-5_e10_seed42/log/2026-08-19T181004_sthelar40x_klt_5class_slideind_foldB_fullft_lr1e-5_e10_seed42/checkpoints/model_best.pth` | 8,397,780,353 | 8,397,780,353 |
| `run/sthelar40x_klt_5class_slideind_foldB_lp_final_heads_e10_seed42/log/2026-08-19T123449_sthelar40x_klt_5class_slideind_foldB_lp_final_heads_e10_seed42/checkpoints/model_best.pth` | 2,799,322,874 | 2,799,322,874 |

The six removed names represented 28,183,707,788 logical bytes. Actual
reclaimed storage was **22,489,780,232 bytes (20.945240 GiB)** because two pairs
already shared their payload through hard links. After cleanup,
`ruche-quota` reported 426 GiB / 500 GiB and `run/` occupied 293 GiB.

No failed seed-43 attempt contained a `.pth`, no
`.checkpoint_retention_smoke` artifact existed, and no other ambiguous,
canonical COMPAYL, release, pretrained, adapter, or run artifact was removed.

## Storage projection and staging

Measured checkpoint sizes from the matched completed seed-42 runs give a final
additional footprint of 28,374,586,432 bytes (**26.425893 GiB**) for these six
retries. The projected final quota usage is therefore approximately
**452.426 / 500 GiB**.

During an atomic epoch checkpoint save, the validation-best state, previous
numeric checkpoint, and temporary new checkpoint can briefly coexist. Running
all six without staging could therefore add approximately **79.277679 GiB**,
exceeding the roughly 74 GiB free after cleanup. The submitted dependency graph
is:

```text
1469655 A Selected PEFT ----+
1469656 B Selected PEFT ----+
1469657 A NT-header1 -------+--> afterok 1469659 A FullFT
1469658 B NT-header1 -------+                    |
                                                 +--> afterok 1469660 B FullFT
```

The largest projected staged addition is **42.067993 GiB**, at the Fold B
FullFT phase after the four small checkpoints and Fold A FullFT persist. This
projects a temporary high-water quota of approximately **468.068 / 500 GiB**,
leaving about **31.932 GiB** headroom.

## GPU-19 infrastructure failure and second retry — 2026-08-22

Job 1469655 progressed through eight completed epochs on `ruche-gpu19`, then
failed during back-propagation with `CUDA error: unspecified launch failure`.
The node subsequently stopped providing a usable CUDA driver, so jobs
1469656--1469658 failed at `model.to(device)` with `CUDA driver initialization
failed`. Slurm later placed `ruche-gpu19` in `DRAIN+POWERED_DOWN` with reason
`Kill task failed (JobId=1470187 StepId=0)`. These are infrastructure failures,
not scientific results.

The blocked jobs 1469659 and 1469660 were cancelled and replaced together with
the four failed upstream jobs. The original configs, resources, fixed epoch-10
policy, and staged dependency structure are unchanged. The exclusion list is
now `ruche-gpu11,ruche-gpu14,ruche-gpu16,ruche-gpu17,ruche-gpu19`; no requested
node is set, so the jobs remain pending until another permitted A100 node is
healthy.

| Fold | Method | Seed | Failed/blocked job | Replacement job | State at 2026-08-22 18:31 CEST | Dependency |
|---|---|---:|---:|---:|---|---|
| A | Selected PEFT | 43 | 1469655 | 1472754 | PENDING (`ReqNodeNotAvail`) | none |
| B | Selected PEFT | 43 | 1469656 | 1472755 | PENDING (`ReqNodeNotAvail`) | none |
| A | NT-header1 PEFT | 43 | 1469657 | 1472756 | PENDING (`ReqNodeNotAvail`) | none |
| B | NT-header1 PEFT | 43 | 1469658 | 1472757 | PENDING (`ReqNodeNotAvail`) | none |
| A | FullFT | 43 | 1469659 | 1472758 | PENDING (`Dependency`) | `afterok:1472754:1472755:1472756:1472757` |
| B | FullFT | 43 | 1469660 | 1472759 | PENDING (`Dependency`) | `afterok:1472758` |

The partial job-1469655 run initially retained `checkpoint_8.pth` and
`model_best.pth` until the fixed-policy cleanup documented below. Its incomplete
efficiency record and all failure provenance remain. It is not selected for test
inference or used as a campaign result.

## Hardware-matched A100 resubmission — 2026-08-23

The six pending retry-2 jobs 1472754--1472759 were verified to have elapsed
zero seconds, no assigned node, and no new run result before cancellation. They
were cancelled without touching any completed scientific job. The replacement
jobs remain on `gpua100`; there is no V100 fallback and no requested node.

The live node audit found zero currently healthy eligible A100 nodes. Jobs
exclude `ruche-gpu11,ruche-gpu14,ruche-gpu16,ruche-gpu17,ruche-gpu19` because
their recent CUDA/driver failure or diagnostic state remains unresolved.
`ruche-gpu12,ruche-gpu13,ruche-gpu15,ruche-gpu18` are currently down for
`ResumeTimeout`, but are deliberately not excluded because they have no recent
campaign CUDA-failure evidence; the jobs can therefore become eligible
automatically if an administrator restores any of them. No other accessible
A100 partition was visible.

| Fold | Method | Seed | Backbone | Old pending ID | Cancellation | New ID | State at submission | Dependency | Walltime | Config |
|---|---|---:|---|---:|---|---:|---|---|---|---|
| A | Selected PEFT | 43 | CellViT-SAM-H x40 | 1472754 | `CANCELLED`, elapsed 0 | 1473855 | `PENDING (ReqNodeNotAvail)` | none | 16 h | `configs/slide_exp/training/training_sthelar40x_klt_5class_slideind_foldA_lora_adaptformer_heads_seed43.yaml` |
| B | Selected PEFT | 43 | CellViT-SAM-H x40 | 1472755 | `CANCELLED`, elapsed 0 | 1473856 | `PENDING (ReqNodeNotAvail)` | none | 16 h | `configs/slide_exp/training/training_sthelar40x_klt_5class_slideind_foldB_lora_adaptformer_heads_seed43.yaml` |
| A | NT-header1 PEFT | 43 | CellViT-SAM-H x40 | 1472756 | `CANCELLED`, elapsed 0 | 1473857 | `PENDING (ReqNodeNotAvail)` | none | 16 h | `configs/slide_exp/training/training_sthelar40x_klt_5class_slideind_foldA_lora_adaptformer_nt_header1_np_hv_heads_seed43.yaml` |
| B | NT-header1 PEFT | 43 | CellViT-SAM-H x40 | 1472757 | `CANCELLED`, elapsed 0 | 1473858 | `PENDING (ReqNodeNotAvail)` | none | 16 h | `configs/slide_exp/training/training_sthelar40x_klt_5class_slideind_foldB_lora_adaptformer_nt_header1_np_hv_heads_seed43.yaml` |
| A | FullFT | 43 | CellViT-SAM-H x40 | 1472758 | `CANCELLED`, elapsed 0 | 1473859 | `PENDING (Dependency)` | `afterok:1473855:1473856:1473857:1473858` | 16 h | `configs/slide_exp/training/training_sthelar40x_klt_5class_slideind_foldA_fullft_seed43.yaml` |
| B | FullFT | 43 | CellViT-SAM-H x40 | 1472759 | `CANCELLED`, elapsed 0 | 1473860 | `PENDING (Dependency)` | `afterok:1473859` | 16 h | `configs/slide_exp/training/training_sthelar40x_klt_5class_slideind_foldB_fullft_seed43.yaml` |

All six were submitted at 2026-08-23 10:12:36 CEST with one A100, eight
CPUs, 64 GiB RAM, batch size 4, AMP, and the validated seed-43 configs. The
16-hour scheduler limit has more than six hours of margin over the maximum
observed complete matched job elapsed time (9:35:28); scientific training still
runs exactly ten epochs. Parameter scopes remain Selected PEFT 7,908,779 /
707,644,395, NT-header1 7,945,835 / 707,644,395, and FullFT 699,736,523 /
699,736,523.

The checkpoint policy remains fixed epoch 10. Final inference must load
`checkpoint_10.pth`; atomic retention metadata preserve validation history and
exactly that one `.pth` remains after successful inference. Efficiency JSONs
now record `gpu_model`, `partition`, and explicit backbone identity.

## Restored A100 nodes and retry 4 — 2026-08-24

`ruche-gpu13` and `ruche-gpu15` returned to service on 2026-08-24. Two
CellViT-256 FullFT jobs began successful CUDA training on `ruche-gpu13`, while
the four independent SAM-H jobs from retry 3 had already failed with
`NODE_FAIL` on `ruche-gpu18`. Their downstream jobs 1473859 and 1473860 could
therefore never satisfy their `afterok` dependencies and were cancelled.

The retry-4 jobs permit Slurm to choose either restored node 13 or 15 and
exclude `ruche-gpu11,ruche-gpu12,ruche-gpu14,ruche-gpu16,ruche-gpu17,ruche-gpu18,ruche-gpu19`.
Nodes 12 and 18 were added because they caused the immediately preceding
`NODE_FAIL` attempts. No requested node is set.

| Fold | Method | Seed | Failed/blocked job | Retry-4 job | State at 2026-08-24 12:25 CEST | Node | Dependency |
|---|---|---:|---:|---:|---|---|---|
| A | Selected PEFT | 43 | 1473855 | 1475933 | RUNNING | `ruche-gpu15` | none |
| B | Selected PEFT | 43 | 1473856 | 1475934 | RUNNING | `ruche-gpu15` | none |
| A | NT-header1 PEFT | 43 | 1473857 | 1475935 | PENDING (`QOSMaxGRESPerUser`) | unassigned | none |
| B | NT-header1 PEFT | 43 | 1473858 | 1475936 | PENDING (`QOSMaxGRESPerUser`) | unassigned | none |
| A | FullFT | 43 | 1473859 | 1475941 | PENDING (`Dependency`) | unassigned | `afterok:1475933:1475934:1475935:1475936` |
| B | FullFT | 43 | 1473860 | 1475942 | PENDING (`Dependency`) | unassigned | `afterok:1475941` |

All retry-4 jobs use `gpua100`, one A100, 8 CPUs, 64 GiB RAM, the unchanged
validated configs, fixed epoch-10 evaluation, and checkpoint-10-only retention
after successful inference. Submission timestamps span 12:24:49--12:25:26
CEST. The dependency graph continues to stage the two large FullFT checkpoints
for disk safety.

### FullFT Fold B dependency release — 2026-08-25

At user request, the `afterok:1475941` scheduling dependency was removed from
job 1475942 after the campaign storage constraint was reassessed. Immediately
before the update, job 1475942 was still pending with zero elapsed time and no
assigned node. No scientific configuration, resource request, checkpoint
policy, exclusion, or input dataset was changed. Slurm started job 1475942 on
`ruche-gpu15` at 2026-08-25 10:55:04 CEST, concurrently with Fold A FullFT job
1475941 on `ruche-gpu13`.

Before resubmission, two non-scientific checkpoint files from failed job
1469655 were removed while all failure provenance was retained. Details are in
`reports/storage_cleanup_a100_resubmission_20260823.md`. Quota changed from
431/500 GiB to 426/500 GiB. Conservative projections from measured checkpoint
sizes are 453.82/500 GiB final and 470.86/500 GiB temporary high-water when the
small CellViT-256 transient footprint is allowed to overlap the staged SAM-H
peak.

## Final scientific reconciliation — 2026-08-27

The final retry-4 jobs all completed successfully. Earlier FAILED, NODE_FAIL,
dependency-blocked, and zero-runtime cancelled attempts are SUPERSEDED
infrastructure provenance, not stochastic replicates. The canonical seed-43
conditions are:

| Fold | Method | Canonical job | Final status | Canonical run directory |
|---|---|---:|---|---|
| A | LP | 1468274 | COMPLETED VALID | `run/sthelar40x_klt_5class_slideind_foldA_lp_final_heads_e10_seed43/log/2026-08-20T165926_sthelar40x_klt_5class_slideind_foldA_lp_final_heads_e10_seed43/` |
| B | LP | 1468275 | COMPLETED VALID | `run/sthelar40x_klt_5class_slideind_foldB_lp_final_heads_e10_seed43/log/2026-08-20T194225_sthelar40x_klt_5class_slideind_foldB_lp_final_heads_e10_seed43/` |
| A | Selected PEFT | 1475933 | COMPLETED VALID | `run/sthelar40x_klt_5class_slideind_foldA_lora_adaptformer_r8_a8_red16_heads_e10_seed43/log/2026-08-24T122637_sthelar40x_klt_5class_slideind_foldA_lora_adaptformer_r8_a8_red16_heads_e10_seed43/` |
| B | Selected PEFT | 1475934 | COMPLETED VALID | `run/sthelar40x_klt_5class_slideind_foldB_lora_adaptformer_r8_a8_red16_heads_e10_seed43/log/2026-08-24T122637_sthelar40x_klt_5class_slideind_foldB_lora_adaptformer_r8_a8_red16_heads_e10_seed43/` |
| A | NT-header1 | 1475935 | COMPLETED VALID | `run/sthelar40x_klt_5class_slideind_foldA_lora_adaptformer_r8_a8_red16_nt_header1_np_hv_heads_e10_seed43/log/2026-08-24T175805_sthelar40x_klt_5class_slideind_foldA_lora_adaptformer_r8_a8_red16_nt_header1_np_hv_heads_e10_seed43/` |
| B | NT-header1 | 1475936 | COMPLETED VALID | `run/sthelar40x_klt_5class_slideind_foldB_lora_adaptformer_r8_a8_red16_nt_header1_np_hv_heads_e10_seed43/log/2026-08-24T183826_sthelar40x_klt_5class_slideind_foldB_lora_adaptformer_r8_a8_red16_nt_header1_np_hv_heads_e10_seed43/` |
| A | FullFT | 1475941 | COMPLETED VALID | `run/sthelar40x_klt_5class_slideind_foldA_fullft_lr1e-5_e10_seed43/log/2026-08-25T015318_sthelar40x_klt_5class_slideind_foldA_fullft_lr1e-5_e10_seed43/` |
| B | FullFT | 1475942 | COMPLETED VALID | `run/sthelar40x_klt_5class_slideind_foldB_fullft_lr1e-5_e10_seed43/log/2026-08-25T105706_sthelar40x_klt_5class_slideind_foldB_fullft_lr1e-5_e10_seed43/` |

Every row contains `checkpoint_10.pth`, completed canonical test inference,
both efficiency JSONs, and retention metadata. Seeds 42 and 43 therefore form
the current two-seed SAM-H held-out-slide evidence. Seed 44 is MISSING for LP,
Selected PEFT and FullFT and was not launched in this audit. Primary scientific
role: Selected PEFT and FullFT MAIN PAPER, LP baseline MAIN PAPER, NT-header1
SUPPORTING ANALYSIS.
