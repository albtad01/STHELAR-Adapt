# KLT checkpoint epoch 10 versus validation-best audit

> **Later reconciliation:** this file was an interim 12-run snapshot. After the
> remaining comparable KLT runs completed, the authoritative combined count is
> 4/16 (25.00%) with epoch 10 validation-best: LP 1/5, Selected PEFT 2/4,
> NT-header1 0/3, and FullFT 1/4. See
> `reports/checkpoint_selection_policy_audit.md`. The campaign nevertheless
> fixed `checkpoint_10.pth` as primary for original-paper consistency and an
> equal ten-epoch optimization budget; the decision was not made from test
> performance.

Date: 2026-08-20

## Scope and method

This is a read-only audit of completed, comparable 10-epoch canonical KLT runs for LP, Selected PEFT, NT-header1 PEFT, and FullFT. It includes canonical within-slide runs and the available completed seed-42 slide-independent runs. A run was included only when it had ten logged validation epochs and completed final inference.

The checkpoint-selection score is validation bPQ, parsed from each `Validation epoch stats` record. The validation-best epoch is the last epoch followed by `New best model - save checkpoint`, matching the trainer's strict-improvement behavior. Scores below are the four-decimal values preserved in `logs.log`; no test metric was used for checkpoint selection or for this audit.

## Result

Across 12 completed comparable runs, epoch 10 was validation-best in **4/12 runs (33.33%)**.

| Method | Runs | Best epochs | Epoch 10 best | Percentage |
|---|---:|---|---:|---:|
| LP | 3 | 3, 10, 4 | 1/3 | 33.33% |
| Selected PEFT | 3 | 10, 9, 10 | 2/3 | 66.67% |
| NT-header1 PEFT | 2 | 8, 7 | 0/2 | 0.00% |
| FullFT | 4 | 7, 10, 8, 9 | 1/4 | 25.00% |

| Protocol | Method | Seed | Fold | Best epoch | Best validation bPQ | Epoch-10 validation bPQ | Epoch 10 best? |
|---|---|---:|---|---:|---:|---:|---|
| Within-slide | FullFT | 42 | - | 7 | 0.6268 | 0.6199 | No |
| Within-slide | FullFT | 43 | - | 10 | 0.6291 | 0.6291 | Yes |
| Within-slide | LP | 42 | - | 3 | 0.5521 | 0.5496 | No |
| Within-slide | Selected PEFT | 42 | - | 10 | 0.6179 | 0.6179 | Yes |
| Within-slide | Selected PEFT | 43 | - | 9 | 0.6145 | 0.6121 | No |
| Within-slide | NT-header1 PEFT | 42 | - | 8 | 0.6146 | 0.6124 | No |
| Slide-independent | FullFT | 42 | A | 8 | 0.5622 | 0.5619 | No |
| Slide-independent | Selected PEFT | 42 | A | 10 | 0.5463 | 0.5463 | Yes |
| Slide-independent | NT-header1 PEFT | 42 | A | 7 | 0.5462 | 0.5442 | No |
| Slide-independent | LP | 42 | A | 10 | 0.5020 | 0.5020 | Yes |
| Slide-independent | FullFT | 42 | B | 9 | 0.5003 | 0.4845 | No |
| Slide-independent | LP | 42 | B | 4 | 0.4142 | 0.4081 | No |

## Interpretation

Epoch 10 is not generally identical to `model_best.pth`. In this limited sample, NT-header1 peaked earlier in both available runs and FullFT peaked earlier in three of four runs; Selected PEFT reached epoch 10 most often. These are descriptive tendencies only because method-specific sample sizes are small and protocols differ.

For the slide-independent campaign, `checkpoint_10.pth` remains the primary checkpoint by predeclared protocol consistency across folds and seeds. This audit does not reinterpret or replace any primary test result. Each new run will preserve its best validation epoch/score and epoch-10 score in `checkpoint_retention_metadata.json` before the redundant best checkpoint is removed.
