# KLT checkpoint-selection policy audit

> **Final campaign decision, reconciled 2026-08-27:** the recommendation made
> in this 2026-08-21 decision audit was considered but **not adopted**. The
> frozen, final KLT policy is `checkpoint_10.pth` for primary test inference,
> matching the original COMPAYL fixed ten-epoch protocol. All canonical
> slide-independent results in `reports/neurips2026_paper_results/` use epoch
> 10. No completed run is to be retrained merely to recover `model_best.pth`,
> and no checkpoint choice is to be revisited using test performance. The
> historical analysis below is preserved as decision provenance, not as the
> current recommendation.

Date: 2026-08-21
Mode: read-only scientific audit. No job was submitted; no checkpoint, config,
retention wrapper, or run artifact was changed.

## Executive conclusion

The original COMPAYL KLT evaluations used `checkpoint_10.pth`, not
`model_best.pth`. This is established independently by both the canonical
release configurations (`eval_checkpoint: latest_checkpoint.pth`) and the
completed runs' `inference.log` files, which resolve that alias to
`checkpoint_10.pth`.

`model_best.pth` is selected by **maximizing class-agnostic validation bPQ**.
It is the repository's validation-selected model in the conventional machine-
learning sense. The selection uses no test data. Exact ties count as a new best,
so the saved model is the **last** epoch attaining the maximum full-precision
validation bPQ.

For the revised slide-independent experiment, the methodological recommendation
is **B, validation-selected `model_best.pth`**, declared before any further
training or test evaluation. This recommendation is based on validation-only
selection and fairness to methods with different optimization dynamics, not on
test performance. It is a deliberate change from the original KLT paper's
fixed-final checkpoint policy and must be disclosed as such.

At present, 6 of 10 completed trainable slide-independent runs still possess
both checkpoints. Four more completed runs would require retraining to recover
their earlier validation-best state. Frozen is not a trained model and therefore
has no `model_best.pth`/`checkpoint_10.pth` choice.

## 1. Exact implementation

### Validation metric and direction

The CellViT validation loop collects binary/class-agnostic PQ for every
validation patch and returns `np.nanmean(pq_scores)` as the early-stopping
scalar:

- `cell_segmentation/trainer/trainer_cellvit.py:533-587`
- the value is logged as `bPQ-Score` and as `bPQ/Validation`;
- it is not validation loss, class-aware mPQ, F1 type, or a test metric.

`ExperimentCellViTPanNuke` constructs `EarlyStopping` with
`strategy="maximize"` (`cell_segmentation/experiments/experiment_cellvit_pannuke.py:203-210`).
Consequently, larger validation bPQ is better.

`EarlyStopping.__call__` uses `best_metric <= metric` for the maximize branch
(`base_ml/base_early_stopping.py:71-78`). Therefore:

1. epoch 1 is initially best;
2. a strictly larger score replaces it;
3. an exactly equal full-precision score also replaces it;
4. the last epoch tied for the maximum is retained.

The earlier checkpoint audit's wording "strict improvement" was therefore
imprecise. The completed-run best epochs in this report were reconstructed from
the actual `New best model - save checkpoint` events, which preserve the true
decision even though `logs.log` prints bPQ to only four decimal places.

The trainer writes `model_best.pth` immediately after such a selection event
(`base_ml/base_trainer.py:253-265`). The checkpoint stores the zero-based
`epoch`, `best_metric`, and `best_epoch` together with model, optimizer,
scheduler, AMP scaler, and configuration state (`base_ml/base_trainer.py:387-425`).
Epoch numbers in this report are converted to the scientific one-based form.

Independently, the trainer writes numeric epoch checkpoints such as
`checkpoint_10.pth` according to the checkpointing configuration
(`base_ml/base_trainer.py:271-280`). With ten epochs, validation every epoch,
and patience 10, the canonical completed KLT runs reach epoch 10; patience 10
cannot stop a ten-epoch run before its final epoch under this loop.

### Inference resolution

The inference code treats `latest_checkpoint.pth` as an alias: it enumerates
`checkpoint_*.pth`, parses the numeric suffix, and loads the largest epoch
(`cell_segmentation/inference/inference_cellvit_experiment_pannuke.py:194-203`
and `:317-331`). It does not resolve this name to `model_best.pth`.

Thus, for a normal completed ten-epoch run:

- `eval_checkpoint: latest_checkpoint.pth` means `checkpoint_10.pth`;
- `eval_checkpoint: model_best.pth` means the validation-bPQ-selected state.

## 2. Did the criterion or policy change?

No change was found in the **criterion that creates `model_best.pth`** for the
relevant KLT runs:

- all inspected canonical within-slide and slide-independent logs explicitly
  say `Using early stopping with a range of 10 and maximize strategy`;
- all log validation records expose bPQ, and every `New best` event is
  consistent with maximizing it;
- Git history traces both the validation return of mean bPQ and the maximize
  strategy back to the initial implementation (`a601338`); no later criterion
  change was found.

There is, however, repository-level heterogeneity in the **checkpoint chosen
for inference**. The canonical KLT configs use `latest_checkpoint.pth`, while a
small number of later tissue-specific release configs use `model_best.pth`.
That is an evaluation-policy difference, not a change to how `model_best.pth`
is selected.

## 3. Original within-slide COMPAYL KLT policy

All five canonical release KLT YAMLs under
`configs/release/compayl2026/klt/` specify:

```yaml
training:
  epochs: 10
  early_stopping_patience: 10
eval_checkpoint: latest_checkpoint.pth
```

The corresponding LP seed 42, Selected PEFT seeds 42/43, and FullFT seeds
42/43 `inference.log` files all record an actual load from
`checkpoints/checkpoint_10.pth`. The canonical within-slide NT-header1 seed-42
run does the same. Therefore the submitted within-slide KLT results used a
**fixed final-epoch checkpoint**.

This distinction is important:

- `model_best.pth` means the state selected solely by maximum validation bPQ;
- `checkpoint_10.pth` means the state after the fixed tenth optimization epoch,
  whether or not validation bPQ had already peaked.

The conventional validation-selected model in this codebase is
`model_best.pth`; the original KLT paper nevertheless chose the fixed-final
checkpoint through its configs.

## 4. Completed slide-independent inventory

Completion here requires ten logged validation epochs and completed final
inference. Scores are the four-decimal values in `logs.log`; selection decisions
come from the unrounded-value `New best` events. All trainable rows use
validation bPQ, maximize, last tie wins.

| Fold | Method | Seed | Best validation epoch | `model_best.pth` | `checkpoint_10.pth` | Checkpoint used for inference | Selection metric | Best validation bPQ | Epoch-10 validation bPQ |
|---|---|---:|---:|:---:|:---:|---|---|---:|---:|
| A | Frozen | 42 | N/A | No | No | official untouched `CellViT-SAM-H-x40.pth` | N/A; no training | N/A | N/A |
| A | LP | 42 | 10 | Yes | Yes | `checkpoint_10.pth` | validation bPQ, maximize | 0.5020 | 0.5020 |
| A | LP | 43 | 7 | No | Yes | `checkpoint_10.pth` | validation bPQ, maximize | 0.4989 | 0.4983 |
| A | Selected PEFT | 42 | 10 | Yes | Yes | `checkpoint_10.pth` | validation bPQ, maximize | 0.5463 | 0.5463 |
| A | NT-header1 PEFT | 42 | 7 | Yes | Yes | `checkpoint_10.pth` | validation bPQ, maximize | 0.5462 | 0.5442 |
| A | FullFT | 42 | 8 | Yes | Yes | `checkpoint_10.pth` | validation bPQ, maximize | 0.5622 | 0.5619 |
| B | Frozen | 42 | N/A | No | No | official untouched `CellViT-SAM-H-x40.pth` | N/A; no training | N/A | N/A |
| B | LP | 42 | 4 | Yes | Yes | `checkpoint_10.pth` | validation bPQ, maximize | 0.4142 | 0.4081 |
| B | LP | 43 | 4 | No | Yes | `checkpoint_10.pth` | validation bPQ, maximize | 0.4137 | 0.4088 |
| B | Selected PEFT | 42 | 9 | No | Yes | `checkpoint_10.pth` | validation bPQ, maximize | 0.4784 | 0.4751 |
| B | NT-header1 PEFT | 42 | 9 | No | Yes | `checkpoint_10.pth` | validation bPQ, maximize | 0.4825 | 0.4696 |
| B | FullFT | 42 | 9 | Yes | Yes | `checkpoint_10.pth` | validation bPQ, maximize | 0.5003 | 0.4845 |

Every trainable row's `inference.log` explicitly names
`checkpoint_10.pth`; this was not inferred merely from the YAML.

The two Frozen run directories contain a symbolic link named
`frozen_pretrained.pth` to the same official pretrained checkpoint. Frozen has
no optimizer, validation selection, or trainable epoch state and must stay out
of the checkpoint-policy comparison.

### Runs retaining both checkpoints

There are **6** completed trainable runs where both filenames still exist:

1. Fold A / LP / seed 42;
2. Fold A / Selected PEFT / seed 42;
3. Fold A / NT-header1 PEFT / seed 42;
4. Fold A / FullFT / seed 42;
5. Fold B / LP / seed 42;
6. Fold B / FullFT / seed 42.

A no-retraining sensitivity analysis is technically possible for these six,
but no test inference was launched during this audit. For Fold A LP and Fold A
Selected PEFT, `model_best.pth` and `checkpoint_10.pth` are hard links to the
same inode because epoch 10 was best; their predictions must be identical. The
other four pairs are distinct states and can provide an informative checkpoint
sensitivity analysis.

This would be a **partial**, not fully balanced, sensitivity study: the four
completed runs listed below no longer retain their earlier best state.

### Runs requiring retraining under a `model_best` policy

Exactly **4 completed trainable runs** require retraining if
`model_best.pth` becomes mandatory for primary evaluation:

| Fold | Method | Seed | Lost best epoch | Reason retraining is required |
|---|---|---:|---:|---|
| A | LP | 43 | 7 | retention metadata records removal of `model_best.pth`; only epoch 10 remains |
| B | LP | 43 | 4 | retention metadata records removal of `model_best.pth`; only epoch 10 remains |
| B | Selected PEFT | 42 | 9 | retention metadata records removal of `model_best.pth`; only epoch 10 remains |
| B | NT-header1 PEFT | 42 | 9 | retention metadata records removal of `model_best.pth`; only epoch 10 remains |

Their `checkpoint_retention_metadata.json` files preserve best epoch, best
score, epoch-10 score, and deletion provenance, but not the earlier model
weights. Intermediate numeric checkpoints are also absent. Metadata cannot
reconstruct weights, so copying or relabeling `checkpoint_10.pth` would be
scientifically invalid.

## 5. Historical epoch-10 versus validation-best audit

To avoid an ambiguous denominator, results are reported at three levels.

### Original/canonical within-slide runs only

Across the six comparable within-slide LP, Selected PEFT, NT-header1, and
FullFT runs, epoch 10 was validation-best in **2/6 (33.33%)**.

| Method | Seed(s) | Best epoch(s) | Epoch 10 best |
|---|---|---|---:|
| LP | 42 | 3 | 0/1 |
| Selected PEFT | 42, 43 | 10, 9 | 1/2 |
| NT-header1 PEFT | 42 | 8 | 0/1 |
| FullFT | 42, 43 | 7, 10 | 1/2 |

### Completed slide-independent runs only

Across the ten completed trainable slide-independent runs, epoch 10 was
validation-best in **2/10 (20.00%)**. The two cases were Fold A LP seed 42 and
Fold A Selected PEFT seed 42.

### Combined comparable KLT evidence

Across all 16 completed comparable KLT runs from both protocols, epoch 10 was
validation-best in **4/16 (25.00%)**.

| Method | Runs | Best-epoch distribution | Epoch 10 best | Percentage |
|---|---:|---|---:|---:|
| LP | 5 | 3, 10, 7, 4, 4 | 1/5 | 20.00% |
| Selected PEFT | 4 | 10, 9, 10, 9 | 2/4 | 50.00% |
| NT-header1 PEFT | 3 | 8, 7, 9 | 0/3 | 0.00% |
| FullFT | 4 | 7, 10, 8, 9 | 1/4 | 25.00% |

The descriptive pattern is that Selected PEFT tends to peak latest, while
NT-header1 peaked before epoch 10 in every available run. LP peaked early in
four of five runs; FullFT peaked early in three of four. Sample sizes are small
and protocols are mixed, so these are optimization tendencies, not claims of
method superiority. They do show that a fixed tenth epoch is not neutral with
respect to method-specific learning dynamics.

## 6. Methodological comparison

| Criterion | A: primary `checkpoint_10.pth` | B: primary `model_best.pth` |
|---|---|---|
| Independence from test data | Yes, if epoch 10 is fixed before testing | Yes; selection uses only training-slide validation bPQ |
| Original-paper consistency | Exact match to original KLT inference policy | Deliberate protocol change that must be disclosed |
| Fairness across optimization dynamics | Weaker: methods can be compared after different amounts of post-peak degradation | Stronger: every method receives the same validation-only selection rule |
| Reproducibility | Very simple: fixed epoch and numeric filename | Strong if metric, direction, tie rule, seed, and selected epoch are recorded |
| Current retraining cost | None for completed runs | Four completed runs must be retrained; six retained pairs need no retraining |
| Relation to repository convention | Matches many existing KLT configs | Is the semantically conventional validation-selected model |

Both policies are independent of test data when fixed in advance. Therefore
test performance must not be consulted to choose between them.

### Recommendation

Use **B: `model_best.pth` selected by maximum validation bPQ** as the primary
checkpoint for the revised slide-independent campaign, for all trainable
methods, folds, and seeds.

Reasons independent of test performance:

1. it is a predeclared, test-independent selection rule;
2. it is the conventional purpose of `model_best.pth` in this repository;
3. only 25% of the comparable KLT runs ended with epoch 10 as validation-best;
4. the early-peak frequency differs materially by method, so fixed epoch 10
   can penalize optimization dynamics unevenly;
5. validation selection makes checkpoint choice an explicit scientific rule
   rather than an incidental consequence of the epoch budget.

The cost is real: four completed runs require retraining, and the revised paper
must state that the slide-independent protocol uses validation-selected bPQ
while the original within-slide COMPAYL results used epoch 10. For direct
apples-to-apples comparison with the original paper, retain the existing
epoch-10 within-slide numbers and label the checkpoint-policy difference, or
report an epoch-10 slide-independent sensitivity analysis as secondary. Do not
choose, tune, or switch checkpoints using held-out test results.

Before any further slide-independent training, the chosen policy should be
frozen in configs and provenance. This audit does not make that change.

## 7. Recommended one-checkpoint storage policy

Under the recommended validation-selected policy, the persistent final state
should contain exactly one full checkpoint per completed trainable run:

```text
KEEP:   model_best.pth
REMOVE: checkpoint_10.pth, but only after successful model_best inference
```

No deletion should occur until all of the following have been verified and
preserved atomically in machine-readable metadata:

- successful final test inference explicitly loaded `model_best.pth`;
- checkpoint path, byte size, and SHA-256;
- selection metric = validation bPQ and direction = maximize;
- one-based best validation epoch and full-precision best score;
- epoch-10 validation score and whether best epoch equals 10;
- exact tie rule (last maximum), seed, fold, method, run/config identity;
- inference-results path and completion status.

When best epoch is 10 and the two names are hard links to one inode, there is
already only one physical payload, but the redundant filename should still be
removed after metadata and inference verification to make the policy
unambiguous.

Do not implement this retention policy until the primary-checkpoint decision is
accepted and no active run can be affected. In particular, do not leave the
current checkpoint-10-only post-run wrapper active for new runs if
`model_best.pth` is adopted.

## Final requested facts

- **Original paper checkpoint policy:** fixed final epoch,
  `latest_checkpoint.pth` resolved to `checkpoint_10.pth`.
- **`model_best` selection metric:** mean class-agnostic validation bPQ,
  maximized; last exact tie wins.
- **Completed slide-independent runs still possessing `model_best`:** 6 of 10
  trainable runs (plus Frozen is not applicable).
- **Runs requiring retraining under a `model_best` policy:** 4 — Fold A LP
  seed 43; Fold B LP seed 43; Fold B Selected PEFT seed 42; Fold B NT-header1
  PEFT seed 42.
- **Recommended primary checkpoint:** `model_best.pth`.
- **Basis:** validation-only selection, conventional semantics, and fairer
  handling of different optimization dynamics; no test metric was inspected or
  used to make this recommendation.
