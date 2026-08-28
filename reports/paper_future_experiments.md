# Future experiments and analyses for the STHELAR-Adapt paper

Status: 2026-08-28. This is a planning document only. No job was launched and
no listed item should be interpreted as an existing result. The primary method
remains the pre-selected CellViT-SAM-H x40 LoRA(Q,V) rank-8/alpha-8 plus
AdaptFormer reduction-16 GELU plus final NP/HV/NT heads configuration.

## Priority A — highest value for the main claims

### Third SAM-H stochastic seed

The main held-out-slide comparison currently has seeds 42 and 43. The most
valuable additional runs are:

1. Selected PEFT seed 44, Fold A.
2. Selected PEFT seed 44, Fold B.
3. FullFT seed 44, Fold A.
4. FullFT seed 44, Fold B.

These four runs directly strengthen the paper's core PEFT-versus-FullFT claim
under slide shift. A third seed improves robustness and variance estimation; it
is not a magical threshold for statistical validity. If completed, report mean
and sample SD across seeds separately inside Fold A and Fold B. Never pool the
two folds and three seeds into an `n=6` replicate sample.

LP seed 44 Fold A/B is useful but secondary. It would make the optimization-
variance reporting symmetric across all trainable SAM-H baselines, but it adds
less value than completing Selected PEFT and FullFT first. Frozen has no
meaningful training-seed replication. NT-header1 already has seeds 42/43 and is
a supporting typing ablation; do not prioritize a seed-44 NT run over the
primary methods.

## Priority B — backbone-scale robustness

CellViT-256 currently has only seed 42. Useful lightweight-backbone
replications, in order, are:

1. CellViT-256 Selected PEFT seed 43, Fold A/B.
2. CellViT-256 FullFT seed 43, Fold A/B.
3. CellViT-256 Selected PEFT seed 44, Fold A/B.
4. CellViT-256 FullFT seed 44, Fold A/B.

Extra CellViT-256 LP seeds rank below these. The scientifically relevant
question is whether the lightweight PEFT-versus-FullFT trade-off and the
comparison with SAM-H PEFT persist across optimization seeds. These runs are
useful for a longer paper but are not mandatory for a deadline-constrained
revision.

## Priority C — tissue-specific reciprocal validation

The primary nine-tissue one-direction Fold-A study is complete. Reciprocal
Fold-B evidence is complete for Tonsil, Breast, Pancreatic, Skin, Ovary, Colon
and Lung. Do not duplicate these completed conditions.

The missing high-value reciprocal conditions are:

1. Kidney Selected PEFT seed 42: train/validate `kidney_s1`, test all
   `kidney_s0`.
2. Liver Selected PEFT seed 42: train/validate `liver_s1`, test all
   `liver_s0`.

Together with completed Tonsil Fold B, these would permit reciprocal,
slide-direction-matched generalist-KLT versus tissue-specialist comparisons for
all three KLT tissues. This is primarily longer-term supporting work. Do not
propose three seeds times nine tissues as a near-term requirement.

### Existing-output analysis still needed

For Tonsil Fold A, compute a composition-matched generalist-versus-specialist
table on the common 9,879 patch identifiers. Both inference JSONs already
exist; the specialist whole-slide result covers 21,083 patches. This is an
analysis of existing predictions, not a new training or inference experiment.
Kidney and Liver are already patch-identical and need only tabulation.

## Priority D — conventional baseline

No scientifically valid held-out-slide STHELAR HoVer-Net baseline exists. The
repository contains HoVer-Net-derived utilities/post-processing, but not a
complete compatible model/checkpoint/training/evaluation integration.

A rigorous HoVer-Net comparison would require selecting a defensible pretrained
checkpoint, reproducing its normalization and output semantics, adapting the
STHELAR label head without inventing a PanNuke mapping, validating post-processing,
and evaluating the same Fold-A/B held-out slides. Its scientific value is high
because it answers the remaining request for a conventional
segmentation architecture; engineering cost is also high. Treat it as desirable
for the longer version and optional for the near-term revision unless implementation can
be completed and validated without compromising the core revision.

## Priority E — optional and longer-term

- CellViT++ on the exact held-out-slide protocol, only after resolving its
  pretraining/taxonomy/evaluator compatibility. This is a modern external anchor,
  not a replacement for the central SAM-H PEFT study.
- A genuinely patient- or specimen-independent extension after obtaining an
  authoritative slide-to-patient/specimen mapping or additional biological
  replicates. This has greater biological value than adding many optimization
  seeds.
- External/brain-domain extension after the main STHELAR claims are stable and
  the domain's biological metadata and label correspondence are explicit.
- Generate CellViT-256 matched-nuclei confusion-matrix tables/plots and the
  final nine-tissue per-class/support/confusion summaries from existing result
  JSONs. These are analyses, not new experiments.
- A narrowly preregistered typing-specific decoder study only if the longer
  paper makes typing transfer a primary mechanistic claim. Do not revive broad
  LoRA/AdaptFormer searches, NT-header2/3 fishing, Fisher/QC as central
  experiments, or arbitrary post-hoc adapter selection.

## Statistical interpretation

- Three seeds quantify optimization variability and improve stochastic
  robustness; they do not create biological independence.
- Reciprocal folds quantify sensitivity to slide/domain composition.
- Biological independence depends on patient/specimen metadata, which is not
  currently mapped to the paired slide IDs in the local dataset documentation.
- Thousands of patches must not be treated as thousands of independent
  biological replicates.
- Patch-level p-values should not be the primary significance analysis.
- Report seed mean and sample SD within each fold, keep folds separate, and
  describe tissue/slide results as biological-unit-level observations.

## Priority summary

| Rank | Future work | Scientific value | Near-term status |
|---:|---|---|---|
| 1 | SAM-H Selected PEFT seed 44 A/B | Direct robustness of the central method | Highest priority |
| 2 | SAM-H FullFT seed 44 A/B | Completes the core PEFT-versus-FullFT three-seed comparison | Highest priority |
| 3 | CellViT-256 Selected PEFT and FullFT seed 43 A/B | Tests backbone-scale robustness of the adaptation trade-off | High value, deadline-dependent |
| 4 | Kidney/Liver tissue-specific reciprocal Fold B | Completes reciprocal specialist/generalist KLT comparison | Longer-term supporting work |
| 5 | HoVer-Net held-out-slide baseline | Addresses conventional-baseline criticism | High engineering cost |
