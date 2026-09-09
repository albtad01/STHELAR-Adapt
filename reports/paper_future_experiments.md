# Future experiments and analyses for the STHELAR-Adapt paper

Status update: 2026-08-29 17:07 CEST. The SAM-H seed-44, CellViT-256
seed-43, and Kidney/Liver reciprocal Fold-B conditions listed in the earlier
plan are now scientifically complete. CellViT-256 seed-44 PEFT/FullFT and
CellViT-256 Kidney/Liver/Tonsil specialist conditions were submitted as the
next preregistered extension. Infrastructure-only `gpu11` attempts were
superseded by the canonical active/pending job IDs recorded in
`reports/cellvit256_slide_exp_seed44_jobs.md` and
`reports/cellvit256_tissue_specific_klt_seed42_jobs.md`; they are not
results until canonical epoch-10 inference completes. SAM-H LP seed44 A/B
remains `READY_NOT_SUBMITTED`. This file remains a planning record.

## Priority A — highest value for the main claims

### Third SAM-H stochastic seed

The main held-out-slide Selected-PEFT/FullFT comparison now has completed
seeds 42, 43 and 44 in each fold. Report mean and sample SD separately inside
Fold A and Fold B; never pool the two folds and three seeds into an `n=6`
replicate sample.

LP seed 44 Fold A/B is useful but secondary. It would make the optimization-
variance reporting symmetric across all trainable SAM-H baselines, but it adds
less value than completing Selected PEFT and FullFT first. Frozen has no
meaningful training-seed replication. NT-header1 already has seeds 42/43 and is
a supporting typing ablation; do not prioritize a seed-44 NT run over the
primary methods.

## Priority B — backbone-scale robustness

CellViT-256 Selected PEFT and FullFT seeds 42/43 are complete. Seed-44 Fold
A/B jobs 1558251--1558254 are submitted and currently running. Do not
duplicate them. After canonical completion, the remaining lightweight
replication question is whether extra LP seeds are worth the lower marginal
value.

Extra CellViT-256 LP seeds rank below these. The scientifically relevant
question is whether the lightweight PEFT-versus-FullFT trade-off and the
comparison with SAM-H PEFT persist across optimization seeds. These runs are
useful for a longer paper but are not mandatory for a deadline-constrained
revision.

## Priority C — tissue-specific reciprocal validation

The primary nine-tissue one-direction Fold-A study is complete. Reciprocal
Fold-B evidence is complete for Tonsil, Breast, Pancreatic, Skin, Ovary, Colon
and Lung. Do not duplicate these completed conditions.

Kidney and Liver Fold B are now complete, so reciprocal SAM-H specialist
coverage exists for all nine tissues. CellViT-256 specialist Kidney, Liver and
Tonsil Fold A/B jobs 1558255--1558260 are submitted to form the corresponding
backbone-scale generalist-versus-specialist comparison. Do not duplicate these
conditions and do not propose three seeds times nine tissues as a near-term
requirement.

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
| 1 | CellViT-256 Selected PEFT and FullFT seed 44 A/B | Completes lightweight three-seed robustness | Submitted: 1558251--1558254 |
| 2 | CellViT-256 K/L/T tissue specialists A/B | Backbone-scale generalist/specialist comparison | Submitted: 1558255--1558260 |
| 3 | SAM-H LP seed 44 A/B | Symmetric LP seed variance | READY_NOT_SUBMITTED; secondary |
| 4 | HoVer-Net held-out-slide baseline | Addresses conventional-baseline criticism | High engineering cost |
