# Slide-independent KLT computational and deployment efficiency audit

Audit date: 2026-08-25. Scope: completed seed-42 Fold A/B slide-independent KLT test runs only. Every run used an NVIDIA A100-SXM4-40GB, training batch size 4 (Frozen has no training), inference batch size 16, and CUDA FP16 autocast; trained methods used `GradScaler`. Values below are arithmetic means over the two matched folds. No new training was launched, no canonical checkpoint was changed, and no validation metric is interpreted.

Reconciliation note (2026-08-27): the measurements and serialized-size audit
remain complete and internally consistent. Early SAM-H seed-42 efficiency JSONs
predate the explicit `partition` and `backbone` fields; their A100 GPU identity
is recorded in the JSON and `gpua100` is recoverable from the Slurm provenance.
Later CellViT-256 PEFT JSONs contain both fields directly. Thus the matched
hardware claim is verified, but not every historical JSON independently stores
the partition string. Frozen has no training-time or training-batch measurement
by definition.

## Paper-ready efficiency table

Sizes are measured serialized sizes, not parameter-count estimates. `Train ckpt` is the current `checkpoint_10.pth` including optimizer/scheduler/scaler state. `Inference weights` is a separately serialized inference-only model `state_dict`. `Adapter` is the verified flat adapter-only `state_dict`. GiB and MiB use powers of 1024.

| Backbone | Method | Total params | Trainable params (%) | Train wall (h) | Mean epoch (min) | Train CUDA GiB alloc/res | Train ckpt (GiB) | Inference weights (GiB) | Adapter (MiB) | Inference CUDA GiB alloc/res | End-to-end patches/s |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| SAM-H | Frozen | 699,741,149 | 0 (0%) | — | — | — | — | 2.6071 | — | 3.8935 / 4.6406 | 9.555 |
| SAM-H | LP | 699,736,523 | 650 (0.000093%) | 5.339 | 32.036 | 3.8932 / 4.7480 | 2.6071 | 2.6070 | — | 7.8202 / 8.3926 | 9.223 |
| SAM-H | Selected PEFT | 707,644,395 | 7,908,779 (1.1176%) | 7.398 | 44.386 | 9.3618 / 11.0918 | 2.6958 | 2.6366 | 30.369 | 6.8344 / 11.0918 | 8.329 |
| SAM-H | FullFT | 699,736,523 | 699,736,523 (100%) | 7.986 | 47.914 | 15.5077 / 18.4287 | 7.8210 | 2.6070 | — | 7.8823 / 18.4287 | 8.931 |
| CellViT-256 | Frozen | 46,750,349 | 0 (0%) | — | — | — | — | 0.1744 | — | 1.3362 / 1.8672 | 10.248 |
| CellViT-256 | LP | 46,743,419 | 650 (0.001391%) | 4.322 | 25.934 | 1.3362 / 1.8799 | 0.1744 | 0.1743 | — | 1.6106 / 1.9141 | 9.985 |
| CellViT-256 | Selected PEFT | 47,116,967 | 374,198 (0.7942%) | 6.044 | 36.261 | 2.0403 / 3.3047 | 0.1787 | 0.1758 | 1.548 | 1.5811 / 3.3047 | 9.388 |
| CellViT-256 | FullFT | 46,743,419 | 46,743,419 (100%) | 5.577 | 33.461 | 2.6674 / 4.0293 | 0.5229 | 0.1743 | — | 1.6178 / 4.0293 | 8.448 |

Frozen uses the untouched pretrained taxonomy and therefore has class-agnostic test metrics only; mPQ is intentionally not reported for Frozen. The Frozen base checkpoint is an inference source, not a training checkpoint.

The inference reserved-memory values for trained methods inherit the CUDA allocator cache present when inference followed training in the same process. They are retained as the recorded peak-reserved measurements, but inference peak **allocated** VRAM is the cleaner cross-method deployment measure.

## Adapter-only artifact verification

Each artifact is a flat PyTorch `state_dict`, with no optimizer, scheduler, scaler, epoch metadata, or copied base parameters. A fresh model was reconstructed from the shared pretrained base, the adapter-only state was overlaid, and every resulting state tensor was compared with the corresponding canonical checkpoint. A deterministic 256×256 forward was then executed before and after canonical-state loading. All NP, HV, NT, and tissue-logit tensors were bit-exact (maximum absolute difference 0) for both folds and both backbones.

| Backbone | Fold | Serialized bytes | Trainable elements | Trainable components | Required mutable buffers | Verification |
|---|---|---:|---:|---|---:|---|
| SAM-H | A | 31,844,024 | 7,908,779 | LoRA 1,310,720; AdaptFormer 6,597,152; NP/HV/NT heads 650; tissue classifier 257 | 105 BN buffers / 19,171 elements | exact state + exact forward |
| SAM-H | B | 31,844,024 | 7,908,779 | same | 105 BN buffers / 19,171 elements | exact state + exact forward |
| CellViT-256 | A | 1,623,604 | 374,198 | LoRA 147,456; AdaptFormer 226,092; NP/HV/NT heads 650 | 105 BN buffers / 13,891 elements | exact state + exact forward |
| CellViT-256 | B | 1,623,604 | 374,198 | same | 105 BN buffers / 13,891 elements | exact state + exact forward |

Two details are necessary for an honest deployable artifact:

- The canonical SAM-H `heads_only` policy also trained the one-class tissue classifier (257 elements). Omitting its two tensors changes the raw tissue logit, so they are included and disclosed. CellViT-256 did not train its tissue head.
- Although decoder parameters outside the final heads stayed frozen, 105 BatchNorm running-stat/counter buffers changed during training. They are state, not trainable parameters, but omitting them fails exact reconstruction and changes inference. The artifact therefore contains the requested trainable components plus exactly these required mutable buffers; it contains zero frozen-parameter overrides.

## Performance versus VRAM

All performance numbers are matched slide-independent **test** results averaged over Fold A/B. No statistical-significance claim is made.

| Backbone | Method | mPQ | bPQ | Detection F1 | Train peak alloc/res (GiB) | Inference peak alloc (GiB) | Patches/s |
|---|---|---:|---:|---:|---:|---:|---:|
| SAM-H | Frozen | — | 0.5164 | 0.8388 | — | 3.8935 | 9.555 |
| SAM-H | LP | 0.2053 | 0.5114 | 0.8413 | 3.8932 / 4.7480 | 7.8202 | 9.223 |
| SAM-H | Selected PEFT | 0.2237 | 0.5647 | 0.8528 | 9.3618 / 11.0918 | 6.8344 | 8.329 |
| SAM-H | FullFT | 0.2032 | 0.5642 | 0.8469 | 15.5077 / 18.4287 | 7.8823 | 8.931 |
| CellViT-256 | Frozen | — | 0.5180 | 0.8457 | — | 1.3362 | 10.248 |
| CellViT-256 | LP | 0.1865 | 0.4977 | 0.8408 | 1.3362 / 1.8799 | 1.6106 | 9.985 |
| CellViT-256 | Selected PEFT | 0.1928 | 0.5180 | 0.8419 | 2.0403 / 3.3047 | 1.5811 | 9.388 |
| CellViT-256 | FullFT | 0.1986 | 0.5537 | 0.8509 | 2.6674 / 4.0293 | 1.6178 | 8.448 |

## Performance versus deployment storage

Storage uses the measured inference-only shared-base/full-model serialization and verified adapter serialization. Strategy B is `2,799,311,538 + N × 31,844,024` bytes; D is `187,221,926 + N × 1,623,604` bytes. The PEFT strategy is slightly larger for a single domain because it stores one complete base plus an adapter, but it becomes smaller at N≥2.

| Strategy | Model | mPQ / bPQ / F1 | N=1 GiB | N=3 GiB | N=6 GiB | N=9 GiB |
|---|---|---:|---:|---:|---:|---:|
| A | SAM-H FullFT: N full models | 0.2032 / 0.5642 / 0.8469 | 2.607 | 7.821 | 15.642 | 23.463 |
| B | SAM-H PEFT: one base + N adapters | 0.2237 / 0.5647 / 0.8528 | 2.637 | 2.696 | 2.785 | 2.874 |
| C | CellViT-256 FullFT: N full models | 0.1986 / 0.5537 / 0.8509 | 0.174 | 0.523 | 1.046 | 1.569 |
| D | CellViT-256 PEFT: one base + N adapters | 0.1928 / 0.5180 / 0.8419 | 0.176 | 0.179 | 0.183 | 0.188 |

At N=9, SAM-H PEFT uses 22.108 GB fewer bytes than SAM-H FullFT (−87.75%), while CellViT-256 PEFT uses 1.483 GB fewer than CellViT-256 FullFT (−88.02%). CellViT-256 PEFT remains much smaller than SAM-H PEFT: 201,834,362 versus 3,085,907,754 bytes at N=9 (−93.46% when D is compared with B).

## Absolute and relative deltas

The complete metric-by-metric absolute (`lhs − rhs`) and relative (`100 × (lhs/rhs − 1)`) results, including storage at N={1,3,6,9}, are in `performance_cost_deltas.csv`. Selected headline deltas are:

| Comparison (lhs vs rhs) | ΔmPQ (relative) | ΔbPQ (relative) | ΔF1 (relative) | Δtrain time (relative) | Δtrain alloc VRAM (relative) | Δinfer alloc VRAM (relative) | Δpatches/s (relative) |
|---|---:|---:|---:|---:|---:|---:|---:|
| SAM-H PEFT vs SAM-H FullFT | +0.02049 (+10.08%) | +0.00053 (+0.09%) | +0.00592 (+0.70%) | −0.588 h (−7.36%) | −6.146 GiB (−39.63%) | −1.048 GiB (−13.29%) | −0.602 (−6.74%) |
| CellViT-256 PEFT vs CellViT-256 FullFT | −0.00580 (−2.92%) | −0.03566 (−6.44%) | −0.00900 (−1.06%) | +0.467 h (+8.37%) | −0.627 GiB (−23.51%) | −0.037 GiB (−2.27%) | +0.941 (+11.14%) |
| SAM-H PEFT vs CellViT-256 PEFT | +0.03089 (+16.02%) | +0.04670 (+9.01%) | +0.01094 (+1.30%) | +1.354 h (+22.41%) | +7.321 GiB (+358.84%) | +5.253 GiB (+332.27%) | −1.060 (−11.29%) |
| SAM-H PEFT vs CellViT-256 FullFT | +0.02509 (+12.63%) | +0.01103 (+1.99%) | +0.00194 (+0.23%) | +1.821 h (+32.65%) | +6.694 GiB (+250.97%) | +5.217 GiB (+322.45%) | −0.119 (−1.40%) |

These are descriptive differences between the two matched folds, not significance estimates.

## Where parameter efficiency does not become wall-clock efficiency

- **CellViT-256 PEFT is the clearest counterexample:** it trains 99.20% fewer parameters than FullFT but takes 8.37% longer end to end. The adapter branches reduce optimizer state and training VRAM, but add forward/backward operations and do not remove the frozen backbone forward pass.
- **SAM-H PEFT is only modestly faster:** 98.87% fewer trainable parameters yields just 7.36% lower wall time, not a proportional speedup. Its inference is 6.74% slower than FullFT because LoRA and AdaptFormer remain active in the inference graph.
- **LP is also not proportional to its parameter count:** 650 trainable parameters still require 66.86% of SAM-H FullFT wall time and 77.50% of CellViT-256 FullFT wall time because the complete model forward and decoder computation remain.
- Parameter efficiency does translate into lower optimizer/checkpoint storage and training allocated VRAM, and into strong multi-domain storage savings once the base is shared. It should not be presented as synonymous with throughput or wall-clock efficiency.

## Files

- `paper_ready_efficiency.csv`: eight-row paper-ready aggregate table.
- `matched_fold_measurements.csv`: Fold A/B source measurements before averaging.
- `performance_vs_vram.csv`: performance/VRAM/throughput view.
- `performance_vs_deployment_storage.csv`: N-domain strategies and performance.
- `performance_cost_deltas.csv`: full absolute and relative delta table.
- `inference_weight_sizes.json`: measured stripped-model serialization evidence.
- `adapters/`: four adapter-only state dictionaries plus verification metadata and SHA256.
