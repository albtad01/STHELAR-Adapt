# CellViT-SAM-H x40 Parameter and Adapter Trainability Breakdown

Generated on 2026-07-04 for the KLT 5-class spatial margin128 benchmark in STHELAR-Adapt.

This note explains where the model parameters are, what is trainable under each PEFT strategy, and how the reported trainable percentages in the KLT ablation table arise.

## Short Answer on Additional Convolutional Heads

Yes, it makes sense to test adapting additional convolutional blocks in the decoder, but we should distinguish three different ideas:

1. **Final heads only**

   Train only the last `1x1` convolution in each output branch:

   ```text
   nuclei_binary_map_decoder.decoder0_header.2.*
   hv_map_decoder.decoder0_header.2.*
   nuclei_type_maps_decoder.decoder0_header.2.*
   ```

   This is extremely small: only `650` parameters for NP/HV/NT heads.

2. **Last-stage decoder tuning**

   Train the final convolutional stage before prediction, i.e. the full `decoder0_header.*` for each branch, plus shared `decoder0.*`.

   This is what `decoder_train_scope: last_stage` already does. It adds about `352,512` trainable decoder parameters beyond final heads and classifier.

3. **Decoder convolution adapters**

   Keep original decoder convolutions frozen, but wrap selected stride-1 decoder convolutions with small trainable bottleneck residual adapters:

   ```text
   y = frozen_conv(x) + alpha * up(act(down(x)))
   ```

   This is `decoder_train_scope: conv_adapters`. It adds `552,095` trainable decoder-adapter parameters, plus final heads and classifier.

So yes: adapting more convolutional head capacity is a sensible ablation. But we already tested two versions of that idea:

- `last_stage`: directly trains the late original convolutional decoder/head blocks.
- `conv_adapters`: inserts lightweight convolutional residual adapters around frozen decoder convolutions.

Given current results, I would keep **LoRA+AdaptFormer final heads** as the clean deployment-oriented default. The extra convolutional strategies are useful for understanding whether type-F1 can be improved by spending decoder capacity.

## CellViT Decoder Structure

For the non-shared `CellViTSAM` model used here, the decoder has:

- shared skip/helper decoder modules:
  - `decoder0`
  - `decoder1`
  - `decoder2`
  - `decoder3`
- three task-specific upsampling branches:
  - `nuclei_binary_map_decoder.*` for NP / binary nuclei map
  - `hv_map_decoder.*` for HV regression
  - `nuclei_type_maps_decoder.*` for nuclei type classification

Each task branch is created by `create_upsampling_branch(num_classes)` and contains:

```text
bottleneck_upsampler
decoder3_upsampler
decoder2_upsampler
decoder1_upsampler
decoder0_header
```

The final prediction head is the last `1x1` convolution:

```text
<branch>.decoder0_header.2.*
```

## Base Model Parameter Decomposition

The base KLT CellViT-SAM-H x40 model without LoRA/AdaptFormer/VeRA/decoder adapters has:

```text
699,736,523 parameters
```

| Component | Parameters | Share of base model | Notes |
|---|---:|---:|---|
| Encoder base | 637,026,048 | 91.04% | SAM-H encoder weights |
| Shared decoder / non-branch helpers | 17,457,664 | 2.50% | mostly shared decoder helpers such as `decoder1/2/3` and related non-branch weights |
| NP decoder body | 15,077,440 | 2.15% | `nuclei_binary_map_decoder`, excluding final head |
| HV decoder body | 15,077,440 | 2.15% | `hv_map_decoder`, excluding final head |
| NT decoder body | 15,077,440 | 2.15% | `nuclei_type_maps_decoder`, excluding final head |
| Shared `decoder0` | 19,584 | 0.003% | late shared skip/helper block |
| NP/HV/NT final heads | 650 | <0.001% | final `1x1` conv heads |
| Classifier head | 257 | <0.001% | tissue classifier head |
| **Total** | **699,736,523** | **100%** |  |

The important point is that the SAM-H encoder dominates the parameter count. Each task-specific decoder branch body is about `15.08M` parameters, while the final `1x1` heads are tiny.

## Adapter Parameter Blocks

These are the reusable trainable parameter blocks that appear in the ablations:

| Block | Parameters | Meaning |
|---|---:|---|
| Final NP/HV/NT heads | 650 | last `1x1` convs for NP, HV, NT |
| Classifier head | 257 | tissue classifier head |
| LoRA q/v r8 | 1,310,720 | q/v LoRA adapters in encoder attention |
| VeRA q/v r16 | 82,976 | q/v VeRA scaling parameters and trainable alpha |
| AdaptFormer red16 | 6,597,152 | encoder MLP AdaptFormer modules |
| Decoder conv adapters | 552,095 | residual bottleneck adapters around decoder convs |
| Last-stage decoder original params | 352,512 | shared `decoder0` plus branch `decoder0_header` conv blocks, excluding final heads |
| Full NT decoder body | 15,077,440 | entire NT branch body, excluding final head |

## KLT Ablation Trainability Table

Metrics are the current KLT ablation values supplied in the paper table. Trainable counts are computed from the instantiated configs or from the exact trainability rule used by the training code.

| Encoder | Decoder scope | Total params | Trainable params | Train. % | Trainable decomposition | mPQ | bPQ | F1_det | F1_type |
|---|---|---:|---:|---:|---|---:|---:|---:|---:|
| Frozen | none | 699,736,523 | 0 | 0.0000% | none | 0.003 | 0.447 | 0.817 | 0.012 |
| Frozen | final heads | 699,736,523 | 650 | 0.0001% | final NP/HV/NT heads 650 | 0.204 | 0.446 | 0.818 | 0.430 |
| LoRA | frozen | 701,047,243 | 1,310,720 | 0.1870% | LoRA q/v 1,310,720 | 0.113 | 0.430 | 0.831 | 0.224 |
| AdaptFormer | frozen | 706,333,675 | 6,597,152 | 0.9340% | AdaptFormer 6,597,152 | 0.119 | 0.418 | 0.806 | 0.237 |
| VeRA | frozen | 699,819,499 | 82,976 | 0.0119% | VeRA q/v 82,976 | 0.065 | 0.457 | 0.827 | 0.145 |
| Frozen | conv. adapt. | 700,288,618 | 553,002 | 0.0790% | decoder conv adapters 552,095 + final heads 650 + classifier 257 | 0.232 | 0.459 | 0.828 | 0.562 |
| LoRA+AdaptFormer | heads | 707,644,395 | 7,908,779 | 1.1176% | AdaptFormer 6,597,152 + LoRA 1,310,720 + final heads 650 + classifier 257 | 0.294 | 0.504 | 0.835 | 0.578 |
| LoRA+AdaptFormer | last stage | 707,644,395 | 8,261,291 | 1.1674% | AdaptFormer 6,597,152 + LoRA 1,310,720 + last-stage decoder 352,512 + final heads 650 + classifier 257 | 0.296 | 0.503 | 0.826 | 0.654 |
| LoRA+AdaptFormer | conv. adapt. | 708,196,490 | 8,460,874 | 1.1947% | AdaptFormer 6,597,152 + LoRA 1,310,720 + decoder conv adapters 552,095 + final heads 650 + classifier 257 | 0.288 | 0.493 | 0.829 | 0.651 |
| VeRA+AdaptFormer | heads | 706,416,651 | 6,681,035 | 0.9458% | AdaptFormer 6,597,152 + VeRA 82,976 + final heads 650 + classifier 257 | 0.287 | 0.493 | 0.833 | 0.579 |
| VeRA+AdaptFormer | conv. adapt. | 706,968,746 | 7,233,130 | 1.0231% | AdaptFormer 6,597,152 + VeRA 82,976 + decoder conv adapters 552,095 + final heads 650 + classifier 257 | 0.285 | 0.492 | 0.829 | 0.652 |

## Why Some Percentages Are Slightly Different from Intuition

The denominator changes when adapters are inserted. For example:

- base frozen model: `699,736,523` parameters
- LoRA model: `699,736,523 + 1,310,720 = 701,047,243`
- AdaptFormer model: `699,736,523 + 6,597,152 = 706,333,675`
- LoRA+AdaptFormer model: `699,736,523 + 1,310,720 + 6,597,152 = 707,644,395`
- LoRA+AdaptFormer+conv-adapter model: `707,644,395 + 552,095 = 708,196,490`

So the trainable percentage is always:

```text
trainable parameters / total parameters after inserting adapters
```

not divided by the original frozen model size.

## The `NP/HV heads + NT all` Probe

The exploratory `np_hv_heads_nt_all` run trains:

- LoRA q/v
- AdaptFormer
- full NT decoder branch
- NP/HV final heads only
- NT final head
- classifier head

Its trainable count is:

```text
NT decoder body        15,077,440
AdaptFormer             6,597,152
LoRA q/v                1,310,720
NT final head                 390
HV final head                 130
NP final head                 130
Classifier head               257
---------------------------------
Trainable             22,986,219
Total model          707,644,395
Trainable ratio           3.2483%
```

This validates the intuition that it is roughly:

```text
one full task-specific branch
+ LoRA
+ AdaptFormer
+ tiny final heads/classifier
```

It is still much smaller than FullFT, but much less lightweight than the selected `LoRA+AdaptFormer heads` setting.

## Current Interpretation

For the paper/default method, **LoRA+AdaptFormer heads** remains the cleanest choice:

- encoder base frozen
- decoder body frozen
- only encoder adapters plus final prediction heads are trainable
- strong mPQ and detection metrics
- compact adapter export story

The stronger type-F1 observed in `last_stage` and `conv_adapters` suggests that type assignment benefits from a little decoder-side capacity. But these variants are better framed as analysis or classification-sensitive variants unless they clearly outperform the heads-only method on the final metrics.

