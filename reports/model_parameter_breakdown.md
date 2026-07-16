# CellViT-SAM-H x40 Parameter Breakdown for STHELAR PEFT

Generated on 2026-07-04 from the local STHELAR-Adapt codebase.

## Decoder structure

In the non-shared `CellViTSAM` model used by these configs, the encoder produces multi-scale features/tokens that are consumed by a decoder with:

- shared skip/decoder helper blocks:
  - `decoder0`
  - `decoder1`
  - `decoder2`
  - `decoder3`
- three task-specific upsampling branches:
  - `nuclei_binary_map_decoder.*` for NP / nuclei binary map
  - `hv_map_decoder.*` for horizontal-vertical regression
  - `nuclei_type_maps_decoder.*` for nuclei type prediction

Each task-specific branch is built by `create_upsampling_branch(num_classes)` and contains:

- `bottleneck_upsampler`
- `decoder3_upsampler`
- `decoder2_upsampler`
- `decoder1_upsampler`
- `decoder0_header`

The final prediction head is the last `1x1` convolution inside:

```text
<branch>.decoder0_header.2.*
```

So `heads_only` means only these final `1x1` heads are trainable, while `nuclei_type_maps_decoder.*` means the full NT branch is trainable, including all its upsampling blocks plus its final head.

## Why `NP/HV heads + NT all` is about 3.25%

For the KLT config:

```text
training_sthelar40x_kidney_liver_tonsil_5class_spatial_margin128_lora_adaptformer_r8_a8_red16_decoder_np_hv_heads_nt_all_lr5e-5_e10_seed42_CLEAN.yaml
```

the instantiated model has:

```text
total parameters:     707,644,395
trainable parameters: 22,986,219
trainable ratio:      3.248273%
```

The trainable part is:

| Component | Total params | Trainable params | Notes |
|---|---:|---:|---|
| NT decoder body | 15,077,440 | 15,077,440 | full `nuclei_type_maps_decoder`, except counted final head separately below |
| AdaptFormer MLP adapters | 6,597,152 | 6,597,152 | encoder MLP adapters |
| LoRA q/v | 1,310,720 | 1,310,720 | q/v attention adapters |
| NT final head | 390 | 390 | `nuclei_type_maps_decoder.decoder0_header.2.*` |
| HV final head | 130 | 130 | `hv_map_decoder.decoder0_header.2.*` |
| NP final head | 130 | 130 | `nuclei_binary_map_decoder.decoder0_header.2.*` |
| Classifier head | 257 | 257 | tissue classifier head |

Sum:

```text
15,077,440
+ 6,597,152
+ 1,310,720
+ 390
+ 130
+ 130
+ 257
= 22,986,219 trainable parameters
```

That is why the ratio is not just `decoder_params / 3 + LoRA/AF` in a vague sense; it is specifically:

```text
one full branch body, NT
+ tiny final heads for NP/HV/NT
+ LoRA q/v
+ AdaptFormer
+ classifier head
```

The three branch bodies are symmetric in parameter count:

| Component | Total params | Trainable in `np_hv_heads_nt_all` |
|---|---:|---:|
| NP decoder body | 15,077,440 | no |
| HV decoder body | 15,077,440 | no |
| NT decoder body | 15,077,440 | yes |

The full model is dominated by the SAM-H encoder base:

| Component | Total params | Trainable in `np_hv_heads_nt_all` |
|---|---:|---:|
| Encoder base | 637,026,048 | no |
| AdaptFormer MLP adapters | 6,597,152 | yes |
| LoRA q/v | 1,310,720 | yes |
| NP decoder body | 15,077,440 | no |
| HV decoder body | 15,077,440 | no |
| NT decoder body | 15,077,440 | yes |
| Shared `decoder0` | 19,584 | no |
| Final heads + classifier | 907 | yes |
| Other non-encoder parameters | 17,457,664 | no |

## Relevant trainability rules

The `decoder_train_scope: np_hv_heads_nt_all` rule is:

```python
train = (
    name.startswith("nuclei_type_maps_decoder")
    or name.startswith("nuclei_binary_map_decoder.decoder0_header.2.")
    or name.startswith("hv_map_decoder.decoder0_header.2.")
    or is_classifier_parameter(name)
)
```

Together with `adapter_type: lora_adaptformer`, this means:

- LoRA q/v adapters are trainable.
- AdaptFormer encoder MLP adapters are trainable.
- Encoder base is frozen.
- Full NT branch is trainable.
- NP/HV branches are frozen except their final `1x1` heads.
- Shared decoder helpers are frozen.

## Interpretation for method selection

`LoRA+AF+final_heads` remains the clean deployment-oriented adapter choice because it keeps the decoder body frozen and uses about 1.12% trainable parameters.

`LoRA+AF+last_stage` is a stronger classification-sensitive PEFT candidate because it improves type-F1 in current KLT results with only about 1.17% trainable parameters.

`LoRA+AF+NP/HV heads + NT all` is an exploratory classification-focused probe. It is still far smaller than FullFT, but at about 3.25% trainable parameters it is no longer as lightweight as the heads-only strategy. It is useful mainly to test whether extra capacity should be placed specifically in the nuclei-type branch.

