# PEFT strategy parameter statistics

Source: `reports/runs_summary.csv`, filtered to final comparable inference/test metrics where noted. This document is meant as the running catalog of implemented PEFT strategies, their trainable parameter budgets, and the current KLT multitissue evidence.

## Executive read

For the KLT multitissue 5-class spatial margin128 benchmark, the main PEFT recommendation is **LoRA+AdaptFormer with decoder `heads_only`**. It trains 7,908,779 parameters, or 1.1176% of the CellViT-SAM-H model, and recovers 94.96% of FullFT mPQ and 97.97% of FullFT bPQ across two completed seeds.

`last_stage` is the strongest secondary LoRA+AdaptFormer ablation by mPQ, but its mean gain over `heads_only` is only 0.0023 absolute mPQ and it has lower bPQ/F1 with a larger parameter budget. `conv_adapters` is useful as a Fisher-motivated decoder-capacity ablation, but it is not the main KLT choice. VeRA+AdaptFormer is compact and stable enough to present as an alternative, but it remains below LoRA+AdaptFormer `heads_only`.

## What each strategy trains

| Strategy | Decoder scope | Trainable components | Practical interpretation |
|---|---|---|---|
| FullFT | all | Full encoder, decoder, and classifier | Upper reference, not parameter-efficient. |
| Freeze | none | Nothing | Sanity check for pretrained transfer without adaptation. |
| LoRA-only | decoder_frozen | LoRA attention adapters only | Tests whether low-rank attention adaptation alone is enough. On KLT, it is not. |
| AdaptFormer-only | decoder_frozen | Encoder MLP adapters only | Tests whether MLP adapter adaptation alone is enough. On KLT, it is not. |
| VeRA-only | decoder_frozen | VeRA attention adapters only | Extremely compact attention-only control. On KLT, mPQ remains low. |
| Frozen encoder + decoder conv adapters | conv_adapters | Decoder conv adapters plus final heads/classifier, no encoder adapters | Tests decoder-local adaptation without encoder PEFT. |
| LoRA+AdaptFormer | heads_only | Encoder LoRA + AdaptFormer, final NP/HV/NT heads, classifier | Main strategy. Freezes decoder body and only adapts the output heads. |
| LoRA+AdaptFormer | last_stage | Encoder LoRA + AdaptFormer, `decoder0.*`, decoder headers, classifier | More decoder plasticity, useful secondary mPQ ablation. |
| LoRA+AdaptFormer | conv_adapters | Encoder LoRA + AdaptFormer, decoder residual conv adapters, final heads, classifier | Fisher-motivated decoder-capacity ablation. |
| VeRA+AdaptFormer | heads_only | Encoder VeRA + AdaptFormer, final NP/HV/NT heads, classifier | Compact alternative to LoRA+AdaptFormer `heads_only`; currently only one completed KLT seed. |
| VeRA+AdaptFormer | conv_adapters | Encoder VeRA + AdaptFormer, decoder residual conv adapters, final heads, classifier | Compact alternative with two completed KLT seeds. |

Implementation note: `heads_only` does not add a new decoder layer. It makes the existing final NP/HV/NT decoder head parameters trainable, plus the classifier head, while the decoder body stays frozen. `conv_adapters` inserts small residual bottleneck adapters inside selected decoder convolutions and trains those adapters plus the final heads/classifier.

## KLT multitissue completed test rows

Dataset: `sthelar40x_kidney_liver_tonsil_5class_spatial_margin128`. Metrics below use completed `test_*` rows only.

| Method | Decoder scope | Seed | Trainable params | Trainable % | Dice | Jaccard | bPQ | mPQ | F1 | Precision | Recall |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| FullFT | all | 42 | 699,736,523 | 100.0000 | 0.7819 | 0.6681 | 0.5084 | 0.3058 | 0.8271 | 0.8652 | 0.7922 |
| FullFT | all | 43 | 699,736,523 | 100.0000 | 0.7870 | 0.6767 | 0.5211 | 0.3126 | 0.8301 | 0.8761 | 0.7887 |
| LoRA+AdaptFormer | heads_only | 42 | 7,908,779 | 1.1176 | 0.7741 | 0.6600 | 0.5076 | 0.2969 | 0.8336 | 0.8835 | 0.7890 |
| LoRA+AdaptFormer | heads_only | 43 | 7,908,779 | 1.1176 | 0.7703 | 0.6553 | 0.5009 | 0.2903 | 0.8355 | 0.8858 | 0.7906 |
| LoRA+AdaptFormer | last_stage | 42 | 8,261,291 | 1.1674 | 0.7771 | 0.6625 | 0.5064 | 0.2978 | 0.8279 | 0.8849 | 0.7777 |
| LoRA+AdaptFormer | last_stage | 43 | 8,261,291 | 1.1674 | 0.7727 | 0.6581 | 0.4988 | 0.2941 | 0.8240 | 0.8790 | 0.7754 |
| LoRA+AdaptFormer | conv_adapters | 42 | 8,460,874 | 1.1947 | 0.7685 | 0.6516 | 0.4890 | 0.2864 | 0.8290 | 0.8803 | 0.7833 |
| LoRA+AdaptFormer | conv_adapters | 43 | 8,460,874 | 1.1947 | 0.7730 | 0.6576 | 0.4972 | 0.2904 | 0.8296 | 0.8774 | 0.7867 |
| VeRA+AdaptFormer | heads_only | 43 | 6,681,035 | 0.9458 | 0.7671 | 0.6500 | 0.4932 | 0.2870 | 0.8326 | 0.8777 | 0.7920 |
| VeRA+AdaptFormer | conv_adapters | 42 | 7,233,130 | 1.0231 | 0.7713 | 0.6547 | 0.4918 | 0.2840 | 0.8328 | 0.8845 | 0.7868 |
| VeRA+AdaptFormer | conv_adapters | 43 | 7,233,130 | 1.0231 | 0.7702 | 0.6539 | 0.4917 | 0.2857 | 0.8247 | 0.8681 | 0.7854 |
| Frozen encoder + decoder conv adapters | conv_adapters | 42 | 553,002 | 0.0790 | 0.7434 | 0.6219 | 0.4587 | 0.2317 | 0.8282 | 0.8787 | 0.7832 |
| LoRA-only | decoder_frozen | 42 | 1,310,720 | 0.1870 | 0.7396 | 0.6140 | 0.4303 | 0.1127 | 0.8308 | 0.9105 | 0.7640 |
| AdaptFormer-only | decoder_frozen | 42 | 6,597,152 | 0.9340 | 0.7131 | 0.5827 | 0.4184 | 0.1185 | 0.8059 | 0.9213 | 0.7163 |
| VeRA-only | decoder_frozen | 42 | 82,976 | 0.0119 | 0.7301 | 0.6078 | 0.4569 | 0.0647 | 0.8272 | 0.8783 | 0.7818 |
| Freeze | none | 42 | 0 | 0.0000 | 0.7030 | 0.5750 | 0.4468 | 0.0026 | 0.8167 | 0.9242 | 0.7316 |

## KLT mean summary

| Method | Decoder scope | Completed seeds | Params | Trainable % | Mean mPQ | Std mPQ | Mean bPQ | Mean F1 | mPQ recovery vs FullFT | bPQ recovery vs FullFT |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| FullFT | all | 2 | 699,736,523 | 100.0000 | 0.3092 | 0.0048 | 0.5147 | 0.8286 | 100.00% | 100.00% |
| LoRA+AdaptFormer | heads_only | 2 | 7,908,779 | 1.1176 | 0.2936 | 0.0047 | 0.5043 | 0.8345 | 94.96% | 97.97% |
| LoRA+AdaptFormer | last_stage | 2 | 8,261,291 | 1.1674 | 0.2960 | 0.0027 | 0.5026 | 0.8259 | 95.71% | 97.65% |
| LoRA+AdaptFormer | conv_adapters | 2 | 8,460,874 | 1.1947 | 0.2884 | 0.0029 | 0.4931 | 0.8293 | 93.26% | 95.79% |
| VeRA+AdaptFormer | heads_only | 1 | 6,681,035 | 0.9458 | 0.2870 | NA | 0.4932 | 0.8326 | 92.83% | 95.82% |
| VeRA+AdaptFormer | conv_adapters | 2 | 7,233,130 | 1.0231 | 0.2849 | 0.0013 | 0.4917 | 0.8287 | 92.12% | 95.53% |
| Frozen encoder + decoder conv adapters | conv_adapters | 1 | 553,002 | 0.0790 | 0.2317 | NA | 0.4587 | 0.8282 | 74.94% | 89.12% |
| LoRA-only | decoder_frozen | 1 | 1,310,720 | 0.1870 | 0.1127 | NA | 0.4303 | 0.8308 | 36.44% | 83.59% |
| AdaptFormer-only | decoder_frozen | 1 | 6,597,152 | 0.9340 | 0.1185 | NA | 0.4184 | 0.8059 | 38.33% | 81.28% |
| VeRA-only | decoder_frozen | 1 | 82,976 | 0.0119 | 0.0647 | NA | 0.4569 | 0.8272 | 20.92% | 88.76% |
| Freeze | none | 1 | 0 | 0.0000 | 0.0026 | NA | 0.4468 | 0.8167 | 0.85% | 86.79% |

## Ranking and interpretation

By mean mPQ among two-seed PEFT runs: `last_stage` is first, `heads_only` second, `conv_adapters` third, and VeRA+AdaptFormer `conv_adapters` fourth. The `last_stage` advantage over `heads_only` is small: 0.2960 versus 0.2936.

By mean bPQ and F1, `heads_only` is the best LoRA+AdaptFormer scope: bPQ 0.5043 and F1 0.8345. This matters because the mPQ difference between `heads_only` and `last_stage` is very small, while `heads_only` is more compact and has better binary PQ/detection.

By parameter efficiency, VeRA-only is the smallest but underfits mPQ badly. Among competitive methods, VeRA+AdaptFormer is the most compact family, while LoRA+AdaptFormer `heads_only` is the best strong method because it stays close to FullFT with only 1.1176% trainable parameters.

The negative controls are informative. Freeze, LoRA-only, AdaptFormer-only, and VeRA-only keep F1/bPQ at nonzero levels but collapse mPQ, which suggests that KLT needs type/head adaptation and not just generic frozen-feature detection. Frozen encoder plus decoder conv adapters partially recovers mPQ, showing that decoder-local adaptation helps, but it is still far below LoRA+AdaptFormer `heads_only`.

## Fisher diagnostic connection

The KLT FullFT seed43 Fisher diagnostic in `reports/fisher_drift/klt_fullft_seed43_blocks_b5` supports the selected strategy:

| Fisher group | PEFT connection | Interpretation |
|---|---|---|
| `final_heads` | `heads_only` | Highest Fisher mean per parameter; strong reason to train the final NP/HV/NT heads. |
| `encoder_mlp` | AdaptFormer | Large normalized Fisher mass; supports MLP bottleneck adapters. |
| `encoder_attention_qkv` / `encoder_attention_proj` | LoRA or VeRA | Supports low-rank attention adaptation. |
| `decoder_conv_like` and late decoder drift | `conv_adapters` or `last_stage` | Supports these as ablations, but test metrics keep them secondary. |

At checkpoint 10, `final_heads` has Fisher mean about `3.13e-05` with only 333,578 parameters. Normalized Fisher mass is led by `other_trainable` 0.4387, `encoder_mlp` 0.2563, `encoder_attention_qkv` 0.1083, `encoder_attention_proj` 0.0763, and `final_heads` 0.0491. JS drift from epoch 5 to 10 is largest in decoder output branches (`decoder_np`, `decoder_hv`, `decoder_nt`), which explains why late-decoder ablations are scientifically motivated even though they are not the default recommendation.

## Current recommendation for tissue-specific work

Use **LoRA+AdaptFormer `heads_only`** as the main tissue-specific PEFT strategy for Group 1. Report **LoRA+AdaptFormer `last_stage`** as the secondary mPQ-oriented ablation if budget allows. Keep **`conv_adapters`** as a Fisher-motivated decoder-capacity ablation, and present **VeRA+AdaptFormer** as a compact alternative rather than the main strategy.
