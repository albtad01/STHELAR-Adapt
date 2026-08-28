# KLT ablation strategy recommendation

> **Paper-registry note, 2026-08-27:** this completed within-slide ablation is
> historical method-selection evidence. It includes LP, LoRA, VeRA,
> AdaptFormer, decoder-convolution/late-decoder combinations, Selected PEFT and
> FullFT families where listed below. The primary LoRA(Q,V)+AdaptFormer+final
> heads method was selected before the held-out-slide campaign. Do not rerun or
> reopen this architecture search for the revised paper; report it in the
> appendix or as historical pre-selection provenance.
> The old table's Frozen class-aware mPQ/type columns are not valid under the
> PanNuke/STHELAR taxonomy mismatch; only its class-agnostic detection and
> segmentation quantities may be reused.

Source metrics: `reports/runs_summary.csv`, using only completed runs with final inference/test metrics (`test_*`). Validation epoch metrics are not mixed into the comparisons below. Earlier partial/no-test rows are excluded when a later completed final-inference row exists for the same run name.

## 1. KLT ablation table

| method | decoder scope | seed | trainable params % | Dice | Jaccard | bPQ | mPQ | F1 detection | precision | recall | job state/status |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| FullFT | all | 42 | 100.0000 | 0.7819 | 0.6681 | 0.5084 | 0.3058 | 0.8271 | 0.8652 | 0.7922 | completed_with_test |
| FullFT | all | 43 | 100.0000 | 0.7870 | 0.6767 | 0.5211 | 0.3126 | 0.8301 | 0.8761 | 0.7887 | completed_with_test |
| LoRA+AdaptFormer | conv_adapters | 42 | 1.1947 | 0.7685 | 0.6516 | 0.4890 | 0.2864 | 0.8290 | 0.8803 | 0.7833 | completed_with_test |
| LoRA+AdaptFormer | conv_adapters | 43 | 1.1947 | 0.7730 | 0.6576 | 0.4972 | 0.2904 | 0.8296 | 0.8774 | 0.7867 | completed_with_test |
| LoRA+AdaptFormer | heads_only | 42 | 1.1176 | 0.7741 | 0.6600 | 0.5076 | 0.2969 | 0.8336 | 0.8835 | 0.7890 | completed_with_test |
| LoRA+AdaptFormer | heads_only | 43 | 1.1176 | 0.7703 | 0.6553 | 0.5009 | 0.2903 | 0.8355 | 0.8858 | 0.7906 | completed_with_test |
| LoRA+AdaptFormer | last_stage | 42 | 1.1674 | 0.7771 | 0.6625 | 0.5064 | 0.2978 | 0.8279 | 0.8849 | 0.7777 | completed_with_test |
| LoRA+AdaptFormer | last_stage | 43 | 1.1674 | 0.7727 | 0.6581 | 0.4988 | 0.2941 | 0.8240 | 0.8790 | 0.7754 | completed_with_test |
| VeRA+AdaptFormer | conv_adapters | 42 | 1.0231 | 0.7713 | 0.6547 | 0.4918 | 0.2840 | 0.8328 | 0.8845 | 0.7868 | completed_with_test |
| VeRA+AdaptFormer | conv_adapters | 43 | 1.0231 | 0.7702 | 0.6539 | 0.4917 | 0.2857 | 0.8247 | 0.8681 | 0.7854 | completed_with_test |
| VeRA+AdaptFormer | heads_only | 43 | 0.9458 | 0.7671 | 0.6500 | 0.4932 | 0.2870 | 0.8326 | 0.8777 | 0.7920 | completed_with_test |

Earlier partial/no-test attempts for some seed43 VeRA and last_stage runs are excluded where a later completed final-inference row exists for the same run name.

## 2. KLT mean table by method

| method | decoder scope | completed seeds | mean mPQ | std mPQ | mean bPQ | mean F1 detection | relative mPQ recovery vs FullFT | relative bPQ recovery vs FullFT |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| FullFT | all | 2 | 0.3092 | 0.0048 | 0.5147 | 0.8286 | 100.0% | 100.0% |
| LoRA+AdaptFormer | conv_adapters | 2 | 0.2884 | 0.0029 | 0.4931 | 0.8293 | 93.3% | 95.8% |
| LoRA+AdaptFormer | heads_only | 2 | 0.2936 | 0.0047 | 0.5043 | 0.8345 | 95.0% | 98.0% |
| LoRA+AdaptFormer | last_stage | 2 | 0.2960 | 0.0027 | 0.5026 | 0.8259 | 95.7% | 97.6% |
| VeRA+AdaptFormer | conv_adapters | 2 | 0.2849 | 0.0013 | 0.4917 | 0.8287 | 92.1% | 95.5% |
| VeRA+AdaptFormer | heads_only | 1 | 0.2870 | NA | 0.4932 | 0.8326 | 92.8% | 95.8% |

FullFT mean is mPQ 0.3092 and bPQ 0.5147 across seeds 42/43.

## 3. PEFT ranking

By mean mPQ: LoRA+AdaptFormer last_stage (0.2960) > LoRA+AdaptFormer heads_only (0.2936) > LoRA+AdaptFormer conv_adapters (0.2884) > VeRA+AdaptFormer heads_only (0.2870, one seed) > VeRA+AdaptFormer conv_adapters (0.2849).

By mean bPQ: LoRA+AdaptFormer heads_only (0.5043) > LoRA+AdaptFormer last_stage (0.5026) > VeRA+AdaptFormer heads_only (0.4932, one seed) > LoRA+AdaptFormer conv_adapters (0.4931) > VeRA+AdaptFormer conv_adapters (0.4917).

By seed stability in mPQ among two-seed LoRA PEFT runs: last_stage std 0.0027, conv_adapters std 0.0029, heads_only std 0.0047. These are all small relative to the FullFT two-seed std of 0.0048.

By trainable-parameter efficiency: VeRA+AdaptFormer heads_only uses the least among completed encoder+decoder PEFT rows (0.9458%) but currently has one completed KLT seed; VeRA+AdaptFormer conv_adapters has two completed seeds at 1.0231%; LoRA+AdaptFormer heads_only is the strongest two-seed efficient LoRA choice at 1.1176%; last_stage costs slightly more at 1.1674%; conv_adapters costs the most of these LoRA scopes at 1.1947%.

## 4. Interpretation

LoRA+AdaptFormer heads_only is still the best main PEFT recommendation on balance. Last_stage has the highest two-seed mean mPQ, but the gain over heads_only is only 0.0023 absolute mPQ, while heads_only has higher bPQ, higher detection F1, and fewer trainable parameters.

LoRA+AdaptFormer last_stage does not beat heads_only by a meaningful margin for the main story. It is a useful secondary ablation because it slightly improves mPQ and is stable, but the bPQ/F1 tradeoff makes it less compelling as the default.

Conv_adapters are worth reporting, but not emphasizing as the main KLT choice. They are stable and biologically/architecturally motivated, but they trail heads_only by 0.0053 mPQ and 0.0112 bPQ while using more trainable parameters.

VeRA+AdaptFormer is competitive enough to present as a compact alternative, not as the main recommendation. The completed r16 conv two-seed mean recovers 92.1% of FullFT mPQ and 95.5% of FullFT bPQ with 1.0231% trainable parameters; the completed heads_only seed43 row recovers 92.8% mPQ and 95.8% bPQ with 0.9458% trainable parameters. These are useful compact baselines, but they remain below LoRA+AdaptFormer heads_only.

The completed VeRA rows improve the compact-alternative comparison, but they do not change the tissue-specific Group 1 default: LoRA+AdaptFormer heads_only remains the main PEFT strategy.

## 5. Fisher diagnostic connection

The Fisher diagnostic in `reports/fisher_drift/klt_fullft_seed43_blocks_b5` supports the PEFT design but does not replace test-set ranking. At checkpoint 10, `final_heads` has the highest Fisher mean despite only 333,578 parameters, directly supporting the heads_only decoder scope as the most parameter-efficient target. The encoder signal is also clear: `encoder_mlp` is rank 2 by normalized Fisher mass, motivating AdaptFormer, while `encoder_attention_qkv` and `encoder_attention_proj` are ranks 3 and 4 by normalized Fisher mass, motivating LoRA/VeRA attention adaptation.

Decoder evidence is mixed but useful. `decoder_conv_like` has nontrivial checkpoint-10 Fisher mass and drift, supporting conv_adapters as a real ablation, while late JS drift is strongest in decoder branches (`decoder_np`, `decoder_hv`, `decoder_nt`), supporting last_stage/late-decoder exploration. This aligns with the observed result: last_stage can marginally improve mPQ, but heads_only remains the cleaner default because the most parameter-efficient Fisher signal is concentrated in final heads and the test metrics favor heads_only on bPQ/F1.

## 6. Recommendation

Main PEFT strategy for tissue-specific Group 1: launch/report LoRA+AdaptFormer with decoder `heads_only` as the default. It gives the best balanced KLT recovery at 94.96% of FullFT mPQ and 97.97% of FullFT bPQ using 1.1176% trainable parameters, and it has the best mean bPQ and F1 among the completed PEFT scopes.

Secondary ablation to report: LoRA+AdaptFormer `last_stage`, because it has the highest mean mPQ and the lowest two-seed mPQ std among the LoRA scopes, but should be framed as a marginal mPQ-oriented variant rather than a replacement for heads_only. Include conv_adapters as the decoder-capacity/Fisher-motivated ablation, but keep the emphasis lower unless tissue runs show a clearer benefit.

VeRA should be reported as a compact alternative after checking that the completed seed43 rows remain in the refreshed run table. It should not block tissue PEFT with the LoRA+AdaptFormer heads_only default, because the LoRA decision is supported by two completed KLT seeds and by the Fisher diagnostic.

## 7. Tissue FullFT summary

| tissue | Dice | Jaccard | bPQ | mPQ | F1 detection |
|---|---:|---:|---:|---:|---:|
| kidney | 0.7464 | 0.6311 | 0.4945 | 0.2592 | 0.8285 |
| liver | 0.8280 | 0.7265 | 0.5952 | 0.4124 | 0.8839 |
| ovary | 0.7933 | 0.6829 | 0.5293 | 0.3195 | 0.8311 |

Kidney has decent detection F1 (0.8285) but lower mPQ (0.2592), which points to classification/type-quality or class-conditional PQ being the bottleneck rather than nucleus finding alone. Its bPQ is moderate (0.4945), so binary segmentation/detection is not collapsing; the larger drop in mPQ suggests poorer multi-class instance quality, likely from harder kidney morphology, class imbalance, or more confusing phenotype boundaries under the 5-class tissue-specific label set.

## Compact recommendation paragraph

Across completed KLT test runs, LoRA+AdaptFormer with decoder heads_only is the strongest main PEFT strategy because it gives the best balanced recovery of FullFT performance with only 1.1176% trainable parameters. Last_stage has a slightly higher mean mPQ than heads_only, but the margin is small and is offset by lower bPQ/F1 and a larger parameter budget. Conv_adapters are a useful Fisher-motivated ablation, but the completed KLT metrics do not justify emphasizing them as the default. VeRA+AdaptFormer is now a reportable compact alternative, with two completed conv_adapters seeds and one completed heads_only seed, but it remains below LoRA+AdaptFormer heads_only. For tissue-specific Group 1, proceed with LoRA+AdaptFormer heads_only as the main PEFT and report last_stage as the secondary mPQ-oriented ablation.
