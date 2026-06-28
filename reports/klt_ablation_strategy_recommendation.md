# KLT ablation strategy recommendation

Source metrics: `reports/runs_summary.csv`, using only completed runs with final inference/test metrics (`test_*`). Validation epoch metrics are not mixed into the comparisons below. Job-state context for jobs 1172354 and 1172355 was checked separately with `sacct` because the CSV job-id fields are blank for these rows.

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

Pending/non-comparable KLT rows in the CSV are excluded from the metric table because they do not have final inference metrics. In particular, KLT VeRA+AdaptFormer r16 `conv_adapters` seed43 and `heads_only` seed43 currently appear as `partial_or_timeout`/no-test rows in the CSV.

## 2. KLT mean table by method

| method | decoder scope | completed seeds | mean mPQ | std mPQ | mean bPQ | mean F1 detection | relative mPQ recovery vs FullFT | relative bPQ recovery vs FullFT |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| FullFT | all | 2 | 0.3092 | 0.0048 | 0.5147 | 0.8286 | 100.0% | 100.0% |
| LoRA+AdaptFormer | conv_adapters | 2 | 0.2884 | 0.0029 | 0.4931 | 0.8293 | 93.3% | 95.8% |
| LoRA+AdaptFormer | heads_only | 2 | 0.2936 | 0.0047 | 0.5043 | 0.8345 | 95.0% | 98.0% |
| LoRA+AdaptFormer | last_stage | 2 | 0.2960 | 0.0027 | 0.5026 | 0.8259 | 95.7% | 97.6% |
| VeRA+AdaptFormer | conv_adapters | 1 | 0.2840 | NA | 0.4918 | 0.8328 | 91.8% | 95.5% |

FullFT mean is mPQ 0.3092 and bPQ 0.5147 across seeds 42/43.

## 3. PEFT ranking

By mean mPQ: LoRA+AdaptFormer last_stage (0.2960) > LoRA+AdaptFormer heads_only (0.2936) > LoRA+AdaptFormer conv_adapters (0.2884) > VeRA+AdaptFormer conv_adapters (0.2840, one seed).

By mean bPQ: LoRA+AdaptFormer heads_only (0.5043) > LoRA+AdaptFormer last_stage (0.5026) > LoRA+AdaptFormer conv_adapters (0.4931) > VeRA+AdaptFormer conv_adapters (0.4918, one seed).

By seed stability in mPQ among two-seed LoRA PEFT runs: last_stage std 0.0027, conv_adapters std 0.0029, heads_only std 0.0047. These are all small relative to the FullFT two-seed std of 0.0048.

By trainable-parameter efficiency: VeRA+AdaptFormer conv_adapters uses the least among completed encoder+decoder PEFT rows (1.0231%) but only has one completed KLT seed; LoRA+AdaptFormer heads_only is the strongest two-seed efficient choice at 1.1176%; last_stage costs slightly more at 1.1674%; conv_adapters costs the most of these LoRA scopes at 1.1947%.

## 4. Interpretation

LoRA+AdaptFormer heads_only is still the best main PEFT recommendation on balance. Last_stage has the highest two-seed mean mPQ, but the gain over heads_only is only 0.0023 absolute mPQ, while heads_only has higher bPQ, higher detection F1, and fewer trainable parameters.

LoRA+AdaptFormer last_stage does not beat heads_only by a meaningful margin for the main story. It is a useful secondary ablation because it slightly improves mPQ and is stable, but the bPQ/F1 tradeoff makes it less compelling as the default.

Conv_adapters are worth reporting, but not emphasizing as the main KLT choice. They are stable and biologically/architecturally motivated, but they trail heads_only by 0.0053 mPQ and 0.0112 bPQ while using more trainable parameters.

VeRA+AdaptFormer is competitive enough to present as a compact alternative, not yet as the main recommendation. The completed r16 conv seed42 recovers 91.8% of FullFT mPQ and 95.5% of FullFT bPQ with 1.0231% trainable parameters and strong F1, but KLT seed stability is unresolved.

Jobs 1172354 and 1172355 should be allowed to finish before finalizing the VeRA comparison, but they should not block launching tissue-specific LoRA+AdaptFormer heads_only runs. `sacct` reports both jobs as RUNNING at the time of this report: job 1172354 elapsed 12:29:33 and job 1172355 elapsed 10:04:25.

## 5. Fisher diagnostic connection

The Fisher diagnostic in `reports/fisher_drift/klt_fullft_seed43_blocks_b5` supports the PEFT design but does not replace test-set ranking. At checkpoint 10, `final_heads` has the highest Fisher mean despite only 333,578 parameters, directly supporting the heads_only decoder scope as the most parameter-efficient target. The encoder signal is also clear: `encoder_mlp` is rank 2 by normalized Fisher mass, motivating AdaptFormer, while `encoder_attention_qkv` and `encoder_attention_proj` are ranks 3 and 4 by normalized Fisher mass, motivating LoRA/VeRA attention adaptation.

Decoder evidence is mixed but useful. `decoder_conv_like` has nontrivial checkpoint-10 Fisher mass and drift, supporting conv_adapters as a real ablation, while late JS drift is strongest in decoder branches (`decoder_np`, `decoder_hv`, `decoder_nt`), supporting last_stage/late-decoder exploration. This aligns with the observed result: last_stage can marginally improve mPQ, but heads_only remains the cleaner default because the most parameter-efficient Fisher signal is concentrated in final heads and the test metrics favor heads_only on bPQ/F1.

## 6. Recommendation

Main PEFT strategy for tissue-specific Group 1: launch/report LoRA+AdaptFormer with decoder `heads_only` as the default. It gives the best balanced KLT recovery at 94.96% of FullFT mPQ and 97.97% of FullFT bPQ using 1.1176% trainable parameters, and it has the best mean bPQ and F1 among the completed PEFT scopes.

Secondary ablation to report: LoRA+AdaptFormer `last_stage`, because it has the highest mean mPQ and the lowest two-seed mPQ std among the LoRA scopes, but should be framed as a marginal mPQ-oriented variant rather than a replacement for heads_only. Include conv_adapters as the decoder-capacity/Fisher-motivated ablation, but keep the emphasis lower unless tissue runs show a clearer benefit.

VeRA jobs should be waited on before making a final VeRA claim, especially for seed43 and heads_only. They do not need to be waited on before starting tissue PEFT with the LoRA+AdaptFormer heads_only default, because the LoRA decision is already supported by two completed KLT seeds and by the Fisher diagnostic.

## 7. Tissue FullFT summary

| tissue | Dice | Jaccard | bPQ | mPQ | F1 detection |
|---|---:|---:|---:|---:|---:|
| kidney | 0.7464 | 0.6311 | 0.4945 | 0.2592 | 0.8285 |
| liver | 0.8280 | 0.7265 | 0.5952 | 0.4124 | 0.8839 |
| ovary | 0.7933 | 0.6829 | 0.5293 | 0.3195 | 0.8311 |

Kidney has decent detection F1 (0.8285) but lower mPQ (0.2592), which points to classification/type-quality or class-conditional PQ being the bottleneck rather than nucleus finding alone. Its bPQ is moderate (0.4945), so binary segmentation/detection is not collapsing; the larger drop in mPQ suggests poorer multi-class instance quality, likely from harder kidney morphology, class imbalance, or more confusing phenotype boundaries under the 5-class tissue-specific label set.

## Compact recommendation paragraph

Across completed KLT test runs, LoRA+AdaptFormer with decoder heads_only is the strongest main PEFT strategy because it gives the best balanced recovery of FullFT performance with only 1.1176% trainable parameters. Last_stage has a slightly higher mean mPQ than heads_only, but the margin is small and is offset by lower bPQ/F1 and a larger parameter budget. Conv_adapters are a useful Fisher-motivated ablation, but the completed KLT metrics do not justify emphasizing them as the default. VeRA+AdaptFormer is promising as a compact alternative, but its KLT conclusion should remain provisional until the running seed43/head-scope jobs finish. For tissue-specific Group 1, proceed with LoRA+AdaptFormer heads_only as the main PEFT and report last_stage as the secondary mPQ-oriented ablation.
