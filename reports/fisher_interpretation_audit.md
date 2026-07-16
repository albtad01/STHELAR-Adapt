# Fisher interpretation audit

Source directory: `reports/fisher_drift/klt_fullft_seed43_blocks_b5`

Source files read:

- `fisher_summary.csv`
- `fisher_drift_summary.csv`
- Existing derived figure CSVs were inspected for consistency, but the audit values below are recomputed from the two source CSVs.

Full numeric audit table: `reports/fisher_interpretation_audit_values.csv`

## Focus groups

| group | params | mass@10 | mass % @10 | mass/param @10 | 1e5 mass/param | JS 1->5 | JS 5->10 | JS/param 1->5 | JS/param 5->10 | 1e7 JS/param 1->5 | 1e7 JS/param 5->10 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| final_heads | 333,578 | 10.4404 | 4.91 | 3.130e-05 | 3.1298 | 0.4640 | 0.3030 | 1.391e-06 | 9.083e-07 | 1.391e+01 | 9.083e+00 |
| decoder_nt | 14,966,464 | 7.1570 | 3.37 | 4.782e-07 | 0.0478 | 0.7434 | 0.7114 | 4.967e-08 | 4.753e-08 | 4.967e-01 | 4.753e-01 |
| decoder_conv_like | 17,477,248 | 3.9566 | 1.86 | 2.264e-07 | 0.0226 | 0.4034 | 0.3638 | 2.308e-08 | 2.081e-08 | 2.308e-01 | 2.081e-01 |
| encoder_attention_qkv | 157,409,280 | 23.0206 | 10.83 | 1.462e-07 | 0.0146 | 0.4519 | 0.4101 | 2.871e-09 | 2.605e-09 | 2.871e-02 | 2.605e-02 |
| encoder_attention_proj | 52,469,760 | 16.2225 | 7.63 | 3.092e-07 | 0.0309 | 0.4524 | 0.4048 | 8.623e-09 | 7.715e-09 | 8.623e-02 | 7.715e-02 |
| encoder_mlp | 419,635,200 | 54.4690 | 25.63 | 1.298e-07 | 0.0130 | 0.4255 | 0.4212 | 1.014e-09 | 1.004e-09 | 1.014e-02 | 1.004e-02 |

## Rankings

### Top Fisher mass at checkpoint 10

| group | mass@10 | mass % | rank |
| --- | --- | --- | --- |
| other_trainable | 93.2120 | 43.87 | 1 |
| encoder_mlp | 54.4690 | 25.63 | 2 |
| encoder_attention_qkv | 23.0206 | 10.83 | 3 |
| encoder_attention_proj | 16.2225 | 7.63 | 4 |
| final_heads | 10.4404 | 4.91 | 5 |
| decoder_nt | 7.1570 | 3.37 | 6 |
| decoder_conv_like | 3.9566 | 1.86 | 7 |
| encoder_early_blocks | 2.2523 | 1.06 | 8 |

### Top Fisher mass per parameter at checkpoint 10

| group | mass/param | 1e5 mass/param | rank |
| --- | --- | --- | --- |
| final_heads | 3.130e-05 | 3.1298 | 1 |
| encoder_early_blocks | 1.879e-05 | 1.8794 | 2 |
| other_trainable | 1.304e-05 | 1.3044 | 3 |
| encoder_mid_blocks | 7.051e-06 | 0.7051 | 4 |
| encoder_late_blocks | 1.541e-06 | 0.1541 | 5 |
| decoder_nt | 4.782e-07 | 0.0478 | 6 |
| encoder_attention_proj | 3.092e-07 | 0.0309 | 7 |
| decoder_conv_like | 2.264e-07 | 0.0226 | 8 |

### Top raw JS drift 1->5

| group | JS 1->5 | rank |
| --- | --- | --- |
| decoder_np | 0.7594 | 1 |
| decoder_hv | 0.7561 | 2 |
| decoder_nt | 0.7434 | 3 |
| encoder_late_blocks | 0.5162 | 4 |
| final_heads | 0.4640 | 5 |
| encoder_attention_proj | 0.4524 | 6 |
| encoder_attention_qkv | 0.4519 | 7 |
| encoder_mlp | 0.4255 | 8 |

### Top raw JS drift 5->10

| group | JS 5->10 | rank |
| --- | --- | --- |
| decoder_np | 0.7563 | 1 |
| decoder_hv | 0.7386 | 2 |
| decoder_nt | 0.7114 | 3 |
| encoder_late_blocks | 0.4746 | 4 |
| encoder_mlp | 0.4212 | 5 |
| encoder_attention_qkv | 0.4101 | 6 |
| encoder_attention_proj | 0.4048 | 7 |
| decoder_conv_like | 0.3638 | 8 |

### Top JS drift per parameter 1->5

| group | JS/param 1->5 | 1e7 JS/param | rank |
| --- | --- | --- | --- |
| encoder_late_blocks | 4.084e-06 | 4.084e+01 | 1 |
| encoder_mid_blocks | 3.308e-06 | 3.308e+01 | 2 |
| encoder_early_blocks | 3.101e-06 | 3.101e+01 | 3 |
| final_heads | 1.391e-06 | 1.391e+01 | 4 |
| decoder_np | 5.074e-08 | 5.074e-01 | 5 |
| decoder_hv | 5.052e-08 | 5.052e-01 | 6 |
| decoder_nt | 4.967e-08 | 4.967e-01 | 7 |
| other_trainable | 4.851e-08 | 4.851e-01 | 8 |

### Top JS drift per parameter 5->10

| group | JS/param 5->10 | 1e7 JS/param | rank |
| --- | --- | --- | --- |
| encoder_late_blocks | 3.755e-06 | 3.755e+01 | 1 |
| encoder_mid_blocks | 3.032e-06 | 3.032e+01 | 2 |
| encoder_early_blocks | 2.703e-06 | 2.703e+01 | 3 |
| final_heads | 9.083e-07 | 9.083e+00 | 4 |
| decoder_np | 5.053e-08 | 5.053e-01 | 5 |
| decoder_hv | 4.935e-08 | 4.935e-01 | 6 |
| decoder_nt | 4.753e-08 | 4.753e-01 | 7 |
| other_trainable | 4.741e-08 | 4.741e-01 | 8 |

## Explicit answers

**A) Is it true that final_heads dominate the parameter-normalized view at both 1->5 and 5->10?** False if interpreted globally across all groups: small encoder block groups have larger JS drift per parameter. True within the PEFT-relevant groups listed in the paper: final_heads is the largest in Fisher mass per parameter and in JS drift per parameter for both 1->5 and 5->10.

**B) Is it true that encoder MLP / attention dominate late drift?** False for raw late JS drift: decoder_np, decoder_hv, and decoder_nt dominate 5->10 drift. Encoder MLP and attention QKV/proj have substantial but not dominant late drift. They do, however, dominate or rank near the top in total Fisher mass among PEFT-relevant encoder groups.

**C) Which groups dominate in total Fisher mass?** At checkpoint 10, total Fisher mass is dominated by other_trainable, encoder_mlp, encoder_attention_qkv, encoder_attention_proj, then final_heads. Among interpretable PEFT-relevant groups, encoder_mlp is the largest, followed by attention QKV/proj; final_heads is smaller in total mass but very high per parameter.

**D) Most defensible narrative.**

Fisher diagnostics were used as a post-hoc sensitivity analysis rather than a model-selection rule. In the KLT full fine-tuning probe, checkpoint-10 Fisher mass is concentrated in encoder MLP (25.6% of total) and attention QKV/projection groups (10.8% and 7.6%), supporting encoder-side PEFT placement in MLP and attention pathways. Final decoder heads do not dominate total mass (4.9%), but have by far the largest Fisher mass per parameter among PEFT-relevant groups (3.13 after scaling by 10^5), supporting final-head tuning as a parameter-efficient decoder adaptation. Raw JS drift from checkpoint 5 to 10 is highest in decoder branches, especially NP/HV/NT, so late drift motivates decoder-scope ablations but does not by itself prove that broader decoder tuning is preferable. The final choice of LoRA+AdaptFormer+final-head tuning should therefore be framed as consistent with Fisher sensitivity and selected empirically by validation/test ablations.

## Recommended paper paragraph

Fisher diagnostics were used as a post-hoc sensitivity analysis rather than a model-selection rule. In the KLT full fine-tuning probe, checkpoint-10 Fisher mass is concentrated in encoder MLP (25.6% of total) and attention QKV/projection groups (10.8% and 7.6%), supporting encoder-side PEFT placement in MLP and attention pathways. Final decoder heads do not dominate total mass (4.9%), but have by far the largest Fisher mass per parameter among PEFT-relevant groups (3.13 after scaling by 10^5), supporting final-head tuning as a parameter-efficient decoder adaptation. Raw JS drift from checkpoint 5 to 10 is highest in decoder branches, especially NP/HV/NT, so late drift motivates decoder-scope ablations but does not by itself prove that broader decoder tuning is preferable. The final choice of LoRA+AdaptFormer+final-head tuning should therefore be framed as consistent with Fisher sensitivity and selected empirically by validation/test ablations.

## Caution

- The raw JS drift and JS drift per parameter support different claims. Raw JS drift highlights where Fisher structure changes most in absolute group terms; JS drift per parameter strongly favors small groups and can be dominated by small non-PEFT groups.
- Therefore, the paper should avoid saying that final_heads dominate all Fisher views, or that encoder MLP/attention dominate late JS drift. The safer claim is that encoder MLP/attention dominate total Fisher relevance among encoder PEFT targets, while final_heads dominate parameter-normalized efficiency among PEFT-relevant groups.
