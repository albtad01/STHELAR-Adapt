# Fisher figures summary

Generated from audited values in `reports/fisher_interpretation_audit_values.csv`.

## Outputs

- `figures/fisher_workshop_main_panel.pdf` / `.png`
- `figures/fisher_workshop_supp_heatmap.pdf` / `.png`
- `figures/fisher_workshop_supp_top3.pdf` / `.png`

## Values used

| group        |   param_count |   fisher_mass_percent_ckpt10 |   fisher_mean_x1e5_ckpt10 |   js_per_param_x1e7_1_to_5 |   js_per_param_x1e7_5_to_10 |
|:-------------|--------------:|-----------------------------:|--------------------------:|---------------------------:|----------------------------:|
| encoder MLP  |     419635200 |                      25.6329 |                    0.0130 |                     0.0101 |                      0.0100 |
| attn QKV     |     157409280 |                      10.8334 |                    0.0146 |                     0.0287 |                      0.0261 |
| attn proj    |      52469760 |                       7.6343 |                    0.0309 |                     0.0862 |                      0.0771 |
| final heads  |        333578 |                       4.9132 |                    3.1298 |                    13.9101 |                      9.0835 |
| decoder NT   |      14966464 |                       3.3681 |                    0.0478 |                     0.4967 |                      0.4753 |
| decoder conv |      17477248 |                       1.8620 |                    0.0226 |                     0.2308 |                      0.2081 |

## Proposed Figure A caption

Post-hoc Fisher diagnostic on the KLT full fine-tuning probe. Panel (a) shows checkpoint-10 Fisher mass, highlighting total task sensitivity in encoder MLP and attention groups. Panel (b) shows Fisher mass per parameter, where final decoder heads provide the strongest parameter-normalized signal. Panel (c) shows Jensen--Shannon Fisher drift per parameter for checkpoint transitions 1→5 and 5→10. The diagnostic is used to interpret PEFT placement, not to select the final model.

## Recommended Results paragraph

The Fisher diagnostic supports different aspects of the selected PEFT design. At checkpoint 10, total Fisher mass is concentrated in the encoder MLP and attention groups, consistent with placing AdaptFormer modules in MLP blocks and LoRA in the attention QKV pathway. In contrast, final decoder heads have the strongest Fisher mass per parameter among the analyzed PEFT-relevant groups, supporting final-head tuning as an efficient decoder adaptation target. The JS drift view shows that parameter-normalized drift also favors final heads within these PEFT-relevant groups, although raw drift is larger in broader decoder branches. Thus, Fisher is best interpreted as post-hoc support for adapter placement, while final model choice remains determined by the ablation results.
