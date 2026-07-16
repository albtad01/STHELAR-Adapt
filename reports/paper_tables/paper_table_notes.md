# Paper table notes

- KLT FullFT is the reference row: mean mPQ 0.309, bPQ 0.515, detection F1 0.829, and type accuracy 0.781 across 2 seeds.
- The strongest KLT parameter-efficient row by mPQ is LoRA+AF last_stage, reaching 0.296 mPQ, or 95.7% of FullFT, with 1.167% trainable parameters.
- Frozen CellViT keeps detection usable but has near-zero type assignment on KLT: macro type F1 0.010 and mPQ recovery 0.8%.
- Macro type F1 excluding zero-support classes is included because Melanocyte has zero true support in these 5-class final evaluations; this avoids penalizing methods for an absent class.
- Macro type F1 without Other is also included over present non-Other classes, separating core epithelial/immune/stromal behavior from the heterogeneous Other label.
- Tissue-specific PEFT recovers the most FullFT mPQ for Liver (95.2%) and the least for Kidney (88.9%).
- The ovary tissue-specific rows should be discussed with the slide dominance caveat from the slide-wise report, since aggregate ovary metrics are dominated by one test slide.
