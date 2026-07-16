# Type metric interpretation

- KLT FullFT remains the reference ceiling with mPQ 0.309, bPQ 0.515, F1_detection 0.829, and type accuracy 0.781 across 2 seeds.
- LoRA+AF heads_only remains the best default when type accuracy is considered: it has the highest mean type accuracy (0.783) and highest F1_detection (0.835), while retaining 95.0% of FullFT mPQ with 1.118% trainable parameters.
- LoRA+AF last_stage is stronger for macro type-F1 and Other-class behavior: present-class macro F1 is 0.654 versus 0.578 for heads_only, and Other F1 is 0.362 versus 0.065.
- The best non-FullFT mPQ row is LoRA+AF last_stage at mPQ 0.296; the best present-class macro F1 row overall is FullFT all at 0.666.
- Frozen CellViT preserves nucleus detection but fails type assignment: F1_detection 0.817 and bPQ 0.447, but mPQ 0.003, type accuracy 0.022, and macro type-F1 0.010.
- Across tissues with both final FullFT and PEFT results, LoRA+AF heads_only recovers the most mPQ on Liver (95.2%) and the least on Kidney (88.9%).
- Melanocyte has zero true support in these 5-class final evaluations, so `macro_type_f1_present_classes` excludes zero-support classes; otherwise macro F1 would include an absent-class zero.
- The Other/Unknown category is heterogeneous and materially affects macro-F1; `macro_type_f1_without_other` is provided to separate core epithelial, immune, and stromal behavior from the catch-all class.

## Suggested Frozen CellViT sentence

Frozen CellViT transfers nucleus detection reasonably well (F1_detection=0.817, bPQ=0.447) but essentially fails STHELAR type assignment (mPQ=0.003, type accuracy=0.022, macro type-F1=0.010), showing that detection transfer alone is insufficient for cell-type labeling.

## Compact KLT LaTeX table suggestion

```latex
\begin{tabular}{llccccc}
\toprule
Method & Decoder & Train \% & mPQ & bPQ & F1\_det & Type Acc. \\
\midrule
Frozen CellViT & frozen & 0.00 & 0.003 & 0.447 & 0.817 & 0.022 \\
FullFT & all & 100.00 & 0.309 & 0.515 & 0.829 & 0.781 \\
LoRA+AF & heads\_only & 1.12 & 0.294 & 0.504 & 0.835 & 0.783 \\
LoRA+AF & last\_stage & 1.17 & 0.296 & 0.503 & 0.826 & 0.777 \\
LoRA+AF & conv\_adapters & 1.19 & 0.288 & 0.493 & 0.829 & 0.773 \\
VeRA+AF & heads\_only & 0.95 & 0.287 & 0.493 & 0.833 & 0.770 \\
VeRA+AF & conv\_adapters & 1.02 & 0.285 & 0.492 & 0.829 & 0.776 \\
\bottomrule
\end{tabular}
```
