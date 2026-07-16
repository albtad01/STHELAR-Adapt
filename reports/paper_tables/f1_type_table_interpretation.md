# F1_type table interpretation

LoRA+AF heads remains the most conservative default after replacing Type Acc. with matched-nuclei macro type-F1: it has the best PEFT F1_det (0.835) and stays close to the best PEFT mPQ (0.294 versus 0.296), but it is no longer the strongest row for F1_type.

LoRA+AF last_stage has stronger F1_type (0.654 versus 0.578 for heads) because it tunes the final decoder stage rather than only the 1x1 heads, giving the type branch more capacity to rebalance minority and heterogeneous classes, including Other.

Careful paper sentence: LoRA+AF heads is the best default when preserving detection and mPQ is the main constraint, whereas LoRA+AF last_stage trades a small detection drop for substantially better matched-nuclei macro type-F1.

F1_type is preferred over type accuracy under class imbalance because macro-F1 gives each present class comparable weight, while type accuracy can be dominated by abundant matched nuclei from the majority classes.

Supplementary note: Type accuracy is still useful as a matched-nuclei sanity check, but the main tables use F1_type following Félicie's recommendation. Melanocyte support was checked on the tissue-specific matched sets: it is absent in all non-Skin tissues and present only in Skin, so F1_type excludes Melanocyte outside Skin.

For the KLT ablation table with uncertainty, two-seed entries are reported as the seed mean plus/minus half the seed range. Single-seed rows use an imputed uncertainty equal to the median two-seed half-range across KLT runs, computed separately for each metric; these intervals are provisional until additional seeds are run.

The KLT final-head-only baseline trains only the existing NP/HV/NT 1x1 heads (<0.01% trainable parameters). It improves substantially over the fully frozen baseline (mPQ 0.204 and F1_type 0.430), but remains below the selected adapter-based PEFT strategies.

The Tonsil FullFT resume completed successfully and is now included in the tissue-specific table. Tonsil reaches FullFT mPQ 0.2498; the selected PEFT run reaches mPQ 0.2346, corresponding to 93.89% mPQ recovery, with PEFT F1_det 0.8226 and PEFT F1_type 0.5955.

Key KLT values:
- FullFT reference: mPQ 0.309, F1_det 0.829, F1_type 0.666.
- Best PEFT mPQ: \LoRA{}+\AF{} / last stage at 0.296.
- Best PEFT F1_det: \LoRA{}+\AF{} / heads at 0.835.
- Best PEFT F1_type: \LoRA{}+\AF{} / last stage at 0.654.
