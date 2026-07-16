# Slide-wise / region-wise dominance report

## Executive summary

- Aggregated 18 completed type-assignment runs from 84 slide/region rows.
- Using the dominance rule `patch_fraction > 0.70` or `matched_fraction > 0.70`, 2 run(s) are slide-dominated.
- Largest patch share: Ovary FullFT seed42 is dominated by ovary_s1 (0.887).
- Largest matched-nuclei share: Ovary FullFT seed42 is dominated by ovary_s1 (0.979).
- All available slide summaries include non-empty slide/region IDs.

## KLT runs

| Run | n | Patch dominant | Matched dominant | mPQ min/mean/max | bPQ min/mean/max | F1 det min/mean/max | Type acc min/mean/max | Macro F1 min/mean/max | mPQ CV | Acc CV | Dominated |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| KLT Frozen CellViT seed42 | 6 | tonsil_s1 (0.230) | tonsil_s0 (0.453) | 0.000/0.002/0.008 | 0.305/0.440/0.536 | 0.769/0.812/0.853 | 0.002/0.015/0.059 | 0.001/0.006/0.024 | 1.196 | 1.364 | no |
| KLT FullFT seed42 | 6 | tonsil_s1 (0.230) | tonsil_s0 (0.438) | 0.179/0.305/0.405 | 0.321/0.511/0.605 | 0.775/0.845/0.906 | 0.630/0.764/0.855 | 0.335/0.469/0.520 | 0.259 | 0.098 | no |
| KLT FullFT seed43 | 6 | tonsil_s1 (0.230) | tonsil_s0 (0.439) | 0.189/0.310/0.428 | 0.337/0.524/0.630 | 0.777/0.847/0.906 | 0.630/0.762/0.860 | 0.334/0.466/0.517 | 0.276 | 0.101 | no |
| KLT LoRA+AF heads_only seed42 | 6 | tonsil_s1 (0.230) | tonsil_s0 (0.442) | 0.183/0.297/0.389 | 0.327/0.511/0.602 | 0.777/0.851/0.910 | 0.641/0.771/0.844 | 0.306/0.414/0.445 | 0.242 | 0.084 | no |
| KLT LoRA+AF heads_only seed43 | 6 | tonsil_s1 (0.230) | tonsil_s0 (0.442) | 0.175/0.289/0.381 | 0.309/0.504/0.594 | 0.776/0.852/0.909 | 0.626/0.764/0.844 | 0.301/0.415/0.469 | 0.257 | 0.092 | no |
| KLT LoRA+AF last_stage seed42 | 6 | tonsil_s1 (0.230) | tonsil_s0 (0.445) | 0.179/0.298/0.395 | 0.325/0.511/0.600 | 0.768/0.846/0.907 | 0.635/0.760/0.845 | 0.328/0.459/0.525 | 0.250 | 0.090 | no |
| KLT LoRA+AF last_stage seed43 | 6 | tonsil_s1 (0.230) | tonsil_s0 (0.446) | 0.173/0.291/0.391 | 0.304/0.501/0.596 | 0.762/0.843/0.908 | 0.623/0.752/0.848 | 0.336/0.451/0.519 | 0.277 | 0.099 | no |
| KLT LoRA+AF conv_adapters seed42 | 6 | tonsil_s1 (0.230) | tonsil_s0 (0.443) | 0.160/0.286/0.383 | 0.294/0.493/0.582 | 0.771/0.846/0.908 | 0.630/0.751/0.841 | 0.329/0.451/0.509 | 0.269 | 0.096 | no |
| KLT LoRA+AF conv_adapters seed43 | 6 | tonsil_s1 (0.230) | tonsil_s0 (0.441) | 0.174/0.289/0.391 | 0.314/0.501/0.597 | 0.775/0.846/0.907 | 0.630/0.754/0.846 | 0.333/0.447/0.514 | 0.265 | 0.098 | no |
| KLT VeRA+AF conv_adapters seed42 | 6 | tonsil_s1 (0.230) | tonsil_s0 (0.441) | 0.163/0.283/0.377 | 0.300/0.495/0.583 | 0.780/0.848/0.906 | 0.629/0.748/0.845 | 0.330/0.441/0.509 | 0.270 | 0.097 | no |
| KLT VeRA+AF conv_adapters seed43 | 6 | tonsil_s1 (0.230) | tonsil_s0 (0.441) | 0.168/0.285/0.376 | 0.296/0.497/0.584 | 0.769/0.844/0.906 | 0.629/0.757/0.844 | 0.338/0.451/0.529 | 0.261 | 0.092 | no |
| KLT VeRA+AF heads_only seed43 | 6 | tonsil_s1 (0.230) | tonsil_s0 (0.440) | 0.165/0.287/0.385 | 0.298/0.497/0.586 | 0.777/0.849/0.910 | 0.627/0.760/0.845 | 0.302/0.412/0.447 | 0.266 | 0.092 | no |

## Tissue-specific Liver/Kidney/Ovary runs

| Run | n | Patch dominant | Matched dominant | mPQ min/mean/max | bPQ min/mean/max | F1 det min/mean/max | Type acc min/mean/max | Macro F1 min/mean/max | mPQ CV | Acc CV | Dominated |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Liver FullFT seed42 | 2 | liver_s0 (0.624) | liver_s0 (0.580) | 0.409/0.412/0.414 | 0.588/0.594/0.599 | 0.860/0.881/0.903 | 0.821/0.839/0.858 | 0.497/0.518/0.540 | 0.006 | 0.022 | no |
| Liver LoRA+AF heads_only seed42 | 2 | liver_s0 (0.624) | liver_s0 (0.583) | 0.390/0.392/0.394 | 0.578/0.586/0.594 | 0.871/0.890/0.909 | 0.811/0.830/0.849 | 0.449/0.453/0.456 | 0.005 | 0.023 | no |
| Kidney FullFT seed42 | 2 | kidney_s1 (0.572) | kidney_s1 (0.669) | 0.246/0.258/0.269 | 0.401/0.483/0.565 | 0.820/0.835/0.849 | 0.625/0.683/0.742 | 0.311/0.383/0.455 | 0.045 | 0.086 | no |
| Kidney LoRA+AF heads_only seed42 | 2 | kidney_s1 (0.572) | kidney_s1 (0.665) | 0.211/0.228/0.245 | 0.372/0.455/0.537 | 0.816/0.835/0.853 | 0.614/0.680/0.746 | 0.255/0.334/0.414 | 0.075 | 0.097 | no |
| Ovary FullFT seed42 | 2 | ovary_s1 (0.887) | ovary_s1 (0.979) | 0.103/0.225/0.347 | 0.238/0.402/0.566 | 0.640/0.738/0.837 | 0.539/0.677/0.815 | 0.384/0.449/0.514 | 0.543 | 0.204 | yes |
| Ovary LoRA+AF heads_only seed42 | 2 | ovary_s1 (0.887) | ovary_s1 (0.977) | 0.113/0.218/0.324 | 0.252/0.407/0.561 | 0.671/0.759/0.847 | 0.498/0.647/0.795 | 0.339/0.415/0.492 | 0.483 | 0.229 | yes |

## Dominance warnings

- Ovary FullFT seed42: dominant patch slide/region ovary_s1 (0.887); dominant matched slide/region ovary_s1 (0.979). This is a two-slide/region run.
- Ovary LoRA+AF heads_only seed42: dominant patch slide/region ovary_s1 (0.887); dominant matched slide/region ovary_s1 (0.977). This is a two-slide/region run.

## Two-slide tissue-specific context

- Liver FullFT seed42: patch dominance liver_s0 (0.624); matched-nuclei dominance liver_s0 (0.580).
- Liver LoRA+AF heads_only seed42: patch dominance liver_s0 (0.624); matched-nuclei dominance liver_s0 (0.583).
- Kidney FullFT seed42: patch dominance kidney_s1 (0.572); matched-nuclei dominance kidney_s1 (0.669).
- Kidney LoRA+AF heads_only seed42: patch dominance kidney_s1 (0.572); matched-nuclei dominance kidney_s1 (0.665).
- Ovary FullFT seed42: patch dominance ovary_s1 (0.887); matched-nuclei dominance ovary_s1 (0.979).
- Ovary LoRA+AF heads_only seed42: patch dominance ovary_s1 (0.887); matched-nuclei dominance ovary_s1 (0.977).

## Recommendation for the paper

Suggested limitation sentence: "Slide-wise analysis showed that no KLT run exceeded the pre-specified dominance threshold, whereas tissue-specific evaluations used only two test slides per tissue and the ovary runs were dominated by one slide; therefore aggregate tissue-specific metrics should be reported with per-slide dominance diagnostics rather than treated as broad population estimates."

Suggested table addition: include the number of test slides/regions, the dominant patch fraction, the dominant matched-nuclei fraction, and the min/mean/max per-slide mPQ and type accuracy alongside each aggregate final-inference result.

## Files

- Summary CSV: `reports/type_assignment/slide_dominance_summary.csv`
- Tissue-specific barplot: `reports/type_assignment/slide_dominance_barplot.png`
