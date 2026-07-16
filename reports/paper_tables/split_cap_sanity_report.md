# Split/cap sanity report

## Executive summary

- Audited 10 5-class split manifests: KLT plus 9 tissue-specific datasets.
- `before_cap_patches` is counted from `patches_overview_sthelar40x.parquet`; `after_cap_patches` is `train + valid + test + discarded_by_margin`; and `after_spatial_margin_filter_patches` is `train + valid + test`.
- The cap limits maximum patch contribution per slide. It does not balance slides: slides below the cap keep their original counts, and slides above the cap are sampled down to the cap.
- KLT uses a 10,000 patch/slide cap and keeps 49,829 patches after the 128-pixel spatial margin (37,109 train, 8,067 valid, 4,653 test).
- Tissue-specific datasets with one slide above the 70.0% kept-patch threshold: Ovary:ovary_s1 (71.8% kept).

## KLT per-slide counts

| Slide | Before cap | After cap | After margin | Train | Valid | Test | Margin discard | Kept share |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| kidney_s0 | 6,948 | 6,948 | 6,723 | 5,543 | 778 | 402 | 225 | 13.5% |
| kidney_s1 | 4,831 | 4,831 | 4,449 | 3,262 | 650 | 537 | 382 | 8.9% |
| liver_s0 | 20,942 | 10,000 | 9,756 | 7,078 | 1,902 | 776 | 244 | 19.6% |
| liver_s1 | 9,642 | 9,642 | 9,166 | 6,553 | 1,621 | 992 | 476 | 18.4% |
| tonsil_s0 | 23,402 | 10,000 | 9,856 | 7,239 | 1,743 | 874 | 144 | 19.8% |
| tonsil_s1 | 21,351 | 10,000 | 9,879 | 7,434 | 1,373 | 1,072 | 121 | 19.8% |

## KLT split totals

| Before cap | After cap | After margin | Train | Valid | Test | Margin discard |
| --- | --- | --- | --- | --- | --- | --- |
| 87,116 | 51,421 | 49,829 | 37,109 | 8,067 | 4,653 | 1,592 |

## Tissue-specific dominance

| Tissue | Dominant slide | Before cap | After cap share | Kept share | Dominates? |
| --- | --- | --- | --- | --- | --- |
| Ovary | ovary_s1 | 26,006 | 71.8% | 71.8% | yes |
| Liver | liver_s0 | 20,942 | 68.5% | 69.0% | no |
| Pancreatic | pancreatic_s0 | 26,135 | 67.1% | 66.9% | no |
| Lung | lung_s3 | 20,974 | 65.0% | 65.3% | no |
| Colon | colon_s1 | 18,870 | 60.5% | 60.8% | no |
| Kidney | kidney_s0 | 6,948 | 59.0% | 60.2% | no |
| Skin | skin_s2 | 13,234 | 53.2% | 53.3% | no |
| Tonsil | tonsil_s0 | 23,402 | 52.3% | 52.2% | no |
| Breast | breast_s1 | 120,000 | 52.0% | 52.0% | no |

## Tissue-specific split totals

| Tissue | Before cap | After cap | After margin | Train | Valid | Test |
| --- | --- | --- | --- | --- | --- | --- |
| Breast | 166,109 | 96,109 | 95,342 | 71,875 | 14,024 | 9,443 |
| Colon | 31,205 | 31,205 | 30,367 | 23,126 | 3,999 | 3,242 |
| Kidney | 11,779 | 11,779 | 11,172 | 8,805 | 1,428 | 939 |
| Liver | 30,584 | 30,584 | 29,593 | 21,301 | 5,653 | 2,639 |
| Lung | 32,272 | 32,272 | 31,559 | 24,929 | 4,162 | 2,468 |
| Ovary | 36,236 | 36,236 | 35,537 | 28,131 | 5,026 | 2,380 |
| Pancreatic | 38,924 | 38,924 | 38,173 | 30,787 | 4,780 | 2,606 |
| Skin | 24,882 | 24,882 | 23,999 | 18,114 | 3,755 | 2,130 |
| Tonsil | 44,753 | 44,753 | 44,150 | 32,932 | 6,900 | 4,318 |

## Notes for the paper

- The spatial split is within-slide for these datasets: every selected slide contributes train, validation, and test patches separated along the split axis with a 128-pixel boundary margin.
- For capped datasets, the cap should be described as a maximum per-slide contribution, not as class balancing or slide balancing.
- The CSV contains one row per slide plus an `ALL` total row per dataset for direct table construction.

## Files

- CSV: `reports/paper_tables/split_cap_sanity_table.csv`
