# Nine-tissue reciprocal complete-slide Selected PEFT matrix

Both backbones are complete for all 9 tissues × Fold A/B at seed42. Support is matched-plus-unmatched true foreground support from canonical TEST inference. Counts are patch-level nucleus occurrences, not unique biological cells.

| Backbone | Tissue | Fold | TEST patches | True support | bPQ | mPQ | F1det | F1type | Per-class F1 / support |
|---|---|:---:|---:|---:|---:|---:|---:|---:|---|
| CellViT-SAM-H | Breast | A | 49,569 | 820,290 | 0.411262 | 0.238573 | 0.812884 | 0.500014 | Immune 0.509/193,476; Stromal 0.624/205,990; Epithelial 0.838/395,725; Melanocyte nan/0; Other 0.030/25,099 |
| CellViT-SAM-H | Breast | B | 45,773 | 1,251,232 | 0.452459 | 0.277942 | 0.831757 | 0.534307 | Immune 0.692/238,925; Stromal 0.493/234,604; Epithelial 0.930/762,687; Melanocyte nan/0; Other 0.023/15,016 |
| CellViT-SAM-H | Colon | A | 11,903 | 752,353 | 0.284721 | 0.149549 | 0.750952 | 0.426503 | Immune 0.569/195,875; Stromal 0.276/30,566; Epithelial 0.850/511,629; Melanocyte nan/0; Other 0.011/14,283 |
| CellViT-SAM-H | Colon | B | 18,464 | 1,183,780 | 0.286509 | 0.119909 | 0.744118 | 0.386678 | Immune 0.384/208,114; Stromal 0.296/368,241; Epithelial 0.866/590,393; Melanocyte nan/0; Other 0.000/17,032 |
| CellViT-SAM-H | Kidney | A | 4,449 | 104,835 | 0.549529 | 0.090364 | 0.847800 | 0.149249 | Immune 0.070/58,422; Stromal 0.403/26,644; Epithelial 0.121/17,669; Melanocyte nan/0; Other 0.004/2,100 |
| CellViT-SAM-H | Kidney | B | 6,723 | 191,643 | 0.609419 | 0.052900 | 0.900830 | 0.105982 | Immune 0.172/19,732; Stromal 0.156/55,116; Epithelial 0.095/106,874; Melanocyte nan/0; Other 0.000/9,921 |
| CellViT-SAM-H | Liver | A | 9,166 | 315,935 | 0.480070 | 0.171506 | 0.843156 | 0.264902 | Immune 0.017/66,191; Stromal 0.446/72,684; Epithelial 0.596/175,936; Melanocyte nan/0; Other 0.000/1,124 |
| CellViT-SAM-H | Liver | B | 20,427 | 480,093 | 0.521889 | 0.244555 | 0.882647 | 0.403020 | Immune 0.329/106,478; Stromal 0.465/97,143; Epithelial 0.818/271,122; Melanocyte nan/0; Other 0.000/5,350 |
| CellViT-SAM-H | Lung | A | 20,621 | 554,155 | 0.508304 | 0.193408 | 0.842963 | 0.404211 | Immune 0.622/194,173; Stromal 0.639/174,761; Epithelial 0.355/145,102; Melanocyte nan/0; Other 0.001/40,119 |
| CellViT-SAM-H | Lung | B | 10,938 | 320,398 | 0.563651 | 0.251397 | 0.861942 | 0.470082 | Immune 0.726/145,454; Stromal 0.434/71,716; Epithelial 0.719/68,294; Melanocyte nan/0; Other 0.001/34,934 |
| CellViT-SAM-H | Ovary | A | 25,498 | 815,470 | 0.448760 | 0.222645 | 0.834355 | 0.505403 | Immune 0.362/90,259; Stromal 0.775/304,120; Epithelial 0.884/360,874; Melanocyte nan/0; Other 0.000/60,217 |
| CellViT-SAM-H | Ovary | B | 10,039 | 498,877 | 0.387430 | 0.154828 | 0.832848 | 0.480599 | Immune 0.345/104,483; Stromal 0.647/61,888; Epithelial 0.713/310,152; Melanocyte nan/0; Other 0.218/22,354 |
| CellViT-SAM-H | Pancreatic | A | 12,647 | 469,142 | 0.377441 | 0.123931 | 0.733776 | 0.278763 | Immune 0.614/155,985; Stromal 0.381/82,287; Epithelial 0.119/117,706; Melanocyte nan/0; Other 0.002/113,164 |
| CellViT-SAM-H | Pancreatic | B | 25,526 | 440,021 | 0.389166 | 0.174534 | 0.725043 | 0.416725 | Immune 0.574/138,877; Stromal 0.388/73,507; Epithelial 0.695/199,217; Melanocyte nan/0; Other 0.010/28,420 |
| CellViT-SAM-H | Skin | A | 12,790 | 199,847 | 0.440098 | 0.134163 | 0.743151 | 0.242519 | Immune 0.455/77,761; Stromal 0.283/38,319; Epithelial 0.472/10,718; Melanocyte 0.001/72,962; Other 0.002/87 |
| CellViT-SAM-H | Skin | B | 11,209 | 176,792 | 0.213661 | 0.071668 | 0.662776 | 0.293581 | Immune 0.176/7,656; Stromal 0.497/47,505; Epithelial 0.761/60,109; Melanocyte 0.034/5,534; Other 0.000/55,988 |
| CellViT-SAM-H | Tonsil | A | 21,083 | 1,716,308 | 0.452244 | 0.215840 | 0.838259 | 0.487866 | Immune 0.810/1,115,096; Stromal 0.506/341,054; Epithelial 0.636/177,099; Melanocyte nan/0; Other 0.000/83,059 |
| CellViT-SAM-H | Tonsil | B | 23,067 | 2,662,977 | 0.577060 | 0.260302 | 0.855994 | 0.510764 | Immune 0.861/1,855,942; Stromal 0.410/445,916; Epithelial 0.677/334,497; Melanocyte nan/0; Other 0.095/26,622 |
| CellViT-256 | Breast | A | 49,569 | 817,476 | 0.453654 | 0.236617 | 0.791804 | 0.463907 | Immune 0.426/192,844; Stromal 0.598/205,192; Epithelial 0.814/394,395; Melanocyte nan/0; Other 0.018/25,045 |
| CellViT-256 | Breast | B | 45,773 | 1,246,557 | 0.510126 | 0.302282 | 0.834920 | 0.522674 | Immune 0.653/238,576; Stromal 0.490/233,889; Epithelial 0.877/759,164; Melanocyte nan/0; Other 0.071/14,928 |
| CellViT-256 | Colon | A | 11,903 | 751,168 | 0.334596 | 0.115096 | 0.751288 | 0.330118 | Immune 0.476/195,544; Stromal 0.229/30,511; Epithelial 0.569/510,930; Melanocyte nan/0; Other 0.047/14,183 |
| CellViT-256 | Colon | B | 18,464 | 1,170,544 | 0.239509 | 0.107871 | 0.710519 | 0.472448 | Immune 0.435/205,560; Stromal 0.502/362,553; Epithelial 0.882/585,589; Melanocyte nan/0; Other 0.071/16,842 |
| CellViT-256 | Kidney | A | 4,449 | 104,053 | 0.468128 | 0.098889 | 0.833860 | 0.210434 | Immune 0.281/58,172; Stromal 0.345/26,332; Epithelial 0.190/17,483; Melanocyte nan/0; Other 0.025/2,066 |
| CellViT-256 | Kidney | B | 6,723 | 190,866 | 0.568279 | 0.096474 | 0.893834 | 0.172681 | Immune 0.120/19,709; Stromal 0.135/54,995; Epithelial 0.401/106,277; Melanocyte nan/0; Other 0.035/9,885 |
| CellViT-256 | Liver | A | 9,166 | 315,573 | 0.479362 | 0.073691 | 0.832494 | 0.113617 | Immune 0.006/66,172; Stromal 0.368/72,592; Epithelial 0.080/175,685; Melanocyte nan/0; Other 0.000/1,124 |
| CellViT-256 | Liver | B | 20,427 | 472,117 | 0.572147 | 0.251328 | 0.865266 | 0.396318 | Immune 0.430/104,556; Stromal 0.354/95,611; Epithelial 0.801/266,639; Melanocyte nan/0; Other 0.000/5,311 |
| CellViT-256 | Lung | A | 20,621 | 550,114 | 0.412533 | 0.148066 | 0.829681 | 0.404976 | Immune 0.572/193,054; Stromal 0.589/173,776; Epithelial 0.320/143,632; Melanocyte nan/0; Other 0.139/39,652 |
| CellViT-256 | Lung | B | 10,938 | 318,148 | 0.496273 | 0.216827 | 0.828202 | 0.454569 | Immune 0.702/144,815; Stromal 0.376/70,978; Epithelial 0.621/67,669; Melanocyte nan/0; Other 0.119/34,686 |
| CellViT-256 | Ovary | A | 25,498 | 813,312 | 0.412639 | 0.199412 | 0.784555 | 0.515850 | Immune 0.341/89,958; Stromal 0.751/303,337; Epithelial 0.845/360,012; Melanocyte nan/0; Other 0.126/60,005 |
| CellViT-256 | Ovary | B | 10,039 | 496,285 | 0.413871 | 0.147708 | 0.838534 | 0.426264 | Immune 0.357/104,163; Stromal 0.465/60,936; Epithelial 0.654/309,004; Melanocyte nan/0; Other 0.230/22,182 |
| CellViT-256 | Pancreatic | A | 12,647 | 466,409 | 0.350001 | 0.097050 | 0.736281 | 0.232810 | Immune 0.573/155,402; Stromal 0.284/81,687; Epithelial 0.067/117,095; Melanocyte nan/0; Other 0.006/112,225 |
| CellViT-256 | Pancreatic | B | 25,526 | 437,152 | 0.299766 | 0.106697 | 0.630323 | 0.315055 | Immune 0.452/138,213; Stromal 0.321/73,108; Epithelial 0.468/197,644; Melanocyte nan/0; Other 0.020/28,187 |
| CellViT-256 | Skin | A | 12,790 | 199,404 | 0.394440 | 0.083150 | 0.708724 | 0.130921 | Immune 0.193/77,698; Stromal 0.262/38,263; Epithelial 0.189/10,678; Melanocyte 0.009/72,678; Other 0.001/87 |
| CellViT-256 | Skin | B | 11,209 | 176,574 | 0.222075 | 0.030554 | 0.678200 | 0.150494 | Immune 0.016/7,633; Stromal 0.106/47,418; Epithelial 0.630/60,114; Melanocyte 0.000/5,532; Other 0.000/55,877 |
| CellViT-256 | Tonsil | A | 21,083 | 1,716,066 | 0.409102 | 0.176689 | 0.827759 | 0.458659 | Immune 0.777/1,114,971; Stromal 0.479/340,968; Epithelial 0.579/177,080; Melanocyte nan/0; Other 0.000/83,047 |
| CellViT-256 | Tonsil | B | 23,067 | 2,658,310 | 0.527360 | 0.229453 | 0.863211 | 0.499848 | Immune 0.839/1,852,833; Stromal 0.408/445,134; Epithelial 0.690/333,755; Melanocyte nan/0; Other 0.062/26,588 |

Matched FullFT controls cover all nine tissues for CellViT-SAM-H. CellViT-256 FullFT remains restricted to Kidney/Liver/Tonsil; no comparison is implied for its other six tissues.
