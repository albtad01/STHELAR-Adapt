# STHELAR ablation trainable-component table

Generated on 2026-06-29 from the trainable-parameter reports in `run/**/logs.log` and the run catalog in `reports/peft_strategy_parameter_statistics.md`.

Counts are representative for CellViT-SAM-H x40 STHELAR runs. Tiny differences of a few hundred parameters can appear between 5-class, 9-class, single-tissue, and multitissue configs because the classifier/tissue heads change. Values are shown in millions (`M`).

## What is actually trained?

| Mode | Seen in tested runs | Total | Train. | Ratio | Enc. base | LoRA | VeRA | AdaptF. | Dec. conv | Shared dec. | NP | HV | NT |
|---|---|---:|---:|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Frozen | KLT, BPS/multitissue, Tonsil | 699.737M | 0 | 0.00% | frozen | -- | -- | -- | -- | frozen | frozen | frozen | frozen |
| Decoder all, encoder frozen | BPS/multitissue, Tonsil | 699.737M | 62.711M | 8.96% | frozen | -- | -- | -- | -- | 17.477M | 15.078M | 15.078M | 15.078M |
| NTOnly | BPS/multitissue, Tonsil | 699.737M | 15.078M | 2.15% | frozen | -- | -- | -- | -- | frozen | frozen | frozen | 15.078M |
| LoRA r4, decoder all | BPS/multitissue | 700.392M | 63.366M | 9.05% | frozen | 0.655M | -- | -- | -- | 17.477M | 15.078M | 15.078M | 15.078M |
| LoRA r8, decoder all | BPS/multitissue, Liver, Tonsil | 701.047M | 64.021M | 9.13% | frozen | 1.311M | -- | -- | -- | 17.477M | 15.078M | 15.078M | 15.078M |
| LoRA r16, decoder all | BPS/multitissue | 702.358M | 65.332M | 9.30% | frozen | 2.621M | -- | -- | -- | 17.477M | 15.078M | 15.078M | 15.078M |
| LoRA+NTOnly r4 | BPS/multitissue, Liver, Tonsil | 700.392M | 15.734M | 2.25% | frozen | 0.655M | -- | -- | -- | frozen | frozen | frozen | 15.078M |
| LoRA+NTOnly r8 | BPS/multitissue | 701.048M | 16.389M | 2.34% | frozen | 1.311M | -- | -- | -- | frozen | frozen | frozen | 15.078M |
| LoRA+NTOnly r16 | BPS/multitissue | 702.358M | 17.700M | 2.52% | frozen | 2.621M | -- | -- | -- | frozen | frozen | frozen | 15.078M |
| AdaptFormer red8, decoder all | Tonsil | 712.890M | 75.864M | 10.64% | frozen | -- | -- | 13.153M | -- | 17.477M | 15.078M | 15.078M | 15.078M |
| AdaptFormer red16, decoder all | BPS/multitissue, Liver, Tonsil | 706.334M | 69.308M | 9.81% | frozen | -- | -- | 6.597M | -- | 17.477M | 15.078M | 15.078M | 15.078M |
| LoRA+AdaptFormer r4/red16, decoder all | Liver, Tonsil | 706.989M | 69.963M | 9.90% | frozen | 0.655M | -- | 6.597M | -- | 17.477M | 15.078M | 15.078M | 15.078M |
| LoRA+AdaptFormer r8/red16, decoder all | Liver | 707.644M | 70.618M | 9.98% | frozen | 1.311M | -- | 6.597M | -- | 17.477M | 15.078M | 15.078M | 15.078M |
| LoRA+AdaptFormer+NTOnly r8/red16 | Liver, Tonsil | 707.644M | 22.986M | 3.25% | frozen | 1.311M | -- | 6.597M | -- | frozen | frozen | frozen | 15.078M |
| LoRA r8, decoder frozen | KLT | 701.047M | 1.311M | 0.19% | frozen | 1.311M | -- | -- | -- | frozen | frozen | frozen | frozen |
| AdaptFormer red16, decoder frozen | KLT | 706.334M | 6.597M | 0.93% | frozen | -- | -- | 6.597M | -- | frozen | frozen | frozen | frozen |
| VeRA r16, decoder frozen | KLT | 699.819M | 0.083M | 0.01% | frozen | -- | 0.083M | -- | -- | frozen | frozen | frozen | frozen |
| Frozen encoder + decoder conv adapters | KLT, BPS/multitissue | 700.289M | 0.553M | 0.08% | frozen | -- | -- | -- | 0.552M | frozen | <0.001M | <0.001M | <0.001M |
| LoRA r8 + decoder conv adapters | BPS/multitissue | 701.600M | 1.864M | 0.27% | frozen | 1.311M | -- | -- | 0.552M | frozen | <0.001M | <0.001M | <0.001M |
| LoRA+AdaptFormer r8/red16, heads only | KLT, BPS/multitissue, Liver, Kidney, Ovary, Tonsil | 707.644M | 7.909M | 1.12% | frozen | 1.311M | -- | 6.597M | -- | frozen | <0.001M | <0.001M | <0.001M |
| LoRA+AdaptFormer r8/red16, last stage | KLT, BPS/multitissue, Liver, Tonsil | 707.644M | 8.261M | 1.17% | frozen | 1.311M | -- | 6.597M | -- | 0.020M | 0.111M | 0.111M | 0.111M |
| LoRA+AdaptFormer r8/red16, conv adapters | KLT, BPS/multitissue, Liver, Tonsil | 708.196M | 8.461M | 1.19% | frozen | 1.311M | -- | 6.597M | 0.552M | frozen | <0.001M | <0.001M | <0.001M |
| VeRA r8, decoder all | Liver | 699.819M | 62.793M | 8.97% | frozen | -- | 0.082M | -- | -- | 17.477M | 15.078M | 15.078M | 15.078M |
| VeRA+AdaptFormer r8/red16, decoder all | Liver | 706.416M | 69.390M | 9.82% | frozen | -- | 0.082M | 6.597M | -- | 17.477M | 15.078M | 15.078M | 15.078M |
| VeRA+AdaptFormer r8/red16, conv adapters | BPS/multitissue, Liver, Tonsil | 706.968M | 7.233M | 1.02% | frozen | -- | 0.082M | 6.597M | 0.552M | frozen | <0.001M | <0.001M | <0.001M |
| VeRA+AdaptFormer r16/red16, heads only | KLT | 706.417M | 6.681M | 0.95% | frozen | -- | 0.083M | 6.597M | -- | frozen | <0.001M | <0.001M | <0.001M |
| VeRA+AdaptFormer r16/red16, conv adapters | KLT, BPS/multitissue, Liver, Tonsil | 706.969M | 7.233M | 1.02% | frozen | -- | 0.083M | 6.597M | 0.552M | frozen | <0.001M | <0.001M | <0.001M |
| FullFT | KLT, BPS/multitissue, Liver, Kidney, Ovary, Tonsil | 699.737M | 699.737M | 100.00% | train | -- | -- | -- | -- | 17.477M | 15.078M | 15.078M | 15.078M |

## Slide-ready LaTeX

This is a compact Beamer version. It keeps the scientific message of the wider table while fitting a slide better.

```latex
%================================================
\begin{frame}{What is actually trained?}
\tiny
\setlength{\tabcolsep}{2.0pt}
\renewcommand{\arraystretch}{1.10}
\begin{table}
\centering
\resizebox{0.99\textwidth}{!}{%
\begin{tabular}{lrrrrrrrrrrr}
\toprule
\textbf{Mode} &
\textbf{Total} &
\textbf{Train.} &
\textbf{Ratio} &
\textbf{Enc. base} &
\textbf{LoRA} &
\textbf{VeRA} &
\textbf{AdaptF.} &
\textbf{Dec. conv} &
\textbf{Shared dec.} &
\textbf{NP/HV} &
\textbf{NT} \\
\midrule
Frozen &
699.737M & 0 & 0.00\% &
frozen & -- & -- & -- & -- & frozen & frozen & frozen \\

Decoder all, encoder frozen &
699.737M & 62.711M & 8.96\% &
frozen & -- & -- & -- & -- & 17.477M & 15.078M & 15.078M \\

NTOnly &
699.737M & 15.078M & 2.15\% &
frozen & -- & -- & -- & -- & frozen & frozen & 15.078M \\

LoRA r8, decoder all &
701.047M & 64.021M & 9.13\% &
frozen & 1.311M & -- & -- & -- & 17.477M & 15.078M & 15.078M \\

AdaptFormer red16, decoder all &
706.334M & 69.308M & 9.81\% &
frozen & -- & -- & 6.597M & -- & 17.477M & 15.078M & 15.078M \\

LoRA+AdaptFormer r8/red16, decoder all &
707.644M & 70.618M & 9.98\% &
frozen & 1.311M & -- & 6.597M & -- & 17.477M & 15.078M & 15.078M \\

LoRA+AdaptFormer+NTOnly r8/red16 &
707.644M & 22.986M & 3.25\% &
frozen & 1.311M & -- & 6.597M & -- & frozen & frozen & 15.078M \\

LoRA+AdaptFormer r8/red16, heads only &
707.644M & 7.909M & 1.12\% &
frozen & 1.311M & -- & 6.597M & -- & frozen & $<0.001$M & $<0.001$M \\

LoRA+AdaptFormer r8/red16, last stage &
707.644M & 8.261M & 1.17\% &
frozen & 1.311M & -- & 6.597M & -- & 0.020M & 0.111M & 0.111M \\

LoRA+AdaptFormer r8/red16, conv adapters &
708.196M & 8.461M & 1.19\% &
frozen & 1.311M & -- & 6.597M & 0.552M & frozen & $<0.001$M & $<0.001$M \\

VeRA+AdaptFormer r16/red16, heads only &
706.417M & 6.681M & 0.95\% &
frozen & -- & 0.083M & 6.597M & -- & frozen & $<0.001$M & $<0.001$M \\

VeRA+AdaptFormer r16/red16, conv adapters &
706.969M & 7.233M & 1.02\% &
frozen & -- & 0.083M & 6.597M & 0.552M & frozen & $<0.001$M & $<0.001$M \\

FullFT &
699.737M & 699.737M & 100\% &
train & -- & -- & -- & -- & 17.477M & 15.078M & 15.078M \\
\bottomrule
\end{tabular}%
}
\end{table}

\vspace{0.05cm}
\begin{block}{Important wording}
PEFT here means \textbf{frozen SAM-H encoder base plus trainable adapter modules}, with decoder capacity depending on the ablation scope. In `decoder all` runs the decoder branches are fine-tuned; in `heads only` runs only the final NP/HV/NT heads and classifier are trainable; in `conv adapters` runs small decoder residual adapters are trained. FullFT is the only mode where the SAM-H encoder base is trainable.
\end{block}
\end{frame}
```

## Notes for wording

- `decoder all` is the historical broad PEFT setting: encoder base frozen, adapters trainable if present, and the full shared/NP/HV/NT decoder branches trainable.
- `NTOnly` trains the nuclei-type branch only; NP/HV and shared decoder stay frozen.
- `heads only` trains encoder PEFT modules plus the final NP/HV/NT prediction heads/classifier. These final head parameters are tiny at the `M` scale, so they appear as `<0.001M`.
- `last stage` trains encoder PEFT modules plus the late decoder stage and final heads.
- `conv adapters` trains encoder PEFT modules plus small residual decoder convolution adapters and final heads.
- `VeRA` rows are much smaller than LoRA rows because VeRA uses shared random low-rank projections with trainable scaling vectors.
