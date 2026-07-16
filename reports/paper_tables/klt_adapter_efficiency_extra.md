# KLT Adapter Efficiency Probes

Two additional KLT runs were launched on A100:

- `1212868`: AdaptFormer red16 + final NP/HV/NT heads + limited last-stage decoder tuning.
- `1212870`: AdaptFormer red16 + final NP/HV/NT heads, decoder body frozen.
- `1213008`: LoRA+AdaptFormer last-stage seed44. This is the explicit LoRA+AF heads+last-stage confirmation run; in code, `decoder_train_scope: last_stage` already includes final NP/HV/NT heads.
- `1213009`: LoRA+AdaptFormer with NP/HV final heads and the full nuclei-type decoder branch trainable.
- `1214255`: LoRA+AdaptFormer with NT `decoder0_header.1.*` trainable plus NP/HV/NT final heads. The broader `1214248` NT-header job was cancelled before running.

Current recommendation before these two runs finish:

- LoRA+AdaptFormer heads remains the deployment-oriented default because it keeps the decoder body frozen and retains high mPQ/F1 detection with 1.12% trainable parameters.
- LoRA+AdaptFormer last-stage is currently the best-performing PEFT variant for type assignment: it gives similar mPQ and much stronger matched-nuclei macro type-F1, at only 1.17% trainable parameters.
- AdaptFormer+heads+last-stage is the key efficiency test. If it approaches LoRA+AdaptFormer last-stage type-F1 while using about 0.98% trainable parameters, it could become the cleaner efficient default.
- The NP/HV-heads + NT-all run is a classification-focused probe: it spends more parameters than heads/last-stage, but it tests whether most of the type-F1 gap can be closed by specializing the nuclei-type branch while keeping NP/HV bodies and the encoder base frozen.
- The NT-header-1 + final-heads run is the more surgical version of that idea: it adds only the pre-final NT `decoder0_header.1` Conv2DBlock, not the full NT branch and not the first NT header block.
- For now, describe LoRA+AdaptFormer heads as deployment-oriented and LoRA+AdaptFormer last-stage as the best current PEFT trade-off for classification-sensitive analyses.

Trainable parameter checks before launch:

- AdaptFormer+heads: 6,598,059 trainable parameters out of 706,333,675 total, 0.934128%.
- AdaptFormer+heads+last-stage: 6,950,571 trainable parameters out of 706,333,675 total, 0.984035%.
- Existing LoRA+AdaptFormer last-stage already includes the final NP/HV/NT decoder heads, so no duplicate LoRA+AF heads+last-stage job was launched.
- LoRA+AdaptFormer NT-header-1 + final-heads: 7,945,835 trainable parameters out of 707,644,395 total, 1.122857%.
- LoRA+AdaptFormer NP/HV-heads + NT-all: 22,986,219 trainable parameters out of 707,644,395 total, 3.248273%.
