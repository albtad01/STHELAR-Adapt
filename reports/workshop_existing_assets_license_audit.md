# Final existing-assets license audit (post-remediation)

**Audit date:** 2026-09-04
**Scope:** public source/model release, reported experiments, current adapter packages, and the stated future releases of evaluation adapters, all-slides deployment adapters, and fine-tuning code. Ordinary Python dependencies are excluded. This is a provenance/compliance audit, not legal advice.

## Decision

**Verdict: YES for the remediated release candidate (non-commercial academic use).** Every material existing asset has recoverable terms compatible with the reported use. The root and Hugging Face `LICENSE` files now reproduce the current component-specific CellViT terms and full Apache License 2.0; matching `NOTICE` files preserve upstream attribution, disclose the CellViT modifications, identify the STHELAR-derived figure changes, and state that CellViT bases and PanNuke images are not redistributed. The release model card links these files and the CC BY 4.0 terms, records the STHELAR revision, and requires license/notice files in the selective download example. This conclusion becomes true of the public GitHub/Hugging Face releases only after these local files are published there.

The exact upstream CellViT source commit imported into this repository was not recorded, but this does not prevent identification of the applicable terms because the imported license and current official component license agree on the covered components and restrictions. The final main-manuscript source remains unavailable in the workspace, so the paper paragraph, checklist answer, and `hipt`/`pannuke` bibliography entries below must still be inserted in the actual manuscript; that manuscript-only step was not performed here.

## Material assets and authoritative terms

| Material asset | Exact identity and provenance | Authoritative license/terms | Compatibility, redistribution, and attribution | Paper citation status |
|---|---|---|---|---|
| **STHELAR 40x dataset** | Hugging Face `FelicieGS/STHELAR_40x`, audited revision `e32a8cdd50eff2d38e237f3729e9ac85bbb5203b`; dataset DOI `10.57967/hf/6008`; publication DOI `10.1038/s41597-026-06937-6`. [Exact dataset revision](https://huggingface.co/datasets/FelicieGS/STHELAR_40x/tree/e32a8cdd50eff2d38e237f3729e9ac85bbb5203b); [dataset card](https://huggingface.co/datasets/FelicieGS/STHELAR_40x). | **CC BY 4.0**. The official card declares `cc-by-4.0` and links the originating publication/data resources. [CC BY 4.0 terms](https://creativecommons.org/licenses/by/4.0/). | Experimental training/evaluation and adapter distribution are compatible. The updated model card and `NOTICE` give credit, link CC BY 4.0, identify the exact revision, and state that overlays/labels/layout were added to STHELAR image panels. Adapter tensors do not contain dataset images. | STHELAR is cited in repository documentation; insert the exact revision and CC BY 4.0 statement below in the final manuscript. |
| **CellViT source/framework** | Official repository [TIO-IKIM/CellViT](https://github.com/TIO-IKIM/CellViT). This project first imported the framework at local commit `a60133850673bcab85a5b8ccb155316de86944b7`; no upstream CellViT commit/tag is recorded, so the exact upstream source revision is **UNRESOLVED**. Audited STHELAR-Adapt/adapter metadata commit: `5dc3ae1dea3999c5c78863a28e2216cafafa55e6`. | The imported repository carries **Apache License 2.0 with Commons Clause v1.0**, no right to Sell, plus mandatory CellViT citation. The current upstream license distinguishes SAM-derived components (Apache 2.0), HIPT-derived components (Apache 2.0 + Commons Clause), and original CellViT training/inference/segmentation code (Apache 2.0 + Commons Clause and required citation). [Immutable component license](https://github.com/TIO-IKIM/CellViT/blob/5331d0207563b97e704e8d054df1eceb5483b70c/LICENSE); [current license](https://github.com/TIO-IKIM/CellViT/blob/main/LICENSE). | Non-commercial academic modification/use is compatible. The source and HF releases now include the full Apache text, Commons Clause conditions, component/source notices, a repository-wide modification statement, and mandatory CellViT citation. Commercial sale, hosted service, or paid support whose value substantially derives from covered code still needs permission. | The accessible appendix cites `cellvit`; use the final paragraph below. |
| **Official CellViT-SAM-H x40 composite checkpoint** | Official CellViT download ID `1MvRKNzDW2eHbQb5rAgTEp6s2zAXHixRV`; local/release identity `CellViT-SAM-H-x40.pth`, 2,799,315,941 bytes, SHA256 `b324c10fddb0f80f5ab03a0459453a4c4848866934daf63435b46749a6b278cf`. [Official checkpoint listing](https://github.com/TIO-IKIM/CellViT#inference); [official download](https://drive.google.com/uc?export=download&id=1MvRKNzDW2eHbQb5rAgTEp6s2zAXHixRV). No immutable CellViT release tag is attached to the Drive object. | Upstream lists the provided checkpoints under **Apache 2.0 with Commons Clause**. Within the composite file, current CellViT terms classify the SAM-derived component as Apache 2.0 and original CellViT components as Apache 2.0 + Commons Clause/citation; conservatively apply the more restrictive composite conditions to the checkpoint as a whole. | Academic use is compatible. Do not sell the composite checkpoint or a substantially derived service without permission; retain CellViT and SAM notices/citations. The base checkpoint is not present in the adapter release and users are directed to the official download, which materially reduces redistribution exposure but does not remove attribution/terms for the derivative framework and adapter documentation. | CellViT and SAM are cited in the accessible appendix; final bibliography unverified. Add Drive ID and SHA256. |
| **Segment Anything v1 / SAM-H ancestry** | CellViT-SAM-H uses a SAM-H encoder and this repository includes SAM-derived encoder code with Meta copyright headers. The exact original SAM-H blob used by CellViT is not recorded in local checkpoint metadata; the standard official ViT-H link is listed by the [SAM repository](https://github.com/facebookresearch/segment-anything#model-checkpoints), but equivalence should not be asserted without upstream provenance. SA-1B was not downloaded or used by this project. | **Apache License 2.0** for SAM model/code. [Official SAM license](https://github.com/facebookresearch/segment-anything/blob/main/LICENSE). SA-1B's separate research license does not apply merely because a SAM-derived encoder is used. | Research use and redistribution of modified SAM-derived code/weights are permitted subject to Apache 2.0: supply the license, retain notices, mark modified files, and carry any NOTICE attributions. No separate academic citation is imposed by Apache, but SAM citation is scientifically appropriate and is requested by the upstream README. | `sam` is cited in the accessible appendix. |
| **Official CellViT-256 x40 composite checkpoint** | Official CellViT download ID `1tVYAapUo1Xt8QgCN22Ne1urbbCZkah8q`; local `CellViT-256-x40.pth`, 187,224,155 bytes, SHA256 `ee3986922fc500353db3d7692c566e19e1c694c8f10e47cf49dfd90160fc3b2b`; embedded architecture `CellViT256`, epoch 129, 439 tensors, PanNuke pretraining. [Official checkpoint listing](https://github.com/TIO-IKIM/CellViT#inference); [official download](https://drive.google.com/uc?export=download&id=1tVYAapUo1Xt8QgCN22Ne1urbbCZkah8q). | The composite is listed by CellViT under **Apache 2.0 with Commons Clause**. Current CellViT terms classify HIPT-derived weights/code as Apache 2.0 + Commons Clause and original CellViT code as Apache 2.0 + Commons Clause/citation. | Non-commercial academic use is compatible. Adapter-only release may point to the official base rather than redistribute it. Treat commercial distribution/use as permission-required; retain CellViT and HIPT notices and citations. | CellViT is cited in the accessible appendix, but CellViT-256/HIPT provenance is not explicit. Add checkpoint ID/SHA and cite HIPT when discussing this backbone. |
| **HIPT-256 and DINO ancestry of CellViT-256** | CellViT states that teacher models were used for ViT-256; HIPT identifies DINO as its pretraining framework. [Official HIPT repository](https://github.com/mahmoodlab/HIPT); [official DINO repository](https://github.com/facebookresearch/dino). These are indirect components embodied in the CellViT-256 base, not separately downloaded checkpoints in this project. | HIPT: **Apache 2.0 with Commons Clause**, no Sell, notice must accompany required attribution. [HIPT LICENSE](https://github.com/mahmoodlab/HIPT/blob/master/LICENSE). DINO: **Apache 2.0**. [DINO LICENSE](https://github.com/facebookresearch/dino/blob/main/LICENSE). | The reported academic use is compatible. If code or base weights containing these components are distributed, retain their notices/license texts. Adapter-only packages exclude frozen HIPT/DINO base parameters, but release documentation should identify the required base and ancestry. | Not verifiable in the final manuscript. Add HIPT (and DINO if its role is described), especially because the official HIPT README requests DINO/ViT citation when citing HIPT. |
| **PanNuke (indirect upstream checkpoint-training dataset)** | Both official CellViT x40 checkpoints were pretrained upstream on PanNuke; this project did not separately download, train on, evaluate on, or redistribute PanNuke. [CellViT checkpoint statement](https://github.com/TIO-IKIM/CellViT#inference); [original Warwick dataset URL](https://warwick.ac.uk/fac/cross_fac/tia/data/pannuke/); [RationAI dataset card](https://huggingface.co/datasets/RationAI/PanNuke). | The RationAI-maintained card declares **CC BY-NC-SA 4.0**. The original Warwick page could not be retrieved during this audit, and the PanNuke paper does not itself supply operative license text. The exact legal question whether dataset ShareAlike reaches learned model parameters is not resolved by those sources. | Non-commercial academic use is compatible under either the CellViT checkpoint grant or PanNuke's stated NC terms. Do not represent this audit as clearing commercial use of the CellViT-256/SAM-H checkpoints or derived weights. No PanNuke images are in the adapters. For public adapters, retain PanNuke attribution and use a conservative non-commercial posture unless counsel/upstream clarifies the model-weight question. | The accessible appendix mentions the PanNuke taxonomy without a citation on that sentence. Add the PanNuke citation. |

No other external dataset, model checkpoint, or codebase was found to be material to the reported scientific results. LoRA, AdaptFormer, HoVer-Net-style decoding, and metric publications are scientific methods/citations; the inspected local adapter implementations did not carry evidence that another checkpoint or separately copied code asset was used. Architecture artwork is project-created; qualitative image panels are STHELAR-derived and therefore covered by the STHELAR row. Python packages were intentionally not inventoried.

## What the adapter packages contain

The verified export path reconstructs the named base checkpoint, rejects any changed frozen parameter, and exports trainable parameters plus only mutable normalization buffers that changed during training. Representative metadata records `frozen_parameter_override_count: 0`; SAM-H packages have 296 trainable-state tensors plus 105 changed mutable buffers, and CellViT-256 packages have 114 trainable-state tensors plus mutable buffers. Thus the adapter files **do not embed or repackage unchanged upstream pretrained base parameters**. They are nevertheless designed for and named against CellViT, contain newly trained CellViT heads/adapters and changed runtime buffers, and should not be advertised as free of upstream attribution or derivative-work questions.

### A. Fold-specific evaluation adapters

These are the exact models used for held-out-slide paper evaluation. They may be released adapter-only, with the base checkpoint excluded, provided each package keeps the exact base name/Drive ID/SHA256, fold/seed/data scope, CellViT/SAM/HIPT/STHELAR attributions, applicable license links/notices, and an explicit statement that users must obtain the base separately. Do not merge Fold A/B states or relabel them as deployment models.

### B. Future both-slide deployment adapters

These are future, separately trained artifacts and are not evidence for the paper's evaluation numbers. The same upstream terms apply. Give each a distinct identifier and training-data scope, preserve base hash and notices, state that both slides were used, and label it as a deployment/research artifact with no held-out-slide performance claim. This audit does not select a license for those new adapters.

### C. Fine-tuning code

The code release is substantially derived from CellViT and includes SAM/HIPT/DINO-derived areas. The root and Hugging Face releases now provide the full Apache 2.0 text, applicable Commons Clause conditions, component-specific copyright/source notices, mandatory CellViT citation, and a prominent distribution-wide modification statement. The existing release-license choice was preserved; this remediation did not select or broaden a license for future both-slide adapters or other new artifacts.

## Remediation completed and manuscript handoff

1. **Completed:** both release roots contain the full current CellViT/component license and Apache 2.0 text; both contain matching `NOTICE` files with attribution and modification disclosures.
2. **Completed:** repository/model-card documentation links the notices and CC BY 4.0, records the STHELAR revision, explains that base checkpoints are excluded, and adds HIPT/PanNuke citations.
3. **Publication handoff:** include the remediated files in any corresponding public source or model distribution.
4. **Manuscript handoff:** insert the paragraph and checklist justification below and add `hipt` and `pannuke` bibliography entries. The final manuscript source was not present here, so it was not modified.
5. **Commercial-use caveat:** obtain written clarification/permission for Commons-Clause-covered CellViT/HIPT components and resolve the PanNuke model-weight question before any commercial or paid-service release. This does not affect the audited non-commercial academic release.

## Paper-ready Appendix paragraph

```tex
\myparagraph{Existing assets and licenses.}
We use STHELAR 40x revision
\texttt{e32a8cdd50eff2d38e237f3729e9ac85bbb5203b}, released under CC BY 4.0,
and the official \texttt{CellViT-SAM-H-x40.pth} and
\texttt{CellViT-256-x40.pth} checkpoints. CellViT components and checkpoints
are distributed under Apache 2.0 with applicable Commons Clause conditions;
SAM-derived components use Apache 2.0, while CellViT-256 includes HIPT-derived
components subject to Apache 2.0 with the Commons Clause
\citep{cellvit,sam,hipt}. The upstream checkpoints were pretrained on PanNuke
\citep{pannuke}. Our adapter states do not redistribute unchanged frozen base
parameters, which must be obtained separately under the upstream terms.
```

## Generic research-artifact checklist justification

```tex
\item[] Answer: \answerYes{}
\item[] Justification: The material datasets, pretrained checkpoints, and
upstream framework components are cited and their applicable licenses and terms
are documented in Appendix~\ref{app:repro}. The released adapters exclude the
frozen base checkpoints and preserve the required upstream attribution and
redistribution notices.
```

## Exact citations and version text to add

- **STHELAR:** Giraud-Sauveur, F., Blampey, Q., Benkirane, H., et al. “STHELAR, a Multi-Tissue Dataset Linking Spatial Transcriptomics and Histology for Cell Type Annotation.” *Scientific Data* (2026). DOI: `10.1038/s41597-026-06937-6`. Add `FelicieGS/STHELAR_40x`, revision `e32a8cdd50eff2d38e237f3729e9ac85bbb5203b`, and “CC BY 4.0.”
- **CellViT:** Hörst, F., Rempe, M., Heine, L., et al. “CellViT: Vision Transformers for Precise Cell Segmentation and Classification.” *Medical Image Analysis* 94 (2024), 103143. DOI: `10.1016/j.media.2024.103143`.
- **SAM:** Kirillov, A., Mintun, E., Ravi, N., et al. “Segment Anything.” *ICCV* (2023). arXiv: `2304.02643`.
- **HIPT:** Chen, R. J., Chen, C., Li, Y., et al. “Scaling Vision Transformers to Gigapixel Images via Hierarchical Self-Supervised Learning.” *CVPR* (2022), 16144–16155.
- **DINO (if its role is stated):** Caron, M., Touvron, H., Misra, I., et al. “Emerging Properties in Self-Supervised Vision Transformers.” *ICCV* (2021).
- **PanNuke:** Gamper, J., Koohbanani, N. A., Graham, S., et al. “PanNuke Dataset Extension, Insights and Baselines.” arXiv: `2003.10778` (2020); retain the original PanNuke 2019 citation if already present.
- **Repository/model-card metadata only:** retain the official Drive IDs and SHA256 values recorded in the asset table and release documentation; they need not be added to the paper.

The new manuscript keys used by the Appendix snippet can be added as:

```bibtex
@inproceedings{hipt,
  title     = {Scaling Vision Transformers to Gigapixel Images via Hierarchical Self-Supervised Learning},
  author    = {Chen, Richard J. and Chen, Chengkuan and Li, Yicong and Chen, Tiffany Y. and Trister, Andrew D. and Krishnan, Rahul G. and Mahmood, Faisal},
  booktitle = {Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition},
  pages     = {16144--16155},
  year      = {2022}
}

@article{pannuke,
  title   = {PanNuke Dataset Extension, Insights and Baselines},
  author  = {Gamper, Jevgenij and Koohbanani, Navid Alemi and Graham, Simon and Jahanifar, Mostafa and Benes, Ksenija and Khurram, Syed Ali and Azam, Ayesha and Hewitt, Katherine and Rajpoot, Nasir},
  journal = {arXiv preprint arXiv:2003.10778},
  year    = {2020}
}
```
