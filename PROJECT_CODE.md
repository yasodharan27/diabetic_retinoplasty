
## Project Overview

This repository is the baseline implementation for my diabetic retinopathy detection project.

The objective is to extend this repository into the architecture proposed in my 0th Review presentation. Do not rewrite the repository from scratch. Preserve existing functionality and improve it incrementally.

All model training will be performed in Google Colab.

---

## Current Project State and Plan (updated 2026-10-08)

This section records where the project actually stands. Where it conflicts with an older section of this
document or with the design documents (`IMPLEMENTATION_PLAN.md`, `PROJECT_STRUCTURE.md`,
`RACAF_ARCHITECTURE.md`, `JOINT_TRAINING_ARCHITECTURE.md`, `CORN_ARCHITECTURE.md`,
`SEGMENTATION_ARCHITECTURE.md`), **this section is authoritative**. The complete, append-only evidence
is the research record `docs/experiments/RACAF_Gate_Initialization_And_C1_Control.md` (sections §1–§65);
section numbers below refer to it.

### What is built and frozen

| Stage | Component | Status |
|---|---|---|
| 1 | Image Quality Assessment, EfficientNetB0 on EyeQ | Trained, frozen (QWK 0.8987 on the EyeQ test split). Not part of the grading graph. |
| 2 | Preprocessing, gamma + CLAHE ("DR" profile) | Frozen. Applied once at native size; a full-frame resize to 512×512 gives the RGB frame every grader reads. |
| 3 | Vessel segmentation, pretrained LWNet | Frozen, inference only. SHA-256 `91f0cada…961118de1`. |
| 4 | Lesion segmentation **v2**, SE-ResNet-101 U-Net (MA, HE, EX, SE) | Trained on IDRiD segmentation + TJDR, frozen. SHA-256 `cb5fc7a8…766f8ad8`. One-time IDRiD gate passed (§48). The earlier Attention U-Net (Experiment 2C) is superseded and its outputs are deny-listed. |
| 5–8 | Graders (below) + CORN ordinal decoding | Several trained and compared; see results. |
| 9–11 | Uncertainty, explainability, final evaluation | Not started for the current models. Not scheduled until the user says so. |

### Grading models and results (APTOS 2019, fixed split 2,921 / 730, seeds 42 / 123 / 2026)

| Model | What it is | Validation QWK (mean) | Outcome |
|---|---|---|---|
| Original Stage 5–8 with RACAF vs without | Multi-kernel CNN + Swin + cross-attention (+ RACAF) + CORN | about 0.83 | RACAF inconclusive; gate never learned. **Closed** (`Multiseed_Improved_Training_Report.md`; record §1–§5). |
| **P** | ImageNet ConvNeXt-Tiny, RGB only, CORN | **0.917** | The reference baseline (§22). |
| PL | P with vessel + lesion maps as extra input channels | 0.918 | No gain; grade-3 recall worse (§22). |
| C2 | Linear probe: pooled lesion features added to RGB features | – | No increment (§50). |
| Architecture 1 | Gated injection of vessel / lesion features into ConvNeXt | 0.915 | No gain; gates stayed near zero (§55). |
| Pathology grader | Vessel + lesion maps only, no RGB | 0.878 | Grades on its own; redundant with RGB (§57). |
| 50/50 and severity-aware fusion | P + pathology grader, fixed rules | 0.912 / 0.913 | Comparable to P, no gain (§57, §59). |
| **E1** | P + one 1×1 lesion head on the final feature map, trained on aligned Stage-4 targets (λ = 1) | **0.918** | Comparable to P; changes the representation (§62–§63). |
| E2 | E1 with lesion targets shuffled between training images (control) | 0.914 | Does not reproduce E1's representation change (§63.4). |

- **External (IDRiD grading test, one-time, 100-image primary set, §61):** frozen dual-branch pipeline QWK
  0.629 ± 0.044; P 0.642 ± 0.090; pathology grader 0.592 ± 0.083. Ranking (AUROC 0.86–0.92) transferred better
  than decoded grades. No superiority over P.
- **Mechanism (§63.2, §63.4):** a fresh probe on the frozen encoder reads Stage-4 lesion maps much better from
  E1 than from P (mean cell AUROC +0.0685, 95 % interval 0.0659 to 0.0715, 3 / 3 seeds); with shuffled targets
  (E2) about a tenth of that gain remains. Image-aligned lesion supervision changes the representation; a grading
  benefit has **not** been demonstrated.
- **Closed and not to be reopened without new evidence:** RACAF and reliability gating; early concatenation;
  pooled-feature and probability fusion; gated injection; frozen foundation-model branches (RETFound, DINOv2);
  new pooling / read-out heads; the generic-vessel grade-4 route; grade-3/4 specialist heads. Lesion-guided
  routing ("PGER") and structured vessel–lesion representations were rejected on a literature audit.

### Current position

- The strongest defensible claim is mechanistic: image-aligned lesion supervision changes what the grader's
  features encode, at no measurable grading cost and with no grading gain.
- The frozen, externally evaluated system is the dual-branch pipeline of §60–§61. E1 is the current research
  line. No final model has been declared.
- The rule that Stage 3/4 outputs must feed the classifier was lifted by the user on 2026-10-05: an RGB-only
  grading classifier is acceptable, with Stage 3/4 kept as stages (Stage 4 is E1's teacher).
- APTOS validation is also the checkpoint-selection set; it is not an untouched test.

### Approved next phase (2026-10-08; in order)

1. Pre-run record for IDRiD batch 1 — written (§65).
2. **IDRiD batch 1:** E1 and E2 on the locked IDRiD protocol (100-image primary set, 103 as sensitivity); P's
   stored predictions are reused. Evaluator implemented (§65.1); gates and the one-time run are pending on the T4.
3. **DDR** download, verification and overlap screen (expert lesion masks, independent of Stage 4) — done (§66, §66.1).
4. **Ground-truth lesion probe:** the same frozen-encoder probe on P, E1 and E2 against DDR's real masks.
5. **EyePACS adaptation:** Stage 2 on EyePACS, patient-level 90/10 split, one supervised adaptation run of P's
   model, checkpoint pinned by hash. APTOS is never read in this phase.
6. **P-EP and E1-EP** on APTOS (three seeds each, unchanged P / E1 protocol), with probes on both.
7. **E2-EP** only if the E1-EP probe criterion passes.
8. One-shot evaluations of the EP models: the labelled EyePACS test subset and IDRiD batch 2.

Not planned: any further fusion, gating, routing or graph module; E3; backbone or foundation-model swaps; λ
sweeps before the steps above; a new architecture.

### Working rules added during the research phase

- Results and analyses are appended to the research record automatically; earlier sections are never rewritten.
- Rules, tolerances and reading criteria are written into the record **before** a result exists.
- Comparisons are matched-seed against P, with the paired bootstrap of §54 (2,000 resamples, seed 20260927).
- A result is interpreted before any follow-up experiment is designed; follow-ups need a recorded decision.
- Gates are never loosened after a failure; a failed gate is diagnosed first (§62.1–§62.3).
- IDRiD's grading test set has been used once (§61). Further use is limited to the two pre-declared batches
  above, with the prior use disclosed; nothing is ever selected or tuned on it.
- In Keras 3, the mixed-precision policy is set **after** `keras.backend.clear_session()`, never before (§62.2).
- Colab is used only for GPU work (training, inference, feature extraction); everything else is analysed
  locally from the Drive copies.

---

## Existing Baseline

The current repository already implements:

- Image preprocessing
- EfficientNet-based classification
- Swin Transformer hybrid model
- Monte Carlo Dropout
- Grad-CAM
- Training and testing scripts

Reuse these components wherever appropriate instead of replacing them unnecessarily.

---

## Target Pipeline

> This is the pipeline of the original (0th Review) design. Stages 1–4 exist as listed (Stage 4 in its v2
> form). Stages 5–8 were implemented and trained as designed, and the RACAF comparison closed without a
> supported benefit; the graders actually in use are listed under "Current Project State and Plan" above.

1. Image Quality Assessment
2. Image Preprocessing
3. Vessel Segmentation
4. Lesion Segmentation
5. Local Feature Extraction
6. Global Feature Extraction
7. Feature Fusion
8. Reliability-Aware Cross-Attention Fusion (RACAF) — the single approved research innovation; see "Approved Research Innovation" below and `RACAF_ARCHITECTURE.md`
9. Ordinal Classification
10. Uncertainty Estimation
11. Explainability
12. Evaluation

---

## Preprocessing

Stage 02 (Image Preprocessing) is the single, canonical, dataset-agnostic preprocessing pipeline used by every later stage. It is deterministic and model-agnostic:

- Gamma Correction
- CLAHE

Image Quality Assessment (Stage 01) is a separate, upstream classifier (EfficientNetB0) and is not part of Stage 02's transform list — it gates which images reach Stage 02, it does not preprocess them.

Green Channel Extraction, Ben Graham Preprocessing, Median Denoising, and Histogram Equalization are explicitly **not** part of Stage 02. Resizing and Data Augmentation are also excluded from Stage 02's stored output — resizing is performed independently by whichever downstream stage requires it (different stages have different resolution needs), and augmentation is applied at training time, in-graph, by each trainable stage — never baked into a static preprocessed file. See `SEGMENTATION_ARCHITECTURE.md` and `PROJECT_STRUCTURE.md` for the full rationale and the per-stage detail.

### Stage 02 Preprocessing Policy

This is the repository's official preprocessing policy, binding on every dataset and every downstream stage:

> Stage 02 preprocessing is deterministic.
> Each dataset is preprocessed exactly once.
> Processed outputs are stored and reused by every downstream stage.
> No downstream stage should regenerate deterministic preprocessing outputs.

Any model-specific adaptation of Stage 02's output (channel handling, resizing, normalization) happens inside the consuming stage's own adapter, at load time, on top of the already-generated processed file — it never re-runs or duplicates Stage 02's own Gamma/CLAHE computation.

---

## Models

The table below is the original design. As built today: Lesion Segmentation is the **v2 SE-ResNet-101 U-Net**
(the Attention U-Net is superseded); the reference grader is **P (ImageNet ConvNeXt-Tiny + CORN, RGB)** and the
current research model is **E1 (P + auxiliary lesion head)**. The Adaptive Multi-Kernel CNN, Dual-Scale Swin
Transformer, Adaptive Cross-Attention and RACAF were implemented and trained, and are kept in the repository as
the completed original Stage 5–8 line; they are not part of the current graders.

| Module | Model |
|---------|-------|
| Image Quality Assessment | EfficientNetB0 |
| Vessel Segmentation | Pretrained LWNet (external, inference only — not trained within this project) |
| Lesion Segmentation | Attention U-Net |
| Local Feature Extraction | Adaptive Multi-Kernel CNN |
| Global Feature Extraction | Dual-Scale Swin Transformer |
| Feature Fusion | Adaptive Cross-Attention |
| Reliability-Aware Fusion | Reliability-Aware Cross-Attention Fusion (RACAF) — see `RACAF_ARCHITECTURE.md` |
| Classification | CORN Ordinal Classification |
| Uncertainty | Monte Carlo Dropout |
| Explainability | Grad-CAM++, SHAP, Attention Rollout |

Vessel Segmentation uses [`lwnet`](https://github.com/agaldran/lwnet) ("The Little W-Net That Could," Galdrán et al., MIT-licensed), a pretrained, externally-sourced vessel segmentation model, for inference only. This reverses an earlier, since-superseded design ("Baseline U-Net," trained within this project on DRIVE + CHASE_DB1); see `SEGMENTATION_ARCHITECTURE.md` §1.2/§2 and its Appendix for the full design history.

---

## Approved Research Innovation

> **Status 2026-10-08: RACAF is closed.** It was implemented, trained and compared against its no-RACAF control
> in a six-run matched experiment; the result was inconclusive and follow-up diagnostics showed the gate was
> never meaningfully learned (`docs/experiments/Multiseed_Improved_Training_Report.md`; research record §1–§5). The text below is kept as the record of the original
> commitment. The project's current research question — whether image-aligned lesion supervision changes the
> grader's representation and whether that matters for grading — is described under "Current Project State and
> Plan" above. The restriction on "a second, competing downstream research innovation" was superseded by the
> user's later decisions recorded there.

**Reliability-Aware Cross-Attention Fusion (RACAF)** is the single approved research innovation
for this project. Its full specification lives in `RACAF_ARCHITECTURE.md`, which is authoritative
for its design; this section records the decisions that make it binding project policy.

- RACAF is positioned after Adaptive Cross-Attention (Stage 7) and before CORN (now Stage 9 in
  the Target Pipeline above) — it wraps Cross-Attention's output, it does not redefine
  Cross-Attention's own computation.
- Stage 4 (Lesion Segmentation, Attention U-Net) is finalized as Experiment 2C (Weighted-Pooled
  Dice) and remains **frozen** for RACAF and for every downstream stage — never retrained, never
  architecturally modified (no dropout is added to it), as part of building or training RACAF.
- RACAF's reliability signal is computed via **test-time augmentation (TTA) predictive
  disagreement** on Stage 4's frozen output, per image, at inference time — never from Stage 4's
  recorded test-set Dice/IoU or any other held-out-set statistic.
- Stage 4's measured test-set Dice/IoU values (see `SEGMENTATION_ARCHITECTURE.md` and
  `RACAF_ARCHITECTURE.md` §1) remain documented as evaluation results only. They must **never**
  become an input to RACAF, or to any other trainable downstream component.
- No second, competing downstream research innovation should be introduced unless this project's
  specification is deliberately revised. RACAF is the ONE innovation this pipeline commits to
  beyond the already-frozen Stages 1–4.

---

## Datasets

Datasets

- EyeQ
  - Image Quality Assessment
  - Reconstructed (one-time) from EyePACS using the official EyeQ generation repository. The reconstructed EyeQ dataset is the dataset actually used for IQA; EyePACS itself is not part of this project (see "EyePACS" below).
  - Used only for Stage 01.

- APTOS 2019
  - Ordinal DR Classification (Stage 08) and other downstream grading/classification stages.

- IDRiD
  - Segmentation set: Stage 4 lesion-segmentation training (v2 split 44 / 10) and its one-time 27-image test gate (§48, consumed).
  - Grading **test** set (103 images): evaluation only. Used once for the frozen pipeline (§61). Three images
    (`IDRiD_088`, `IDRiD_089`, `IDRiD_091`) are copies of Stage-4 training images and are excluded from the
    100-image primary set. Remaining permitted use: batch 1 (E1, E2) and batch 2 (EP models), §65.
  - Grading training set: never used.

- EyePACS (Kaggle 2015)
  - Used once to reconstruct the official EyeQ dataset (Stage 1).
  - **Approved 2026-10-08 as a controlled experimental factor for grading** (research record §64, §65): the
    35,126 labelled training images (17,563 patients) for one supervised adaptation run with a patient-level
    90 / 10 split; the 16,249 EyeQ-labelled test images as a one-shot evaluation set. APTOS is never read
    during adaptation. Not yet preprocessed or used.
  - An overlap screen found no copy of any APTOS image among these EyePACS images (§64, §64.1); patient-level
    independence cannot be shown and is stated as a limitation.

- TJDR (added 2026-10-01 at the user's explicit request)
  - Stage 4 v2 lesion-segmentation training data only (MA, HE, EX, SE); never used for grading.
  - Pre-training audit, duplicate/leakage exclusions and licence evidence: research record §42.

- DDR (approved 2026-10-08; lesion subset downloaded and verified, research record §66)
  - Its 757 expert lesion-annotated images (383 / 149 / 225) are the planned ground-truth for the frozen-encoder
    lesion probe, because they are independent of Stage 4's training data. Source: Hugging Face mirror
    `ctmedtech/DDR-dataset` at a pinned revision; no overlap with APTOS, EyePACS, IDRiD or TJDR was found.
  - Probe set: 756 images — one validation image (`007-5869-300`) is excluded by decision (§66.1). The grading
    images were not downloaded. No patient identifiers exist; the mirror's dataset card is unreliable.
  - Not approved for training any grader or segmenter.

**Local layout** (all under `datasets/`, which `.gitignore` excludes in full, so git never tracks or touches it). On 2026-10-01 everything was consolidated here from `D:\Projects\` and deduplicated. Removed copies were verified identical to the kept copies (name + size for every file, CRC32 on samples): the Kaggle zip parts, a second extracted EyePACS copy, the original IDRiD download, the original EyeQ reconstruction output, and a 137 GB `datasets.zip` backup.

```
datasets/
  APTOS2019/{raw,processed}
  EyePACS/{raw,processed}        raw = Kaggle EyePACS 2015: train/ (35,126), test/ (53,576), trainLabels.csv, sampleSubmission.csv
  EyeQ/{raw,processed}           raw = the reconstructed EyeQ dataset used by Stage 01
  EyeQ/official_repo/            github.com/HzFu/EyeQ clone (label CSVs, MCF_Net code; nested .git)
  EyeQ/reconstruction/           scripts + validation report that built EyeQ/raw from EyePACS
  IDRiD/{segmentation,grading,localization}/{raw,processed}   (+ IDRiD/LICENSE.txt, IDRiD/CC-BY-4.0.txt)
  TJDR/{raw,processed}           raw/{train,test}/{image,annotation}: md5-verified TJDR release
  TJDR/audit/                    record section 42 audit outputs + source md5 listing
```

Folder names inside each `raw/` are the datasets' official names (e.g. IDRiD `1. Original Images/a. Training Set`). The loaders depend on them, so they are not renamed.

DRIVE and CHASE_DB1 are **not** project datasets. They were approved under an earlier, since-superseded design that trained Vessel Segmentation within this project; Stage 03 now runs a pretrained external checkpoint (LWNet) instead and requires no dataset of its own — see `SEGMENTATION_ARCHITECTURE.md`'s Appendix A.1 for the history. No other datasets are permitted without explicit request.

---

## Development Rules

- Understand the existing implementation before modifying it.
- Implement one module at a time.
- Preserve existing functionality.
- Do not rewrite the repository.
- Do not modify unrelated files.
- Keep the project modular.
- Explain the implementation plan before writing code.
- Wait for approval before moving to the next module.

---

## Training

All deep learning model training must be performed using Google Colab.

The local repository should only contain:

- source code
- notebooks
- trained weights
- evaluation scripts

Datasets must remain inside the datasets directory and should not be committed.

For every trainable model:

- Create a separate Colab notebook.
- Save the best model.
- Export trained weights.
- Integrate the trained model back into the repository.

---

## Coding Standards

- Write clean and modular code.
- Reuse existing code wherever possible.
- Avoid duplicate implementations.
- Add comments only where necessary.
- Do not generate placeholder implementations.
- Do not fabricate evaluation metrics or results.

## Implementation Rules

This is a production/research project, not a demonstration project.

1. Do not implement placeholder logic, simulated outputs, fake metrics, or dummy pipelines.
2. Unit tests may use synthetic or temporary data only, to verify correctness of individual functions.
3. All actual project functionality must operate on the real datasets: EyeQ, APTOS2019, IDRiD, TJDR, and — for the approved next phase only — EyePACS and DDR in the roles stated in the Datasets section. Vessel Segmentation (Stage 03) uses a vendored pretrained checkpoint and requires no project dataset of its own.
4. Do not create "toy" implementations intended to be replaced later.
5. Every module should be fully implementable and immediately usable in the final pipeline.
6. If verification of a full dataset would require hours of execution, perform lightweight correctness tests only -- never replace the actual implementation with a simplified version.
7. Do not fabricate evaluation results or performance metrics. If a model has not been trained, clearly state that no real evaluation exists.
8. Every trainable module must include: dataset loader, model, training, evaluation, inference, and a deployment interface (see Deployment Requirement below).

The final objective is a real-world end-to-end diabetic retinopathy diagnosis pipeline, not an academic prototype.

## Development Workflow

Before implementing any module:

1. Explain the existing implementation.
2. Explain why it needs to be changed.
3. Propose the implementation plan.
4. Wait for approval.
5. Implement the module.
6. Verify integration.
7. Stop and wait for the next task.

## Dataset Handling

Do not automatically download datasets.

Always use datasets available inside:

datasets/

Before implementing any module, inspect the available dataset structure and adapt the data loader accordingly.

Do not assume fixed folder names.

---------

## Existing Repository

Before implementing any module:

- Inspect the current implementation.
- Reuse existing functionality whenever possible.
- Extend instead of replacing.
- Avoid duplicate implementations.

--------

## Training Strategy

Each trainable module must have:

- Separate Google Colab notebook
- Independent training
- Saved best weights
- Easy integration into the main repository

------ 

## Dataset Policy

Never modify the original datasets inside `datasets/*/raw`.

All preprocessing outputs must be written to the corresponding `processed` folder.

The original datasets must always remain untouched.

Ground-truth mask/label data (e.g. IDRiD's lesion masks and grading CSVs) is never run through Stage 02 preprocessing. Only fundus images pass through Stage 02; masks and labels are read directly from `raw/` by each stage's own dataset loader.

-----
## Deployment Requirement

Every trainable module must provide two components:

1. A training implementation used to train and export the model.
2. An inference implementation that loads the exported model and exposes a reusable prediction interface.

The final project must integrate all inference modules into a single end-to-end pipeline capable of accepting one retinal fundus image and producing the complete diabetic retinopathy analysis.

No module should exist only for training.
-----

## Modular Stage Principle

This is a repository-wide architectural rule, not specific to any one stage:

Every trainable stage owns its own:

- dataset (loader)
- model
- training
- evaluation
- inference
- exported model

Stages communicate with each other **only** through their documented input/output contracts (e.g. `pipeline.SegmentationStage`'s `predict()` return shape, or a stage's documented tensor contract in `SEGMENTATION_ARCHITECTURE.md`). No stage may directly depend on another stage's internal implementation — its model class, its training loop, its private helper functions, or its choice of framework. A stage's internals may change freely (including which framework it's implemented in, or whether it is trained within this project at all — see `SEGMENTATION_ARCHITECTURE.md` §6/Appendix A.1 for how Stage 03 changed both) as long as its documented contract stays the same.

This formalizes and generalizes the Deployment Requirement above: it is not just about every trainable module having both a training and an inference half, but about every stage being independently replaceable without touching any other stage's code.

-----

## Architecture Freeze

The architecture described in this document, `IMPLEMENTATION_PLAN.md`, `PROJECT_STRUCTURE.md`, `SEGMENTATION_ARCHITECTURE.md`, `README.md`, and `colab/README.md` is frozen as of the documentation refactor that finalized Stage 02 (RGB, Gamma, CLAHE, deterministic, generated once per the Stage 02 Preprocessing Policy above) and, most recently, revised Stage 03 to a **pretrained LWNet model, run for inference only** (`SEGMENTATION_ARCHITECTURE.md` §1.2/§2), reversing an intermediate design that trained a "Baseline U-Net" within this project on DRIVE + CHASE_DB1. Earlier alternatives considered for these stages (a single-channel canonical image; a trainable Baseline U-Net on DRIVE + CHASE_DB1; a framework-agnostic, still-undecided model storage format for Stage 03) are retained only in `SEGMENTATION_ARCHITECTURE.md`'s dedicated design-history appendix, not restated as live guidance anywhere else. Implementation of Stage 02 (complete) and Stage 03 (LWNet integration) may now proceed, one module at a time, per the Development Workflow above and the Modular Stage Principle above.

Stage 4 (Lesion Segmentation) is now also finalized — Experiment 2C (Weighted-Pooled Dice),
frozen — and `RACAF_ARCHITECTURE.md` is added to this frozen document set as the authoritative
specification for RACAF, the one approved downstream research innovation (see "Approved Research
Innovation" above). RACAF itself is not yet implemented and depends on Stages 5–7's own output
contracts being finalized first; its design is frozen ahead of that implementation, exactly as
this document already froze Stage 3/4's design ahead of their own implementation.

**Update 2026-10-08.** The two paragraphs above describe the freeze as it stood before the research phase. Since
then: Stage 4 was retrained as v2 (SE-ResNet-101 U-Net) and that model, not Experiment 2C, is the frozen Stage 4;
RACAF was implemented, trained and closed; and the dual-branch pipeline of research record §60 was frozen and
evaluated once on IDRiD (§61). What is frozen now is Stages 1–4 (with Stage 4 v2), the P, E1 and E2 checkpoints
(pinned by SHA-256 in §60 and §65), the APTOS split and the locked IDRiD protocol. See "Current Project State and
Plan" at the top of this document.
