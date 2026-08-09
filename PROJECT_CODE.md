
## Project Overview

This repository is the baseline implementation for my diabetic retinopathy detection project.

The objective is to extend this repository into the architecture proposed in my 0th Review presentation. Do not rewrite the repository from scratch. Preserve existing functionality and improve it incrementally.

All model training will be performed in Google Colab.

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

1. Image Quality Assessment
2. Image Preprocessing
3. Vessel Segmentation
4. Lesion Segmentation
5. Local Feature Extraction
6. Global Feature Extraction
7. Feature Fusion
8. Ordinal Classification
9. Uncertainty Estimation
10. Explainability
11. Evaluation

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

| Module | Model |
|---------|-------|
| Image Quality Assessment | EfficientNetB0 |
| Vessel Segmentation | Pretrained LWNet (external, inference only — not trained within this project) |
| Lesion Segmentation | Attention U-Net |
| Local Feature Extraction | Adaptive Multi-Kernel CNN |
| Global Feature Extraction | Dual-Scale Swin Transformer |
| Feature Fusion | Adaptive Cross-Attention |
| Classification | CORN Ordinal Classification |
| Uncertainty | Monte Carlo Dropout |
| Explainability | Grad-CAM++, SHAP, Attention Rollout |

Vessel Segmentation uses [`lwnet`](https://github.com/agaldran/lwnet) ("The Little W-Net That Could," Galdrán et al., MIT-licensed), a pretrained, externally-sourced vessel segmentation model, for inference only. This reverses an earlier, since-superseded design ("Baseline U-Net," trained within this project on DRIVE + CHASE_DB1); see `SEGMENTATION_ARCHITECTURE.md` §1.2/§2 and its Appendix for the full design history.

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
  - Lesion Segmentation (Attention U-Net, Stage 04) and downstream grading tasks.

- EyePACS (historical only)
  - Used once to reconstruct the official EyeQ dataset.
  - Not part of the implemented training or inference pipeline.
  - Not required to reproduce or run this repository.

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
3. All actual project functionality must operate on the real datasets: EyeQ, APTOS2019, IDRiD. EyePACS was used only once, historically, to reconstruct EyeQ, and is not required to reproduce or run this repository (see Datasets section). Vessel Segmentation (Stage 03) uses a vendored pretrained checkpoint and requires no project dataset of its own.
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
