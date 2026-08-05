# Project Structure

Master architectural reference for this repository. `PROJECT_CODE.md` defines the target
architecture and development rules; `IMPLEMENTATION_PLAN.md` tracks the gap between the baseline
and that target; this document describes **where everything actually lives** and **how the
pieces fit together**, kept in sync with the repository as it exists today.

---

## Repository Overview

| Folder | Purpose |
|---|---|
| *(repository root)* | Flat, top-level Python modules -- see "Source Code Organization" below; this repository does **not** use a `src/` layout. |
| `training/` | Reusable, model-agnostic training framework (`Trainer`, callbacks, losses, metrics, optimizers). No models or dataset loading. |
| `evaluation/` | Reusable, model-agnostic evaluation framework (`Evaluator`, metrics, visualization). Operates on prediction arrays only. |
| `pipeline/` | Abstract base classes (`TrainableStage`, `InferenceStage`, `SegmentationStage`, `ClassificationStage`) fixing the contract future pipeline stages must implement. Defines no models. |
| `datasets/` | Real, local datasets: `EyeQ/`, `APTOS2019/`, `IDRiD/`. `raw/` subfolders are read-only; never modified in place. |
| `colab/` | The official Google Colab training infrastructure -- `common/` (reusable setup/verification/experiment modules) and `notebooks/` (one notebook per pipeline stage). See "Colab Workflow" below. |
| `tests/` | Pytest unit tests. Use synthetic/temporary data only, per `PROJECT_CODE.md`'s Implementation Rules -- never a substitute for real-data verification. |
| `docs/` | Operational documentation, e.g. `docs/FIRST_TRAINING_CHECKLIST.md`. |
| `research_papers/` | Reference PDFs backing the target architecture's design choices (Swin Transformer, DR-GAN++, attention-based grading, etc.). |
| `Visualization_Scripts/`, `formula/`, `assets/` | Publication/report-generation utilities (architecture diagrams, LaTeX equation renders, README images) -- not part of the inference pipeline. |
| `models/`, `results/` | Local, gitignored output of CLI training/evaluation scripts (`train_image_quality.py`, `evaluate_image_quality.py`, ...). Colab runs write to Google Drive instead -- see "Output Locations". |

---

## Source Code Organization

There is no `src/` directory. Every pipeline-stage module is a flat, top-level `.py` file,
imported directly by name (`from image_quality_dataset import load_eyeq_datasets`), matching how
`config.py`, the test suite, and every Colab notebook's bootstrap `sys.path` insertion already
expect the repository to be laid out. This is a deliberate, existing convention -- not a gap to
fill in.

- **`training/`** -- `Trainer` / `TrainingConfig` (mixed precision, checkpointing, early
  stopping, `ReduceLROnPlateau`, TensorBoard, resume support), plus `losses.py`, `metrics.py`,
  `optimizers.py`, `callbacks.py`. Any trainable stage builds its own model and dataset, then
  hands both to `Trainer` -- this is how `train_image_quality.py` already works, and how every
  future trainable stage should work too.
- **`evaluation/`** -- `Evaluator` (accuracy/precision/recall/F1/confusion matrix/ROC/AUC/QWK/
  calibration) plus `visualization.py`'s plotting helpers. Operates purely on `(y_true, y_pred,
  y_proba)` arrays -- no model or dataset coupling, so it's reusable by every classification-style
  stage (`evaluate_image_quality.py` already uses it this way).
- **`pipeline/`** -- `TrainableStage` / `InferenceStage` / `SegmentationStage` /
  `ClassificationStage` ABCs. Establishes the contract for stages **implemented from this point
  forward**; the existing IQA module predates this package and does not inherit from it.
- **`datasets/`** -- real data only, `<dataset>/raw/` (read-only) and `<dataset>/processed/`
  (preprocessing output) per dataset, resolved via `config.py`'s `dataset_raw_dir()` /
  `dataset_processed_dir()` helpers (or the dedicated `EyeQPaths` for EyeQ specifically).
- **`colab/`** -- see "Colab Workflow" below.
- **`tests/`** -- `test_config.py`, `test_image_preprocessing.py`, `test_pipeline.py`. Synthetic/
  temporary data only, verifying function-level correctness -- real-data verification happens via
  the Colab notebooks and dataset verification modules, never via this suite alone.
- **`docs/`** -- operational runbooks, not architectural documentation (that's this file and
  `PROJECT_CODE.md`).
- **`research_papers/`** -- background reading, not code.

---

## Dataset Organization

| Dataset | Purpose | Current Usage | Future Usage |
|---|---|---|---|
| **EyeQ** | Image quality classification (`Good`/`Usable`/`Reject`) | **Active** -- trains and evaluates Stage 1 (Image Quality Assessment) today, via `image_quality_dataset.py` / `train_image_quality.py` / `colab/notebooks/stage01_iqa.ipynb`. | Continues to gate every later stage: only `Good`/`Usable` images should reach Stage 2 preprocessing. |
| **APTOS2019** | Diabetic retinopathy severity classification (5 classes, 0-4) | Referenced by the pre-refactor baseline scripts (`efficientnet_model.py`, `swin_transformer.py`, `train_hybrid_model.py`) at the repository root. | Primary training set for Stage 8 (CORN Classification) once the target architecture's classification head is implemented. |
| **IDRiD** | Lesion segmentation (microaneurysms, haemorrhages, hard/soft exudates, optic disc) | Not yet consumed by any implemented stage. | Trains Stage 4 (Lesion Segmentation, Attention U-Net) -- the only segmentation model actually trained in this project (see below). |

**EyePACS is historical only.** It was used once, outside this repository, to reconstruct the
official EyeQ dataset via EyeQ's own generation repository. EyePACS is not present under
`datasets/`, is not read by any script here, and is not required to reproduce or run this
repository -- see `PROJECT_CODE.md`'s Datasets section for the full history. Do not reintroduce
EyePACS as an operational data source without updating that section first.

**Vessel Segmentation (Stage 3) does not train on any of these datasets.** Per
`SEGMENTATION_ARCHITECTURE.md`, it uses a pretrained U-Net for inference only -- resolved
specifically because IDRiD does not reliably provide vessel-specific masks (it is primarily a
lesion-segmentation dataset).

---

## Pipeline Overview

The full 11-stage target architecture (`PROJECT_CODE.md`). "Status" reflects what's actually
implemented and verified today, not aspirational state.

### 1. Image Quality Assessment
- **Purpose:** Gate low-quality fundus images before they reach the rest of the pipeline.
- **Input:** Raw fundus images (EyeQ, or any unlabeled folder via `image_quality_inference.py`).
- **Output:** `Good` / `Usable` / `Reject` classification + per-class confidence.
- **Training dataset:** EyeQ.
- **Inference output:** `{"label", "class_index", "confidence", "probabilities"}` per image (see `image_quality_inference.predict_quality`).
- **Dependencies:** None (first stage).
- **Status:** **Implemented and verified.** `image_quality_dataset.py`, `image_quality_model.py`, `train_image_quality.py`, `evaluate_image_quality.py`, `image_quality_inference.py`, `colab/notebooks/stage01_iqa.ipynb`. Not yet trained for real -- see `docs/FIRST_TRAINING_CHECKLIST.md`.

### 2. Image Preprocessing
- **Purpose:** CLAHE, Gamma Correction, Green Channel Extraction, Ben Graham preprocessing, Median Denoising, resizing, augmentation.
- **Input:** Images passing Stage 1's quality gate.
- **Output:** Normalized images in `datasets/*/processed/`.
- **Training dataset:** N/A (deterministic image transform, not a trained model).
- **Inference output:** Preprocessed image, ready for Stages 3+.
- **Dependencies:** Stage 1.
- **Status:** **Partially implemented.** `image_preprocessing.py` (repository root) implements the transforms and `config.py`'s `PREPROCESSING_PROFILES` (`IQA` = no-op, `DR` = full pipeline); not yet wired into a Colab notebook (`colab/notebooks/stage02_preprocessing.ipynb` is a template only).

### 3. Vessel Segmentation
- **Purpose:** Segment retinal vasculature.
- **Input:** Preprocessed images (Stage 2).
- **Output:** Vessel masks.
- **Training dataset:** None -- **inference only**, pretrained U-Net (see `SEGMENTATION_ARCHITECTURE.md`).
- **Inference output:** Binary/probability vessel mask per image.
- **Dependencies:** Stage 2.
- **Status:** Not implemented. `colab/notebooks/stage03_vessel_segmentation.ipynb` is a template only.

### 4. Lesion Segmentation
- **Purpose:** Segment DR lesions (microaneurysms, hemorrhages, exudates).
- **Input:** Preprocessed images (Stage 2).
- **Output:** Lesion segmentation masks.
- **Training dataset:** IDRiD.
- **Inference output:** Per-lesion-class segmentation mask.
- **Dependencies:** Stage 2.
- **Status:** Not implemented. Design finalized in `SEGMENTATION_ARCHITECTURE.md`; `colab/notebooks/stage04_lesion_segmentation.ipynb` is a template only.

### 5. Local Feature Extraction
- **Purpose:** Fine-grained, lesion-level feature extraction (Adaptive Multi-Kernel CNN).
- **Input:** Preprocessed images (Stage 2) + lesion segmentation maps (Stage 4).
- **Output:** Local feature vectors/maps.
- **Training dataset:** Likely trained jointly with Stages 6-8 (see note below), not independently.
- **Dependencies:** Stages 2, 4.
- **Status:** Not implemented.

### 6. Global Feature Extraction
- **Purpose:** Whole-image, long-range feature extraction (Dual-Scale Swin Transformer).
- **Input:** Preprocessed images (Stage 2).
- **Output:** Global feature vectors/maps.
- **Training dataset:** Likely trained jointly with Stages 5, 7, 8.
- **Dependencies:** Stage 2.
- **Status:** Not implemented.

### 7. Feature Fusion
- **Purpose:** Combine local (Stage 5) and global (Stage 6) features (Adaptive Cross-Attention).
- **Input:** Local + global feature vectors.
- **Output:** Fused feature representation.
- **Dependencies:** Stages 5, 6.
- **Status:** Not implemented.

### 8. CORN Classification
- **Purpose:** Final ordinal DR-severity classification (CORN ordinal regression head).
- **Input:** Fused features (Stage 7).
- **Output:** DR severity grade (0-4, ordinal) + class probabilities.
- **Training dataset:** APTOS 2019.
- **Dependencies:** Stage 7.
- **Status:** Not implemented.

> **Note on Stages 5-8:** `PROJECT_CODE.md`'s Models table and the verified Google Drive layout
> (a single `experiments/FinalClassification/` bucket, not four separate ones) both suggest
> Stages 5-8 are trained **jointly as one model**, not as four independently checkpointed stages.
> Confirm this before implementing any of them independently.

### 9. Uncertainty Estimation
- **Purpose:** Quantify prediction confidence via Monte Carlo Dropout.
- **Input:** Trained Stage 8 classifier + test images.
- **Output:** Per-prediction confidence/uncertainty scores, reliability diagrams.
- **Dependencies:** Stage 8. Inference-only -- does not train a new model.
- **Status:** Not implemented. An unverified MC Dropout implementation exists in the pre-refactor `bayesian_inference.py`, but its reliability diagram explicitly uses **simulated** ground truth (stated in its own print output) -- do not treat it as a real calibration measurement without fixing that first.

### 10. Explainability
- **Purpose:** Visual explanations (Grad-CAM++, SHAP, Attention Rollout).
- **Input:** Trained Stage 8 classifier + test images.
- **Output:** Explanation heatmaps/attributions per prediction.
- **Dependencies:** Stage 8. Inference-only.
- **Status:** Not implemented. A standard (non-plus-plus) Grad-CAM implementation exists in the pre-refactor `explainable_ai.py`, tied to the old hybrid model's `swin_refine` layer -- not directly reusable against the target architecture's classifier.

### 11. Evaluation
- **Purpose:** End-to-end pipeline evaluation across the full test set -- the final gate before any result is reported as real.
- **Input:** Every upstream stage's exported model + held-out test splits.
- **Output:** Consolidated metrics report.
- **Dependencies:** Stages 1-10, all trained.
- **Status:** Not implemented. Per `PROJECT_CODE.md`: never fill this in with placeholder numbers before every upstream stage has a real trained model.

---

## Colab Workflow

Full detail lives in `colab/README.md`; summary here for architectural context.

- **Notebooks call into existing code, never reimplement it.** Each `colab/notebooks/stageNN_*.ipynb`
  orchestrates the corresponding stage's dataset/model/training/evaluation modules (once they
  exist) plus the shared `colab/common/` infrastructure.
- **Setup** (`colab/common/setup.py`): one call mounts Google Drive, clones/updates the
  repository, installs `requirements.txt`, enters the repository, and points dataset environment
  variables at Drive.
- **Experiments** (`colab/common/experiment_manager.py`): every training run gets its own
  timestamped, isolated folder under `experiments/<Module>/YYYY-MM-DD_HH-MM-SS/` on Google Drive
  (`checkpoints/`, `logs/`, `tensorboard/`, `evaluation/`, `predictions/`, `metadata.json`),
  never overwritten, resumable.
- **Model export:** the best checkpoint from each run is copied to a stable
  `exported_models/<Module>/best_model.keras` on Drive (overwritten by each new "best" run), kept
  separate from that run's own permanent `checkpoints/best.keras` archive.
- **TensorBoard:** launched against the current experiment's live `logs/` directory; any past
  experiment's `logs/` can be pointed at directly to review it later.
- **Resuming:** point `RESUME_EXPERIMENT_DIR` at a previous experiment's folder instead of
  leaving it `None` -- see `colab/README.md`'s "How to resume training".

---

## Local Development Workflow

- **VS Code** -- primary local editor for reading/editing repository code and reviewing Colab
  notebook diffs. Does not run training (no local GPU).
- **Claude Code** -- used for structured, incremental implementation work against
  `PROJECT_CODE.md`'s Development Workflow (explain existing implementation -> explain why it
  needs to change -> propose a plan -> wait for approval -> implement -> verify integration ->
  stop). Runs local smoke tests (small real-data slices, synthetic-data unit tests) but never
  fabricates training results.
- **Git** -- version control for source code, notebooks, and documentation. Datasets and heavy
  local outputs (`models/`, `results/`, `datasets.zip`) are gitignored; trained weights that
  should be versioned are committed explicitly and deliberately (see `colab/README.md`'s optional
  commit-and-push step, off by default).
- **Google Colab** -- the only place real model training happens (`PROJECT_CODE.md`'s Training
  policy). Reads code from GitHub (via `colab/common/setup.py`'s clone step) and data/outputs
  from Google Drive; the Colab VM itself is treated as fully ephemeral.

Interaction: code and notebooks are authored/reviewed locally (VS Code + Claude Code) -> pushed
to GitHub -> a Colab notebook clones that exact commit and trains against Drive-hosted data ->
the resulting model/evaluation artifacts are reviewed locally before being committed back.

---

## Output Locations

| Output | Local CLI run (`train_image_quality.py`, etc.) | Colab run |
|---|---|---|
| Trained models (`best.keras`, `last.keras`) | `models/<module>/training_run/checkpoints/` | `experiments/<Module>/<timestamp>/checkpoints/` (Drive) |
| Exported "current best" model | `models/<module>/best_model.keras` | `exported_models/<Module>/best_model.keras` (Drive) |
| Metrics log (`metrics.csv`) | `models/<module>/training_run/checkpoints/metrics.csv` | `experiments/<Module>/<timestamp>/checkpoints/metrics.csv` (Drive) |
| TensorBoard logs | `models/<module>/training_run/logs/` | `experiments/<Module>/<timestamp>/logs/` (live) + `.../tensorboard/` (archival copy) (Drive) |
| Evaluation plots/report | `results/<module>/` | `experiments/<Module>/<timestamp>/evaluation/` (Drive) |
| Prediction samples | *(module-specific, not centrally defined for CLI runs)* | `experiments/<Module>/<timestamp>/predictions/` (Drive) |
| Run metadata | *(not tracked for CLI runs)* | `experiments/<Module>/<timestamp>/metadata.json` (Drive) |
| Session/setup logs | N/A | `logs/setup_<timestamp>.json` (Drive, global) |

---

## Rules

Restated from `PROJECT_CODE.md` (canonical copy) since this document is checked against them:

1. **Never modify `datasets/*/raw`.** Preprocessing output goes to the matching `processed/`
   folder; nothing ever writes back into `raw/`.
2. **One stage at a time.** Explain the existing implementation, explain why it needs to change,
   propose a plan, wait for approval, implement, verify integration, stop.
3. **No fabricated metrics.** If a model has not been trained, state plainly that no real
   evaluation exists -- never substitute placeholder or simulated numbers.
4. **No placeholder evaluation.** Unit tests may use synthetic/temporary data to verify function
   correctness; every actual pipeline evaluation must run against real data.
5. **Train only in Colab.** All real model training happens in Google Colab, against Drive-hosted
   data; local runs (`train_image_quality.py`, etc.) are for quick smoke tests or as the reference
   a notebook mirrors, not for producing results that get reported.
