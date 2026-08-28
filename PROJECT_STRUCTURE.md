# Project Structure

Master architectural reference for this repository. `PROJECT_CODE.md` defines the target
architecture and development rules; `IMPLEMENTATION_PLAN.md` tracks the gap between the baseline
and that target; this document describes **where everything actually lives** and **how the
pieces fit together**, kept in sync with the repository as it exists today. The canonical
end-to-end pipeline diagram lives in `README.md`; this document does not repeat it, to avoid two
diagrams drifting out of sync — see README's "Master Pipeline" section.

---

## Repository Overview

| Folder | Purpose |
|---|---|
| *(repository root)* | Flat, top-level Python modules -- see "Source Code Organization" below; this repository does **not** use a `src/` layout. |
| `training/` | Reusable, model-agnostic training framework (`Trainer`, callbacks, losses, metrics, optimizers). No models or dataset loading. |
| `evaluation/` | Reusable, model-agnostic evaluation framework (`Evaluator`, metrics, visualization). Operates on prediction arrays only. |
| `pipeline/` | Abstract base classes (`TrainableStage`, `InferenceStage`, `SegmentationStage`, `ClassificationStage`) fixing the contract future pipeline stages must implement. Defines no models. |
| `datasets/` | Real, local datasets: `EyeQ/`, `APTOS2019/`, `IDRiD/`. `raw/` subfolders are read-only; never modified in place. Vessel Segmentation (Stage 3) uses a vendored pretrained checkpoint, not a dataset here. |
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
  stopping, `ReduceLROnPlateau`, TensorBoard, resume support), plus `losses.py` (including
  `bce_dice_loss`) and `metrics.py` (including `dice_coefficient`/`iou_score` via
  `build_metrics("segmentation")`), `optimizers.py`, `callbacks.py`. This entire package is
  TensorFlow/Keras-based. Any trainable stage implemented in TensorFlow builds its own model
  and dataset, then hands both to `Trainer` -- this is how `train_image_quality.py` already
  works, and how Lesion Segmentation (Attention U-Net, fixed to TensorFlow) works too. Vessel
  Segmentation does not use this package at all: it is a pretrained PyTorch checkpoint (LWNet,
  `SEGMENTATION_ARCHITECTURE.md` §2/§6), integrated for inference only, with no training loop of
  its own in this project.
- **`evaluation/`** -- `Evaluator` (accuracy/precision/recall/F1/confusion matrix/ROC/AUC/QWK/
  calibration) plus `visualization.py`'s plotting helpers. Operates purely on `(y_true, y_pred,
  y_proba)` arrays -- classification-oriented; segmentation stages (Vessel, Lesion) use
  `training.metrics`'s Dice/IoU instead, not this module.
- **`pipeline/`** -- `TrainableStage` / `InferenceStage` / `SegmentationStage` /
  `ClassificationStage` ABCs. Establishes the contract for stages **implemented from this point
  forward**, including Vessel Segmentation; the existing IQA module predates this package and
  does not inherit from it.
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
| **EyeQ** | Image quality classification (`Good`/`Usable`/`Reject`) | **Active** -- trains and evaluates Stage 1 (Image Quality Assessment) today, via `image_quality_dataset.py` / `train_image_quality.py` / `colab/notebooks/stage01_iqa.ipynb`. | Continues to gate every later stage: only `Good`/`Usable` images should reach Stage 2 preprocessing. Used only for Stage 1. |
| **APTOS2019** | Diabetic retinopathy severity classification (5 classes, 0-4) | Referenced by the pre-refactor baseline scripts (`efficientnet_model.py`, `swin_transformer.py`, `train_hybrid_model.py`) at the repository root. | Primary training set for Stage 8 (CORN Classification) once the target architecture's classification head is implemented. |
| **IDRiD** | Lesion segmentation (microaneurysms, haemorrhages, hard/soft exudates, optic disc); also grading | Not yet consumed by any implemented stage. | Trains Stage 4 (Lesion Segmentation, Attention U-Net) and supports downstream grading tasks. |

**EyePACS is historical only.** It was used once, outside this repository, to reconstruct the
official EyeQ dataset via EyeQ's own generation repository. EyePACS is not present under
`datasets/`, is not read by any script here, and is not required to reproduce or run this
repository -- see `PROJECT_CODE.md`'s Datasets section for the full history.

**Vessel Segmentation (Stage 3) does not train within this project.** It integrates a pretrained,
externally-sourced checkpoint (LWNet, trained by its own authors on DRIVE) for inference only, and
needs no dataset of its own under `datasets/`. An intermediate design added DRIVE and CHASE_DB1 as
project datasets specifically to train a "Baseline U-Net" within this project instead; that design
was itself superseded by the current pretrained-LWNet design — neither DRIVE nor CHASE_DB1 is a
project dataset today. See `SEGMENTATION_ARCHITECTURE.md` for the full specification, including
its design-history appendix documenting this full chronology.

---

## Dataset Flow

Every dataset consumed by this pipeline follows the same lifecycle — Stage 02 is dataset-independent and is never described separately per dataset:

```
Raw Dataset (datasets/<name>/raw/, read-only)
    │
    ▼
Stage 01 IQA gate (EyeQ only — the only dataset this gate applies to today;
    other datasets are not currently gated by IQA)
    │
    ▼
Accepted Images
    │
    ▼
Stage 02 Preprocessing (Gamma Correction → CLAHE, RGB in, RGB out,
    deterministic, dataset-agnostic)
    │
    ▼
Processed Dataset (datasets/<name>/processed/, generated once, reused by
    every downstream consumer)
    │
    ▼
Stage-specific Dataset Loader (image_quality_dataset.py,
    lesion_segmentation_dataset.py, ...)
    │
    ▼
Training / Inference
```

Ground-truth mask/label data (IDRiD's lesion masks and grading CSVs) never enters Stage 02 — only fundus images do. Each stage-specific dataset loader reads mask/label data directly from `raw/`, in parallel with reading the corresponding processed image from `processed/`.

Vessel Segmentation (Stage 3) does not appear in this lifecycle at all as a *dataset* consumer — it has no `datasets/<name>/` directory of its own. It still consumes Stage 02's processed output as its *inference* input, the same as every other downstream stage; it just has no dataset loader or training step to reach that point.

---

## Pipeline Overview

The full 11-stage target architecture (`PROJECT_CODE.md`). "Status" reflects what's actually
implemented and verified today, not aspirational state. See README's "Master Pipeline" diagram
for the visual end-to-end flow.

### 1. Image Quality Assessment
- **Purpose:** Gate low-quality fundus images before they reach the rest of the pipeline.
- **Input:** Raw fundus images (EyeQ, or any unlabeled folder via `image_quality_inference.py`).
- **Output:** `Good` / `Usable` / `Reject` classification + per-class confidence.
- **Training dataset:** EyeQ. Used only for Stage 1.
- **Inference output:** `{"label", "class_index", "confidence", "probabilities"}` per image (see `image_quality_inference.predict_quality`).
- **Dependencies:** None (first stage).
- **Status:** **Completed -- Verified -- Baseline Established.** `image_quality_dataset.py`, `image_quality_model.py`, `train_image_quality.py`, `evaluate_image_quality.py`, `image_quality_inference.py`, `colab/notebooks/stage01_iqa.ipynb`. Trained end-to-end in Google Colab (experiment `2026-08-05_09-11-28`) -- held-out test accuracy 88.05%, F1 86.12%, AUC 96.48%, QWK 0.8987; see `docs/FIRST_TRAINING_CHECKLIST.md`'s completed-run record for full detail.

### 2. Image Preprocessing
- **Purpose:** Gamma Correction and CLAHE only, on RGB images. No green-channel extraction, Ben Graham processing, median denoising, histogram equalization, resizing, or augmentation — all explicitly excluded from Stage 02, per the frozen architecture.
- **Input:** Images passing Stage 1's quality gate.
- **Output:** Normalized RGB PNG images in `datasets/*/processed/`, at native (unresized) resolution.
- **Training dataset:** N/A (deterministic image transform, not a trained model).
- **Inference output:** Preprocessed RGB image, ready for Stages 3+.
- **Dependencies:** Stage 1 (now completed and verified -- see above).
- **Status:** **Frozen, implementation-ready.** `image_preprocessing.py` (repository root) already implements the transforms via `config.py`'s `PREPROCESSING_PROFILES`; not yet wired into a Colab notebook (`colab/notebooks/stage02_preprocessing.ipynb` is a template only). Stage 02 is model-agnostic by design: any model-specific preprocessing (channel adaptation, resizing, normalization) belongs inside the consuming stage, never here.

### 3. Vessel Segmentation
- **Purpose:** Segment retinal vasculature.
- **Input:** Preprocessed RGB images (Stage 2).
- **Output:** Single-channel vessel probability map, `(H, W, 1)`, values in `[0, 1]`.
- **Model:** Pretrained LWNet (`wnet`, MIT-licensed, external) — **inference only, not trained within this project.**
- **Training dataset:** None. LWNet's vendored checkpoint was trained by its original authors on DRIVE, entirely outside this project; Stage 3 stages, trains, or evaluates nothing.
- **Dependencies:** Stage 2.
- **Status:** Not implemented. Design finalized (`SEGMENTATION_ARCHITECTURE.md` §1.2/§2); `colab/notebooks/stage03_vessel_segmentation.ipynb` is a template only. Unlike every other trainable stage, this one has no dataset → training → evaluation → export lifecycle — it vendors a pretrained checkpoint and exposes only `load()`/`predict()`/`predict_batch()` (`pipeline.SegmentationStage`, §5). `training.Trainer` / `training.build_metrics("segmentation")` (TensorFlow-specific) do not apply to this stage.

### 4. Lesion Segmentation
- **Purpose:** Segment DR lesions (microaneurysms, hemorrhages, exudates).
- **Input:** Preprocessed RGB image (Stage 2) + vessel probability map (Stage 3), concatenated — `(H, W, 4)`.
- **Output:** Four lesion probability maps (Microaneurysm, Haemorrhage, Hard Exudate, Soft Exudate) — `(H, W, 4)`.
- **Training dataset:** IDRiD (segmentation subset).
- **Inference output:** Per-lesion-class segmentation mask.
- **Dependencies:** Stage 2, Stage 3 (Lesion Segmentation training requires Stage 3's trained model to generate vessel-mask inputs first).
- **Status:** Not implemented. Design finalized in `SEGMENTATION_ARCHITECTURE.md`; `colab/notebooks/stage04_lesion_segmentation.ipynb` is a template only.

### 5. Local Feature Extraction
- **Purpose:** Fine-grained, lesion-level feature extraction (Adaptive Multi-Kernel CNN).
- **Input:** Preprocessed RGB image (3 channels) + vessel probability map (1 channel) + four lesion probability maps (4 channels), concatenated into a single **8-channel tensor**, `(H, W, 8)`.
- **Output:** Local feature vectors/maps.
- **Training dataset:** Likely trained jointly with Stages 6-8 (see note below), not independently.
- **Dependencies:** Stages 2, 3, 4.
- **Status:** Not implemented.

### 6. Global Feature Extraction
- **Purpose:** Whole-image, long-range, segmentation-independent feature extraction (Dual-Scale Swin Transformer) — complementary to Stage 5's segmentation-anchored local features, not a duplicate of them.
- **Input:** Preprocessed RGB image (Stage 2), directly — parallel to, not sequential with, Local Feature Extraction. Never Stage 5's `local_features`, never vessel or lesion probability maps — Stage 6 has no dependency on Stage 3 or Stage 4 at all.
- **Input shape:** `(256, 256, 3)`, float32, `[0, 1]`. Resized to 256x256 inside Stage 6's own pipeline — Stage 02 itself is never resized.
- **Architecture:** two parallel Swin Transformer branches at different patch granularities, reusing this repository's existing `PatchEmbed`/`BasicLayer`/`PatchMerging` classes (`swin_transformer.py`) unmodified in their own definitions:
  - **Branch A (fine):** `patch_size=4`, `embed_dim=96`, `depths=[2,2,6,2]`, `heads=[3,6,12,24]`, `window_size=8` → final `8x8x768`. Matches this project's existing `create_swin_tiny_model()` configuration exactly; per-head dimension is 32 at every stage, following the original Swin Transformer paper's own design rule (Liu et al., ICCV 2021).
  - **Branch B (coarse):** `patch_size=8`, `embed_dim=96`, `depths=[2,2,6]`, `heads=[3,6,12]`, `window_size=8` → final `8x8x384`. The (4, 8) patch-size pairing is a literature-informed adaptation (DS-TransUNet, Lin et al., arXiv:2106.06716); the one-fewer-stage depth is a project-specific engineering choice, not taken from that paper — Branch A's total downsampling ($4\times2^3=32$) equals Branch B's ($8\times2^2=32$), so both branches converge on an identical, position-aligned final grid with no resize/interpolation step needed before fusion.
  - **Fusion:** channel-wise concatenation only (`8x8x768` + `8x8x384` = `8x8x1152`), flattened to a token sequence. No post-concatenation projection, no cross-branch or cross-attention interaction, no pooling, no classification head — Stage 7 (Adaptive Cross-Attention) and RACAF are responsible for any further interaction or projection; RACAF's own `Ghat = W_r * GAP(G) + b_r` formula (`RACAF_ARCHITECTURE.md` §5) already projects Stage 6's raw output, so a second projection here would be redundant.
  - **Initialization:** random, both branches — no pretrained weights.
- **Output:** `global_features`, shape `(B, 64, 1152)` — un-pooled token sequence ($N=8\times8=64$ tokens, $C=1152$ channels).
- **Training dataset:** APTOS 2019 only, sharing the exact train/val split Stage 5's own loader uses. No standalone Stage 6 training objective — trained jointly with Stages 5, 7, 8, and RACAF, through CORN's eventual ordinal loss (`GlobalFeatureExtractionStage.train()` raises `NotImplementedError`, mirroring `LocalFeatureExtractionStage`).
- **Dependencies:** Stage 2 only.
- **Files:** `swin_transformer.py` (`create_dual_scale_swin_model()`, `GlobalFeatureExtractionStage`) — the function bodies of `create_swin_tiny_model()`/`create_hybrid_model()` are byte-identical to before this stage existed (verified via `git diff` hunk boundaries); `global_feature_extraction_dataset.py` (new, reuses `local_feature_extraction_dataset.py`'s Stage-02-application helpers rather than duplicating them).
- **Shared Swin infrastructure fixes:** three pre-existing defects in `SwinTransformerBlock.build()`/`PatchMerging.call()` (a `float64`/`float32` dtype mismatch, a `tf.Variable`-inside-graph-tracing error, and a symbolic-tensor-as-Python-bool assertion) blocked building any Swin block with `shift_size>0` under the currently-installed Keras version — independently reproduced against the pre-session code and confirmed numerically identical after fixing (same computed values, only the execution mechanism changed). `create_hybrid_model()` is unaffected (its one `SwinTransformerBlock` uses `shift_size=0`, never exercising the fixed code path). A fourth, unrelated pre-existing defect (`SwinTransformer.__init__`'s `self.layers = []` shadowing Keras 3's reserved `Model.layers`, breaking `create_swin_tiny_model()`) was deliberately **left unfixed** — `create_swin_tiny_model()`/`SwinTransformer` have no active caller anywhere in this project (verified by `git grep`; every real consumer of `swin_transformer.py` — `train_hybrid_model.py`, `test_hybrid_model.py`, `explainable_ai.py`, `bayesian_inference.py`, `Visualization_Scripts/create_architecture_diagram.py` — imports `create_hybrid_model` or the low-level building blocks directly, never `SwinTransformer`), so fixing it was judged an unnecessary scope expansion; a regression test pins the current (still-broken) behavior so a future accidental fix or further breakage is caught rather than silently drifting.
- **Checkpoint strategy:** weights-only (`.weights.h5`, `model.save_weights()`/`model.load_weights()` via `GlobalFeatureExtractionStage.save()`/`.load()`), not a full-architecture `.keras` file — because the custom Swin layer classes have no `get_config()` (an existing gap, not fixed here), and because weights-only is the shared `training/callbacks.py`/`training/trainer.py` framework's own **default** (`save_weights_only=True`), not a deviation from it — Stage 4 (`LesionSegmentationStage`) is the one that explicitly opts into full `.keras` checkpoints, which Stage 6 does not need to replicate. The eventual joint Stage 05-08 + RACAF training run will itself default to weights-only checkpointing unless a future implementer deliberately overrides it.
- **Novelty status:** not a research contribution — Swin's core mechanism, the dual-patch-size branch concept, and the (4,8) pairing are all established/cited (see the Stage 06 design-resolution record). RACAF remains the project's sole research innovation.
- **Status:** Implemented and unit-tested. **Not trained, not frozen.**

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

## Stage Dependencies

Every stage depends on the stage(s) immediately before it in the pipeline; no stage bypasses another:

```
Stage 01 → Stage 02 → Stage 03 → Stage 04 → Stage 05 → Stage 06 → Stage 07 → Stage 08
```

Stage 05 additionally depends on Stage 03 and Stage 04 directly (not only transitively through Stage 02), since it consumes both stages' outputs concatenated with the processed image. Stage 06 depends only on Stage 02. Stage 09 and Stage 10 depend on Stage 08's trained classifier and are inference-only branches, not part of the main training chain. Stage 11 depends on every prior stage having a real, trained model — it is never populated with placeholder numbers ahead of that.

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
  never overwritten, resumable. `colab_config.py`'s `PIPELINE_MODULES` still includes
  `"VesselSegmentation"` alongside `"IQA"`, `"LesionSegmentation"`, and `"FinalClassification"` --
  this entry is now used only to resolve `exported_models/VesselSegmentation/` (where the vendored
  LWNet checkpoint lands), since Stage 3 never populates `experiments/VesselSegmentation/` with an
  actual training run.
- **Model export:** the best checkpoint from each run is copied to a stable
  `exported_models/<Module>/best_model.keras` on Drive (overwritten by each new "best" run), kept
  separate from that run's own permanent `checkpoints/best.keras` archive.
- **TensorBoard:** launched against the current experiment's live `logs/` directory; any past
  experiment's `logs/` can be pointed at directly to review it later.
- **Resuming:** point `RESUME_EXPERIMENT_DIR` at a previous experiment's folder instead of
  leaving it `None` -- see `colab/README.md`'s "How to resume training".
- **Stage 3 does not follow Stage 1's 12-step training workflow.** It has no dataset to stage, no
  training loop, and no evaluation run — its notebook is scoped to Bootstrap → Setup →
  Environment Verification → Checkpoint Integration → Inference Verification → Export/Final
  Summary, confirming the vendored LWNet checkpoint loads and predicts correctly.

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
| Trained models (`best.keras`, `last.keras` for TensorFlow-based modules) | `models/<module>/training_run/checkpoints/` | `experiments/<Module>/<timestamp>/checkpoints/` (Drive) |
| Exported "current best" model | `models/<module>/best_model.keras` (TensorFlow-based modules) or `models/vessel_segmentation/best_model.pth` (vendored LWNet checkpoint, not a training output — see `SEGMENTATION_ARCHITECTURE.md` §6) | `exported_models/<Module>/best_model.keras` or `exported_models/VesselSegmentation/best_model.pth` accordingly (Drive) |
| Metrics log (`metrics.csv`) | `models/<module>/training_run/checkpoints/metrics.csv` | `experiments/<Module>/<timestamp>/checkpoints/metrics.csv` (Drive) |
| TensorBoard logs | `models/<module>/training_run/logs/` | `experiments/<Module>/<timestamp>/logs/` (live) + `.../tensorboard/` (archival copy) (Drive) |
| Evaluation plots/report | `results/<module>/` | `experiments/<Module>/<timestamp>/evaluation/` (Drive) |
| Prediction samples | *(module-specific, not centrally defined for CLI runs)* | `experiments/<Module>/<timestamp>/predictions/` (Drive) |
| Run metadata | *(not tracked for CLI runs)* | `experiments/<Module>/<timestamp>/metadata.json` (Drive) |
| Session/setup logs | N/A | `logs/setup_<timestamp>.json` (Drive, global) |

Vessel Segmentation (`models/vessel_segmentation/`) is a special case in this table: it has no `training_run/` and no `results/` directory, since nothing about it is trained or evaluated within this project — only the vendored checkpoint itself lands there.

---

## Rules

Restated from `PROJECT_CODE.md` (canonical copy) since this document is checked against them:

1. **Never modify `datasets/*/raw`.** Preprocessing output goes to the matching `processed/`
   folder; nothing ever writes back into `raw/`. Ground-truth masks/labels are never
   preprocessed — only fundus images pass through Stage 02.
2. **One stage at a time.** Explain the existing implementation, explain why it needs to change,
   propose a plan, wait for approval, implement, verify integration, stop.
3. **No fabricated metrics.** If a model has not been trained, state plainly that no real
   evaluation exists -- never substitute placeholder or simulated numbers.
4. **No placeholder evaluation.** Unit tests may use synthetic/temporary data to verify function
   correctness; every actual pipeline evaluation must run against real data.
5. **Train only in Colab.** All real model training happens in Google Colab, against Drive-hosted
   data; local runs (`train_image_quality.py`, etc.) are for quick smoke tests or as the reference
   a notebook mirrors, not for producing results that get reported.
6. **Stage 02 stays model-agnostic.** Any model-specific preprocessing (channel adaptation,
   resizing, normalization) belongs inside the consuming stage's own adapter, never inside Stage 02.
7. **Stage 02 preprocessing is deterministic and generated once.** Each dataset is preprocessed
   exactly once; processed outputs are stored and reused by every downstream stage. No downstream
   stage regenerates deterministic preprocessing outputs.
8. **Every trainable stage is modular.** It owns its own dataset, model, training, evaluation,
   inference, and exported model, and communicates with other stages only through its documented
   input/output contract -- never by depending on another stage's internal implementation
   (including which framework that stage happens to use).
