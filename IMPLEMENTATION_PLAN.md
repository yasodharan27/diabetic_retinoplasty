# Implementation Plan — Diabetic Retinopathy Detection

This document compares the current baseline repository against the target architecture defined in `PROJECT_CODE.md` and lays out a module-by-module roadmap to close the gap. Sections describing the pre-refactor baseline (§1) are historical and unaffected by the architecture freeze; sections describing the target architecture (§2 onward) reflect the frozen design in `PROJECT_CODE.md` / `SEGMENTATION_ARCHITECTURE.md` / `PROJECT_STRUCTURE.md`.

---

## 0. Implementation Rules

Canonical copy lives in `PROJECT_CODE.md` under "Implementation Rules" -- restated here since this is the document each roadmap step gets checked against:

This is a production/research project, not a demonstration project.

1. Do not implement placeholder logic, simulated outputs, fake metrics, or dummy pipelines.
2. Unit tests may use synthetic or temporary data only, to verify correctness of individual functions.
3. All actual project functionality must operate on the real datasets: EyeQ, DRIVE, CHASE_DB1, APTOS2019, and IDRiD. EyePACS was used only once, historically, to reconstruct EyeQ, and is not required to reproduce or run this repository (see `PROJECT_CODE.md`'s Datasets section).
4. Do not create "toy" implementations intended to be replaced later.
5. Every module should be fully implementable and immediately usable in the final pipeline.
6. If verification of a full dataset would require hours of execution, perform lightweight correctness tests only -- never replace the actual implementation with a simplified version.
7. Do not fabricate evaluation results or performance metrics. If a model has not been trained, clearly state that no real evaluation exists.
8. Every trainable module must include: dataset loader, model, training, evaluation, inference, and a deployment interface.

The final objective is a real-world end-to-end diabetic retinopathy diagnosis pipeline, not an academic prototype.

---

## 1. Current Architecture (as implemented today)

### 1.1 Configuration
All scripts load path/config values from environment variables via `python-dotenv` (`.env_sample` lists the expected keys: `BASE_PATH`, `MODEL_PATH`, `PROCESSED_TEST_DIR`, `RESULTS_DIR`, `PROCESSED_IMAGES_DIR`, `GAN_IMAGES_DIR`, `IMAGE_DIR`, `TEST_IMAGE_DIR`, `CSV_PATH`, `RETRAINED_MODEL_PATH`, `OUTPUT_DIR`, `MODEL_SAVE_PATH`, `RESULTS_PATH`, `IMAGE_SIZE`, `CHANNELS`, `NUM_CLASSES`, `BATCH_SIZE`, `EPOCHS`, `LATENT_DIM`). `config.py` now centralizes this (see `PROJECT_STRUCTURE.md`); the description below documents the historical, pre-refactor baseline scripts, which still exist unmodified in the repository root.

### 1.2 Data
The pre-refactor baseline is built around the **APTOS 2019** CSV format (`id_code`, `diagnosis` columns, PNG fundus images, 5 severity classes 0–4). There is no EyeQ or EyePACS ingestion code in the baseline scripts.

### 1.3 Preprocessing (`pre_process_with_dataset_download.py`, `pre_process_test_and_train.py`) — historical baseline, superseded by Stage 02
Both scripts implement the same pipeline (the second variant adds a test-set branch):
- Green channel extraction (`image[:, :, 1]`)
- Ben Graham preprocessing (`cv2.addWeighted` with Gaussian blur — local contrast enhancement)
- CLAHE (`clipLimit=2.0`, `tileGridSize=(8,8)`)
- Median blur denoising (kernel 5)
- Resize to 224×224
- Normalize to [0,1], output single-channel array
- Stratified train/val split saved as CSVs, `tf.data.Dataset` generators built from disk

This baseline pipeline is **not** the target architecture's Stage 02. Per the frozen architecture (`PROJECT_CODE.md`, `SEGMENTATION_ARCHITECTURE.md`), Stage 02 is Gamma Correction + CLAHE only, on RGB, with no green-channel extraction, Ben Graham processing, median denoising, resizing, or augmentation. See §3 below for the finalized gap analysis.

### 1.4 Classification models
Three parallel, only loosely related classification paths exist:
- **`efficientnet_model.py`** — baseline `EfficientNetB0` (grayscale input tiled to 3 channels), GAP → Dense(256) → Dropout → Dense(5, softmax), categorical cross-entropy.
- **`swin_transformer.py`** — a full from-scratch Swin Transformer implementation (`PatchEmbed`, `WindowAttention`, `SwinTransformerBlock`, `PatchMerging`, `BasicLayer`, `SwinTransformer`), plus `create_swin_tiny_model()` (pure Swin) and `create_hybrid_model()`, which is the repo's current "hybrid" architecture: `EfficientNetB0` backbone → **one** `SwinTransformerBlock` (named `swin_refine`) applied directly to the CNN feature map → GAP → Dense(256) → Dropout → Dense(5, softmax). This is a sequential CNN-then-single-attention-block design, not a dual-branch fusion.
- **`dr_classifier.py`** — a separate experiment: `EfficientNetB2` classifier trained on a combined real+GAN-synthetic dataset, with a `compare_with_without_synthetic()` ablation and its own training/eval/plotting utilities.
- **`train_hybrid_model.py`** — trains `create_hybrid_model()` with focal loss, mixed precision (`mixed_float16`), Keras `RandomFlip/RandomRotation/RandomContrast` augmentation, and `compute_class_weight`-based class weighting.
- **`retrain_efficientnet.py`** — fine-tunes an already-trained EfficientNet `.h5` model with focal loss and manually computed class weights.
- **`testing_efficientnet_model.py`** / **`test_hybrid_model.py`** — inference scripts that load a saved model (the hybrid version needs a `custom_object_scope` including a hand-rolled `Cast` layer to deserialize the mixed-precision graph) and emit prediction CSVs + bar-chart visualizations.
- All classification heads are **nominal** (softmax + categorical/focal cross-entropy) — there is no ordinal-aware loss or head anywhere.

### 1.5 Synthetic data generation (`dr_gan.py`)
A conditional GAN ("DR-GAN++" per README) with a label-conditioned generator (noise + class embedding → transposed-conv upsampling → 224×224×1 image) and discriminator, trained adversarially, used to oversample minority DR-severity classes into `GAN_IMAGES_DIR`. This is not part of the target pipeline in `PROJECT_CODE.md` but is a real, working component used by `dr_classifier.py`.

### 1.6 Uncertainty estimation (`bayesian_inference.py`)
Implements Monte Carlo Dropout: clones the trained hybrid model, forces `training=True` at inference, runs `num_samples=50` stochastic forward passes, and computes mean prediction, per-class standard deviation ("uncertainty"), and predictive entropy. Produces uncertainty bar charts, a class-wise uncertainty distribution analysis, and a reliability diagram. **Note:** the reliability diagram explicitly uses *simulated* ground-truth labels (`simulated_true_classes = np.random.randint(...)`) — this is called out in the code's own print statement and is not a real calibration measurement.

### 1.7 Explainability (`explainable_ai.py`)
Standard (non-plus-plus) Grad-CAM: builds a sub-model exposing the `swin_refine` layer's output and the final prediction, computes gradients of the predicted class w.r.t. that feature map, pools them into channel weights, and produces a ReLU'd heatmap overlaid on the original image.

### 1.8 Publication/reporting utilities (`Visualization_Scripts/`, `formula.py`)
`Visualization_Scripts/` (architecture diagram, performance metrics, explainability summary, publication tables, summary dashboard, and an orchestrating `generate_all_visualizations.py` that shells out via `subprocess`) and `formula.py` (renders LaTeX equation images into `formula/`) are documentation/report generators, not part of the inference pipeline. They consume the outputs of the scripts above (e.g. `swin_transformer.create_hybrid_model`, saved metrics CSVs).

### 1.9 Supporting assets
`assets/` (README images), `formula/` (rendered equation PNGs), `research_papers/` (reference PDFs — several of these look like they map directly onto the target modules below, e.g. Swin Transformer papers, DR-GAN++, attention-based DR grading, and are worth revisiting per-module when each is implemented).

---

## 2. Target Architecture (frozen, per `PROJECT_CODE.md` / `SEGMENTATION_ARCHITECTURE.md`)

11-stage pipeline:

1. Image Quality Assessment — EfficientNetB0
2. Image Preprocessing — Gamma Correction, CLAHE (RGB in, RGB out; no green-channel extraction, Ben Graham, median denoise, histogram equalization, resizing, or augmentation)
3. Vessel Segmentation — Baseline U-Net, **trained within this project** on DRIVE + CHASE_DB1
4. Lesion Segmentation — Attention U-Net, trained on IDRiD
5. Local Feature Extraction — Adaptive Multi-Kernel CNN
6. Global Feature Extraction — Dual-Scale Swin Transformer
7. Feature Fusion — Adaptive Cross-Attention
8. Ordinal Classification — CORN
9. Uncertainty Estimation — Monte Carlo Dropout
10. Explainability — Grad-CAM++, SHAP, Attention Rollout
11. Evaluation

Approved datasets: **EyeQ** (image quality, Stage 01 only), **DRIVE** and **CHASE_DB1** (Vessel Segmentation training, Stage 03 only), **APTOS 2019** (classification), and **IDRiD** (lesion segmentation and grading). EyePACS itself is not part of the implemented training or inference pipeline and is not required to reproduce it. No other datasets permitted without explicit request.

---

## 3. Gap Analysis

| # | Target Module | Target Model | Current State | Status |
|---|---|---|---|---|
| 1 | Image Quality Assessment | EfficientNetB0 (quality classifier) | Implemented, trained, verified — see `PROJECT_STRUCTURE.md`'s Stage 1 results. | **Completed** |
| 2 | Image Preprocessing | Gamma Correction, CLAHE only (RGB in/out, deterministic, generated once) | `image_preprocessing.py` implements exactly this. No green-channel extraction, Ben Graham, median denoise, or resize — all explicitly excluded from Stage 02 per the frozen architecture. | **Frozen / implementation-ready** |
| 3 | Vessel Segmentation | Baseline U-Net, trained on DRIVE + CHASE_DB1 | No code exists yet. Design finalized: DRIVE and CHASE_DB1 are now approved project datasets specifically to make this stage trainable within the project (see `SEGMENTATION_ARCHITECTURE.md` §1.2/§2, and its design-history appendix for why an earlier design used a pretrained, inference-only model instead). | **Missing — design finalized, ready to implement** |
| 4 | Lesion Segmentation | Attention U-Net | No code exists. Design finalized in `SEGMENTATION_ARCHITECTURE.md` §3 — input is the processed RGB image concatenated with Stage 3's vessel probability map (4 channels total). | **Missing — design finalized** |
| 5 | Local Feature Extraction | Adaptive Multi-Kernel CNN | No dedicated "local" feature extractor exists. Input contract finalized: RGB image + vessel map + 4 lesion maps, concatenated into an 8-channel tensor (`SEGMENTATION_ARCHITECTURE.md` §4). | **Missing** |
| 6 | Global Feature Extraction | Dual-Scale Swin Transformer | A complete, hand-written single-scale Swin Transformer exists (`swin_transformer.py`), and `create_swin_tiny_model()` runs it standalone. The *hybrid* model only bolts on a single `SwinTransformerBlock` for feature refinement after a CNN backbone — no dual-scale windowing exists. Reusable foundation, wrong topology. Consumes the processed RGB image directly; any resizing it needs is internal to this stage (Stage 02 stays model-agnostic and unresized). | **Partially implemented** |
| 7 | Feature Fusion | Adaptive Cross-Attention | The current "fusion" is a linear sequence (CNN → one Swin block → GlobalAveragePooling2D → Dense), not an attention-based fusion of two independent feature streams. | **Missing** |
| 8 | Ordinal Classification | CORN | All classifiers use plain softmax + categorical/focal cross-entropy — nominal, not ordinal. No CORN head, no rank-consistent logits, no QWK metric in the baseline scripts (QWK is, however, already implemented and reusable in `evaluation/metrics.py` / `training/metrics.py`). | **Missing** |
| 9 | Uncertainty Estimation | Monte Carlo Dropout | Fully implemented in `bayesian_inference.py`: MC sampling, mean/std, predictive entropy, uncertainty visualizations. Reliability diagram uses simulated labels (documented limitation, not a bug). Will need re-pointing at whatever model results from steps 5–8. | **Implemented** (needs integration once the classifier changes) |
| 10 | Explainability | Grad-CAM++, SHAP, Attention Rollout | Only vanilla Grad-CAM exists (`explainable_ai.py`), hard-coded to the `swin_refine` layer name from the current hybrid model. Grad-CAM++, SHAP, and Attention Rollout are all absent. | **Partially implemented** |
| 11 | Evaluation | — | Confusion matrix, classification report, accuracy/AUC exist per-script but are duplicated across files rather than a single evaluation module, and none compute ordinal-appropriate metrics (QWK) or real (non-simulated) calibration against the target architecture's models. | **Partially implemented** |

**Non-target component present in the repo:** `dr_gan.py` (conditional GAN for synthetic minority-class oversampling) is real and working but is not part of the 11-stage target pipeline. Per the "reuse existing components" rule, it should be kept and can still feed the ordinal classifier's training data, but it is not one of the roadmap's numbered modules.

**Dataset gap — resolved.** Vessel Segmentation (Stage 3) and Lesion Segmentation (Stage 4) both require pixel-level mask ground truth. Neither EyeQ, APTOS 2019, nor IDRiD's grading/localization subsets ship vessel masks; IDRiD's segmentation subset ships lesion (and Optic Disc) masks only. This gap is resolved by adding **DRIVE and CHASE_DB1** as officially approved project datasets, used exclusively to train Stage 3's Baseline U-Net; Lesion Segmentation continues to train on IDRiD's segmentation subset. Both Vessel Segmentation and Lesion Segmentation are now trained within this project — see `SEGMENTATION_ARCHITECTURE.md` for the full specification, including the design-history appendix documenting the earlier pretrained-inference-only alternative that this supersedes.

---

## 4. Development Roadmap

Ordered to match the target pipeline's numbering, since each stage after preprocessing consumes the previous stage's output. A "Step 0" is added first for shared infrastructure every later step depends on.

### Step 0 — Shared Infrastructure & Config Extension
- **Why:** New datasets (DRIVE, CHASE_DB1, IDRiD) and new model stages (segmentation, fusion, ordinal head) need new path/config variables before any of them can be built, following the repo's existing `.env`-driven convention.
- **Files to modify:** `.env_sample` (append new keys, don't remove existing ones); `config.py` (add `VESSEL_SEG_MODEL_DIR` / `VESSEL_SEG_RESULTS_DIR`, mirroring `IQA_MODEL_DIR` / `IQA_RESULTS_DIR` — DRIVE/CHASE_DB1 raw+processed paths need no new dataclass, they resolve through the existing generic `dataset_raw_dir()` / `dataset_processed_dir()` helpers).
- **Expected output:** extended config documenting the additional variables Stage 3 needs. No behavior change to existing scripts.

### Step 1 — Image Quality Assessment (EfficientNetB0)
**Status: Completed, verified, trained, exported.** See `PROJECT_STRUCTURE.md` for full detail. No further action.

### Step 2 — Preprocessing (Gamma Correction + CLAHE)
- **Why:** `PROJECT_CODE.md` specifies Stage 02 as exactly Gamma Correction + CLAHE on RGB — no other transform.
- **Status:** `image_preprocessing.py` already implements this. Frozen and implementation-ready; no further architectural decision remains before Stage 02 is wired into `colab/notebooks/stage02_preprocessing.ipynb` and run once, per-dataset, per `PROJECT_CODE.md`'s Dataset Policy.

### Step 3 — Vessel Segmentation (Baseline U-Net)
- **Why:** Vessel maps are a prerequisite input for Lesion Segmentation (Stage 4) and Local Feature Extraction (Stage 5) in the target architecture.
- **Datasets:** DRIVE + CHASE_DB1, run through Stage 02's own RGB → Gamma → CLAHE pipeline before training, so the model trains on the same distribution it will see at inference time on EyeQ/APTOS/IDRiD-derived images.
- **New files:** `vessel_segmentation_dataset.py`, `vessel_segmentation_model.py`, `train_vessel_segmentation.py`, `evaluate_vessel_segmentation.py`, `vessel_segmentation_inference.py` — mirroring Stage 1's exact file set and structure (dataset → model → train → evaluate → inference).
- **Colab notebook:** yes — `colab/notebooks/stage03_vessel_segmentation.ipynb`, following the same 12-step workflow as `stage01_iqa.ipynb`.
- **Reuse:** `colab/common/experiment_manager.py` and `dataset_staging.py` (framework-agnostic, already dataset-agnostic, no changes needed regardless of Stage 3's final framework), `colab_config.py`'s existing `"VesselSegmentation"` entry in `PIPELINE_MODULES`. `training.Trainer`, `training.get_loss("bce_dice")`, and `training.build_metrics("segmentation")` are TensorFlow/Keras-specific and are reusable as-is only if Stage 3 is implemented in TensorFlow — see `SEGMENTATION_ARCHITECTURE.md` §6 for why this stage's framework is deliberately left open (named "Baseline U-Net," not "Standard U-Net," for the same reason).
- **Expected output:** a trained Baseline U-Net producing single-channel vessel probability maps, `(H, W, 1)`, values in `[0, 1]`, exported to `models/vessel_segmentation/best_model` (file extension depends on the final framework choice — see `SEGMENTATION_ARCHITECTURE.md` §6).

### Step 4 — Lesion Segmentation (Attention U-Net)
- **Why:** Same rationale as Step 3, for lesion (exudate/hemorrhage/microaneurysm) maps.
- **Depends on:** Step 3's trained model, since Lesion Segmentation's training input requires a vessel-mask channel generated by running the (now project-trained, not pretrained) Vessel Segmentation model over every IDRiD/segmentation image first.
- **New files:** `lesion_segmentation_dataset.py`, `lesion_segmentation_model.py` (Attention U-Net + train/infer functions).
- **Input:** processed RGB image + vessel probability map, concatenated — `(H, W, 4)`.
- **Output:** 4 lesion probability maps (Microaneurysm, Haemorrhage, Hard Exudate, Soft Exudate) — `(H, W, 4)`.
- **Colab notebook:** yes.

### Step 5 — Local Feature Extraction (Adaptive Multi-Kernel CNN)
- **Why:** Consumes the vessel/lesion maps and the preprocessed image to extract fine-grained local features, as distinct from the whole-image global branch.
- **Depends on:** Steps 3–4 outputs.
- **Input:** RGB image (3) + vessel map (1) + 4 lesion maps (4), concatenated — `(H, W, 8)`.
- **New files:** `local_feature_extractor.py` (multi-kernel/multi-branch CNN block, adaptively weighted).
- **Expected output:** a callable Keras layer/sub-model producing a local feature tensor, unit-tested in isolation (shape/sanity checks) before wiring into fusion.

### Step 6 — Global Feature Extraction (Dual-Scale Swin Transformer)
- **Why:** The existing `swin_transformer.py` already provides every low-level building block needed for this — the gap is topology (single-scale block used for refinement) vs. target (a genuine dual-scale backbone).
- **Input:** the processed RGB image directly (parallel branch, not sequential with Local Feature Extraction). Any resizing this stage needs is internal to it — Stage 02 does not resize, and no fixed resolution is documented here; the final input resolution is configurable and will be selected during implementation based on memory and model performance.
- **Files to modify:** `swin_transformer.py` — extend with a new `create_dual_scale_swin_model()` builder, without deleting or altering `create_swin_tiny_model()` or `create_hybrid_model()`, which the current baseline still depends on.

### Step 7 — Feature Fusion (Adaptive Cross-Attention)
- **Why:** Combines the Local (Step 5) and Global (Step 6) feature streams.
- **New files:** `feature_fusion.py` (cross-attention module).

### Step 8 — Ordinal Classification (CORN)
- **Why:** Replaces the current nominal softmax heads with a rank-consistent ordinal head appropriate for DR severity grading.
- **New files:** `ordinal_classifier.py` (CORN head + CORN loss + rank-to-class decoding), and an end-to-end training script wiring Preprocessing → Vessel/Lesion → Local → Global → Fusion → CORN together.
- **Colab notebook:** yes — the main end-to-end training notebook.

### Step 9 — Uncertainty Estimation (Monte Carlo Dropout) — integration only
- **Why:** Already implemented and correct in `bayesian_inference.py`; this step re-points it at the new model.

### Step 10 — Explainability (Grad-CAM++, SHAP, Attention Rollout)
- **Why:** Current `explainable_ai.py` only implements vanilla Grad-CAM against a single hard-coded layer name; generalize and add the two missing techniques.

### Step 11 — Evaluation
- **Why:** Consolidate the currently-duplicated evaluation logic into one module, and add QWK / real calibration for the new end-to-end model.

---

## 5. Open Questions Before Implementation Begins

1. **Segmentation datasets (Steps 3–4) — resolved.** DRIVE and CHASE_DB1 are now approved datasets, used to train Stage 3 within this project; IDRiD's segmentation subset trains Stage 4. See `SEGMENTATION_ARCHITECTURE.md`.
2. **`dr_gan.py`'s role going forward:** keep feeding the new ordinal classifier's training set the same way it currently feeds `dr_classifier.py`, or treat it as legacy/optional — undecided, out of scope for this refactor.
3. **Local Feature Extraction's input (Step 5) — resolved.** RGB image + vessel map + lesion maps, concatenated into an 8-channel tensor (§4 of `SEGMENTATION_ARCHITECTURE.md`).
4. **DRIVE test-set masks and CHASE_DB1's train/test split convention** — need verification once the raw files are actually placed on disk; see the migration plan associated with this refactor.

---

## 6. Next Step

Stage 1 is complete. Stage 2 is architecturally frozen and implementation-ready (no further design decision remains). Per the "one module at a time, wait for approval" rule, the next implementation target is **Stage 2's Colab wiring**, followed by **Stage 3 (Vessel Segmentation)** — the first newly-trainable module under the frozen architecture — with explicit approval requested before writing code for each.
