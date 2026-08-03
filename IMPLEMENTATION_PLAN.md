# Implementation Plan — Diabetic Retinopathy Detection

This document compares the current baseline repository against the target architecture defined in `PROJECT_CODE.md` and lays out a module-by-module roadmap to close the gap. No code has been changed to produce this document.

---

## 1. Current Architecture (as implemented today)

### 1.1 Configuration
All scripts load path/config values from environment variables via `python-dotenv` (`.env_sample` lists the expected keys: `BASE_PATH`, `MODEL_PATH`, `PROCESSED_TEST_DIR`, `RESULTS_DIR`, `PROCESSED_IMAGES_DIR`, `GAN_IMAGES_DIR`, `IMAGE_DIR`, `TEST_IMAGE_DIR`, `CSV_PATH`, `RETRAINED_MODEL_PATH`, `OUTPUT_DIR`, `MODEL_SAVE_PATH`, `RESULTS_PATH`, `IMAGE_SIZE`, `CHANNELS`, `NUM_CLASSES`, `BATCH_SIZE`, `EPOCHS`, `LATENT_DIM`). There is no config/package structure — every stage is a standalone top-level script, and shared model code is imported directly between scripts (e.g. `train_hybrid_model.py` imports `create_hybrid_model` from `swin_transformer.py`).

### 1.2 Data
The pipeline is built around the **APTOS 2019** CSV format (`id_code`, `diagnosis` columns, PNG fundus images, 5 severity classes 0–4). There is no EyeQ or EyePACS ingestion code.

### 1.3 Preprocessing (`pre_process_with_dataset_download.py`, `pre_process_test_and_train.py`)
Both scripts implement the same pipeline (the second variant adds a test-set branch):
- Green channel extraction (`image[:, :, 1]`)
- Ben Graham preprocessing (`cv2.addWeighted` with Gaussian blur — local contrast enhancement)
- CLAHE (`clipLimit=2.0`, `tileGridSize=(8,8)`)
- Median blur denoising (kernel 5)
- Resize to 224×224
- Normalize to [0,1], output single-channel array
- Stratified train/val split saved as CSVs, `tf.data.Dataset` generators built from disk
- No Gamma Correction, no dedicated Image Quality Assessment gate, no augmentation (augmentation only appears later, inside `train_hybrid_model.py`, not as a preprocessing-stage artifact)

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

## 2. Target Architecture (per `PROJECT_CODE.md`)

11-stage pipeline:

1. Image Quality Assessment — EfficientNetB0
2. Image Preprocessing — CLAHE, Gamma Correction, Green Channel Extraction, Ben Graham, Median Denoising, Resize, Augmentation
3. Vessel Segmentation — U-Net
4. Lesion Segmentation — Attention U-Net
5. Local Feature Extraction — Adaptive Multi-Kernel CNN
6. Global Feature Extraction — Dual-Scale Swin Transformer
7. Feature Fusion — Adaptive Cross-Attention
8. Ordinal Classification — CORN
9. Uncertainty Estimation — Monte Carlo Dropout
10. Explainability — Grad-CAM++, SHAP, Attention Rollout
11. Evaluation

Approved datasets: **EyeQ** (image quality), **APTOS 2019** (classification), **EyePACS** (additional fine-tuning). No other datasets permitted without explicit request.

---

## 3. Gap Analysis

| # | Target Module | Target Model | Current State | Status |
|---|---|---|---|---|
| 1 | Image Quality Assessment | EfficientNetB0 (quality classifier) | No code exists. `EfficientNetB0` is currently only used as a *diagnosis* classifier, not a quality gate. | **Missing** |
| 2 | Image Preprocessing | CLAHE, Gamma Correction, Green Channel, Ben Graham, Median Denoise, Resize, Augmentation | Green channel ✅, Ben Graham ✅, CLAHE ✅, Median denoise ✅, Resize ✅. Gamma Correction ❌. Augmentation exists only inside `train_hybrid_model.py` (in-graph Keras layers), not as a reusable preprocessing-stage step, and is absent from the other three training scripts. | **Partially implemented** |
| 3 | Vessel Segmentation | U-Net | No code exists. | **Missing** |
| 4 | Lesion Segmentation | Attention U-Net | No code exists. | **Missing** |
| 5 | Local Feature Extraction | Adaptive Multi-Kernel CNN | No dedicated "local" feature extractor; `EfficientNetB0`/`B2` are used as whole-image, single-scale feature extractors. | **Missing** |
| 6 | Global Feature Extraction | Dual-Scale Swin Transformer | A complete, hand-written single-scale Swin Transformer exists (`swin_transformer.py`: `PatchEmbed`, `WindowAttention`, `SwinTransformerBlock`, `BasicLayer`, `PatchMerging`, full `SwinTransformer` model), and `create_swin_tiny_model()` runs it standalone. However, the *hybrid* model only bolts on a single `SwinTransformerBlock` for feature refinement after a CNN backbone — no dual-scale windowing exists. Reusable foundation, wrong topology. | **Partially implemented** |
| 7 | Feature Fusion | Adaptive Cross-Attention | The current "fusion" is a linear sequence (CNN → one Swin block → GlobalAveragePooling2D → Dense), not an attention-based fusion of two independent feature streams. | **Missing** |
| 8 | Ordinal Classification | CORN | All classifiers (`efficientnet_model.py`, `swin_transformer.create_hybrid_model`, `dr_classifier.py`, `retrain_efficientnet.py`) use plain softmax + categorical/focal cross-entropy — nominal, not ordinal. No CORN head, no rank-consistent logits, no QWK (quadratic weighted kappa) metric anywhere despite it being the standard metric for ordinal DR grading. | **Missing** |
| 9 | Uncertainty Estimation | Monte Carlo Dropout | Fully implemented in `bayesian_inference.py`: MC sampling, mean/std, predictive entropy, uncertainty visualizations. Reliability diagram uses simulated labels (documented limitation, not a bug). Will need re-pointing at whatever model results from steps 5–8. | **Implemented** (needs integration once the classifier changes) |
| 10 | Explainability | Grad-CAM++, SHAP, Attention Rollout | Only vanilla Grad-CAM exists (`explainable_ai.py`), hard-coded to the `swin_refine` layer name from the current hybrid model. Grad-CAM++, SHAP, and Attention Rollout are all absent. | **Partially implemented** |
| 11 | Evaluation | — | Confusion matrix, classification report, accuracy/AUC exist per-script (`dr_classifier.py`, `test_hybrid_model.py`, `testing_efficientnet_model.py`) but are duplicated across files rather than a single evaluation module, and none compute ordinal-appropriate metrics (QWK) or real (non-simulated) calibration. | **Partially implemented** |

**Non-target component present in the repo:** `dr_gan.py` (conditional GAN for synthetic minority-class oversampling) is real and working but is not part of the 11-stage target pipeline. Per the "reuse existing components" rule, it should be kept and can still feed the ordinal classifier's training data (mirroring what `dr_classifier.py` already does), but it is not one of the roadmap's numbered modules.

**Dataset gap to flag now:** Vessel Segmentation (U-Net) and Lesion Segmentation (Attention U-Net) both require pixel-level mask ground truth (e.g. vessel masks, exudate/hemorrhage/microaneurysm masks). None of the three approved datasets (EyeQ, APTOS 2019, EyePACS) ship such masks — APTOS/EyePACS are image-level diagnosis labels only, and EyeQ is quality labels only. This is a conflict between the target pipeline and the "use only these datasets" rule that needs your decision before step 3/4 can start (options: request approval for an additional masked dataset such as DRIVE/IDRiD, or use weak/pseudo-labels derived from the approved data, or use pretrained segmentation weights without fine-tuning). Flagged here, not resolved.

---

## 4. Development Roadmap

Ordered to match the target pipeline's numbering, since each stage after preprocessing consumes the previous stage's output. A "Step 0" is added first for shared infrastructure every later step depends on.

### Step 0 — Shared Infrastructure & Config Extension
- **Why:** New datasets (EyeQ, EyePACS) and new model stages (segmentation, fusion, ordinal head) need new path/config variables before any of them can be built, following the repo's existing `.env`-driven convention.
- **Files to modify:** `.env_sample` (append new keys, don't remove existing ones).
- **New files:** none.
- **Expected output:** an extended `.env_sample` documenting the additional variables the new modules will need (e.g. EyeQ dataset paths, EyePACS dataset paths, vessel/lesion mask paths, per-stage model-save paths). No behavior change to existing scripts.

### Step 1 — Image Quality Assessment (EfficientNetB0)
- **Why:** First gate in the target pipeline; determines whether an image proceeds to preprocessing/diagnosis at all.
- **Reuses:** the `EfficientNetB0` transfer-learning pattern already proven in `efficientnet_model.py` (freeze first N layers, GAP, dense head) — same architecture family, different label space (binary/multi-class "quality" instead of DR severity) and different dataset (EyeQ).
- **Files to modify:** none required (kept isolated so the existing baseline keeps working untouched).
- **New files:** an EyeQ dataset loader/downloader script (mirrors `pre_process_with_dataset_download.py`'s structure) and an image-quality model script (mirrors `efficientnet_model.py`'s structure: `build`, `train`, `plot_training_history`).
- **Colab notebook:** yes — dedicated IQA training notebook, exporting best weights back into `MODEL_SAVE_PATH`.
- **Expected output:** a trained quality-classifier `.h5`/weights file and a callable quality-check function that can be inserted at the front of the pipeline.

### Step 2 — Preprocessing Extension (Gamma Correction + unified augmentation)
- **Why:** `PROJECT_CODE.md` explicitly calls out retaining CLAHE/Ben Graham/green-channel/denoise/resize as-is and only adding what's missing — Gamma Correction is the one listed transform that isn't implemented anywhere; augmentation currently exists in only one of four training scripts.
- **Files to modify:** `pre_process_with_dataset_download.py` and `pre_process_test_and_train.py` (add a gamma-correction step into the existing `preprocess_image` pipeline, after/alongside CLAHE — exact ordering to be confirmed before implementation, not decided here since no code is being written yet).
- **New files:** none strictly required; augmentation could be centralized into a small shared helper imported by all training scripts instead of being duplicated, if that's desired — a decision to make at implementation time, not now.
- **Colab notebook:** no (preprocessing is CPU/OpenCV based, no training).
- **Expected output:** preprocessed images that additionally include gamma correction, with existing CLAHE/Ben Graham/denoise behavior unchanged; a documented, single augmentation policy reused by all training entry points.

### Step 3 — Vessel Segmentation (U-Net)
- **Why:** Vessel maps are a prerequisite input for the Local/Global feature-extraction stages in the target architecture.
- **Blocked on:** the dataset conflict noted in Section 3 (no vessel-mask dataset in the approved list) — needs your decision before implementation starts.
- **Files to modify:** none.
- **New files:** a new `vessel_segmentation.py` (U-Net architecture + train/infer functions), following the same standalone-script convention as `swin_transformer.py`.
- **Colab notebook:** yes — dedicated segmentation training notebook.
- **Expected output:** a trained U-Net producing binary/probability vessel maps for a given fundus image, saved weights integrated back via `MODEL_SAVE_PATH`/`MODEL_PATH`-style env vars.

### Step 4 — Lesion Segmentation (Attention U-Net)
- **Why:** Same rationale as Step 3, for lesion (exudate/hemorrhage/microaneurysm) maps.
- **Blocked on:** same dataset conflict as Step 3.
- **Files to modify:** none.
- **New files:** `lesion_segmentation.py` (Attention U-Net + train/infer functions).
- **Colab notebook:** yes.
- **Expected output:** a trained Attention U-Net producing lesion probability maps, weights integrated the same way as Step 3.

### Step 5 — Local Feature Extraction (Adaptive Multi-Kernel CNN)
- **Why:** Consumes the vessel/lesion maps (and/or the preprocessed image) to extract fine-grained local features, as distinct from the whole-image global branch.
- **Depends on:** Steps 3–4 outputs (or, if those are deferred, this can initially run on the preprocessed image alone — a scoping decision for when this step starts).
- **Files to modify:** none.
- **New files:** `local_feature_extractor.py` (multi-kernel/multi-branch CNN block, adaptively weighted).
- **Colab notebook:** yes, if trained end-to-end as part of the full model rather than pretrained separately (to be decided at implementation time, likely trained jointly with fusion + classification in Step 8's notebook rather than standalone).
- **Expected output:** a callable Keras layer/sub-model producing a local feature tensor, unit-tested in isolation (shape/sanity checks) before wiring into fusion.

### Step 6 — Global Feature Extraction (Dual-Scale Swin Transformer)
- **Why:** The existing `swin_transformer.py` already provides every low-level building block (`PatchEmbed`, `WindowAttention`, `SwinTransformerBlock`, `BasicLayer`, `PatchMerging`) needed for this — the gap is topology (single-scale block used for refinement) vs. target (a genuine dual-scale backbone).
- **Files to modify:** `swin_transformer.py` — extend with a new `create_dual_scale_swin_model()` (or similarly named) builder that reuses the existing layer classes at two window/patch scales and merges them, **without deleting or altering** `create_swin_tiny_model()` or `create_hybrid_model()`, which the current baseline (`train_hybrid_model.py`, `test_hybrid_model.py`, `explainable_ai.py`, `bayesian_inference.py`) still depends on.
- **New files:** none required if the new builder lives in `swin_transformer.py` alongside the existing ones; a separate file is also an option, decided at implementation time.
- **Colab notebook:** likely folded into Step 8's end-to-end training notebook rather than trained standalone, since a backbone in isolation has no classification signal.
- **Expected output:** a callable dual-scale Swin feature extractor with verified output shapes, existing single-block hybrid path still functional and untouched.

### Step 7 — Feature Fusion (Adaptive Cross-Attention)
- **Why:** Combines the Local (Step 5) and Global (Step 6) feature streams — the one part of the pipeline with no existing analog at all.
- **Depends on:** Steps 5 and 6.
- **Files to modify:** none.
- **New files:** `feature_fusion.py` (cross-attention module: local features attend to global features and vice versa, adaptively weighted/gated combination).
- **Colab notebook:** folded into Step 8's notebook (fusion has no standalone objective).
- **Expected output:** a callable fusion module producing a single fused feature vector/map, shape-verified against both input streams.

### Step 8 — Ordinal Classification (CORN)
- **Why:** Replaces the current nominal softmax heads with a rank-consistent ordinal head appropriate for DR severity grading (0–4 is an ordered scale, which softmax cross-entropy ignores).
- **Depends on:** Step 7's fused features as input.
- **Files to modify:** none required for the new head itself; `train_hybrid_model.py`'s focal-loss pattern (class weighting under imbalance) is worth reusing conceptually for the CORN training loop rather than being replaced outright.
- **New files:** `ordinal_classifier.py` (CORN head + CORN loss + rank-to-class decoding), and a new end-to-end training script (e.g. `train_dr_pipeline.py`) that wires Preprocessing → (Vessel/Lesion) → Local → Global → Fusion → CORN together, mirroring how `train_hybrid_model.py` currently wires preprocessing output → `create_hybrid_model`.
- **Colab notebook:** yes — this is the main end-to-end training notebook for the new architecture.
- **Expected output:** a trained full-pipeline model, plus QWK (quadratic weighted kappa) added to the evaluation metrics since it's the standard ordinal-grading metric and is currently absent everywhere in the repo.

### Step 9 — Uncertainty Estimation (Monte Carlo Dropout) — integration only
- **Why:** Already implemented and correct in `bayesian_inference.py`; this step is about re-pointing it at the new model, not rebuilding it.
- **Files to modify:** `bayesian_inference.py` — update the `custom_objects` dict and `last_conv_layer`/feature-source references to match the new architecture's layer names (the same pattern already used to support `swin_refine` + the custom `Cast` layer today).
- **New files:** none.
- **Expected output:** MC-Dropout uncertainty outputs (mean prediction, std, predictive entropy, reliability diagram) working against the new ordinal model; the "simulated ground truth" caveat in the reliability diagram should be revisited once real held-out labels are available from Step 8's training run.

### Step 10 — Explainability (Grad-CAM++, SHAP, Attention Rollout)
- **Why:** Current `explainable_ai.py` only implements vanilla Grad-CAM against a single hard-coded layer name; target wants three complementary techniques (Grad-CAM++ for better multi-instance localization, SHAP for feature-attribution outside the conv/attention structure, Attention Rollout specifically for the Swin branch).
- **Files to modify:** `explainable_ai.py` — generalize `make_gradcam_heatmap` into Grad-CAM++ and parameterize the target layer instead of hard-coding `'swin_refine'`, so it keeps working for whichever model is passed in.
- **New files:** an Attention Rollout module (operates on the dual-scale Swin branch's attention maps from Step 6) and a SHAP-based explainer module (likely `shap.DeepExplainer` or `GradientExplainer` over the fused feature/classification path).
- **Colab notebook:** no (inference-time visualization, not training).
- **Expected output:** three explanation modalities producible for any given test image and the new ordinal model, saved alongside the existing Grad-CAM output convention (`RESULTS_DIR`).

### Step 11 — Evaluation
- **Why:** Consolidate the currently-duplicated evaluation logic (confusion matrix / classification report scattered across `dr_classifier.py`, `test_hybrid_model.py`, `testing_efficientnet_model.py`) into one module, and add the ordinal-appropriate metrics that are missing everywhere (QWK, per-class sensitivity/specificity relevant to clinical screening).
- **Files to modify:** none required to existing files (they can keep working as-is per "preserve existing functionality").
- **New files:** `evaluate_pipeline.py` — a single evaluation entry point for the new end-to-end model: confusion matrix, classification report, QWK, calibration (using real labels this time), and comparison against the existing EfficientNet/hybrid baselines' saved metrics (reusing `test_hybrid_model.py`'s existing `create_comparison_with_baseline()` pattern of comparing prediction CSVs).
- **Colab notebook:** no (can run locally or in Colab against exported test predictions).
- **Expected output:** a single evaluation report (metrics CSV + plots) for the new pipeline, directly comparable to the existing baseline's `classification_report.csv`/`model_comparison.csv` outputs.

---

## 5. Open Questions Before Implementation Begins

1. **Segmentation datasets (Steps 3–4):** none of the three approved datasets include vessel/lesion masks. Need a decision: approve an additional masked dataset, use pseudo-labels/pretrained weights, or descope segmentation to weak supervision.
2. **Gamma Correction placement (Step 2):** before or after CLAHE, and on which channel (green channel vs. full image) — needs a decision at implementation time, not assumed here.
3. **Augmentation centralization (Step 2):** whether to keep augmentation duplicated per training script (current pattern) or factor it into one shared helper used by all training entry points.
4. **`dr_gan.py`'s role going forward:** keep feeding the new ordinal classifier's training set the same way it currently feeds `dr_classifier.py`, or treat it as legacy/optional.
5. **Local Feature Extraction's input (Step 5):** raw preprocessed image only, or the vessel/lesion maps from Steps 3–4 — affects whether Step 5 can start before Steps 3–4 are unblocked.

---

## 6. Next Step

Per the development workflow in `PROJECT_CODE.md`, implementation should proceed one module at a time in the order above, starting with **Step 0 (infrastructure)** and **Step 1 (Image Quality Assessment)**, with explicit approval requested before writing code for each step.
