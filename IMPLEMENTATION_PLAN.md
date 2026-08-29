# Implementation Plan — Diabetic Retinopathy Detection

This document compares the current baseline repository against the target architecture defined in `PROJECT_CODE.md` and lays out a module-by-module roadmap to close the gap. Sections describing the pre-refactor baseline (§1) are historical and unaffected by the architecture freeze; sections describing the target architecture (§2 onward) reflect the frozen design in `PROJECT_CODE.md` / `SEGMENTATION_ARCHITECTURE.md` / `PROJECT_STRUCTURE.md`.

---

## 0. Implementation Rules

Canonical copy lives in `PROJECT_CODE.md` under "Implementation Rules" -- restated here since this is the document each roadmap step gets checked against:

This is a production/research project, not a demonstration project.

1. Do not implement placeholder logic, simulated outputs, fake metrics, or dummy pipelines.
2. Unit tests may use synthetic or temporary data only, to verify correctness of individual functions.
3. All actual project functionality must operate on the real datasets: EyeQ, APTOS2019, and IDRiD. EyePACS was used only once, historically, to reconstruct EyeQ, and is not required to reproduce or run this repository (see `PROJECT_CODE.md`'s Datasets section). Vessel Segmentation (Stage 03) uses a vendored pretrained checkpoint and requires no project dataset of its own.
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

12-stage pipeline (11 base stages plus RACAF, the one approved downstream research innovation —
see `PROJECT_CODE.md`'s "Approved Research Innovation" section and `RACAF_ARCHITECTURE.md`):

1. Image Quality Assessment — EfficientNetB0
2. Image Preprocessing — Gamma Correction, CLAHE (RGB in, RGB out; no green-channel extraction, Ben Graham, median denoise, histogram equalization, resizing, or augmentation)
3. Vessel Segmentation — Pretrained LWNet, **inference only, not trained within this project**
4. Lesion Segmentation — Attention U-Net, trained on IDRiD — **finalized, Experiment 2C (Weighted-Pooled Dice), frozen**
5. Local Feature Extraction — Adaptive Multi-Kernel CNN
6. Global Feature Extraction — Dual-Scale Swin Transformer
7. Feature Fusion — Adaptive Cross-Attention — **implemented, not trained** (Global queries Local, $d_{model}=256$, output $E=(B,256)$), see `feature_fusion.py` and `PROJECT_STRUCTURE.md` §7
8. Reliability-Aware Cross-Attention Fusion (RACAF) — the one approved research innovation; wraps Stage 7's output, does not redefine it; **implemented, not trained** (`racaf.py`), see `RACAF_ARCHITECTURE.md`
9. Ordinal Classification — CORN — **implemented, not trained** (`Dense(256->4)` on RACAF's `F`, standard CORN conditional-subset loss, 1,028 trainable parameters), see `corn.py` and `CORN_ARCHITECTURE.md`
10. Uncertainty Estimation — Monte Carlo Dropout
11. Explainability — Grad-CAM++, SHAP, Attention Rollout
12. Evaluation

Approved datasets: **EyeQ** (image quality, Stage 01 only), **APTOS 2019** (classification), and **IDRiD** (lesion segmentation and grading). EyePACS itself is not part of the implemented training or inference pipeline and is not required to reproduce it. Vessel Segmentation (Stage 03) uses a vendored pretrained checkpoint (LWNet) and needs no dataset of its own — DRIVE and CHASE_DB1, approved under an earlier superseded design, are no longer project datasets (see `SEGMENTATION_ARCHITECTURE.md` Appendix A.1). No other datasets permitted without explicit request.

---

## 3. Gap Analysis

| # | Target Module | Target Model | Current State | Status |
|---|---|---|---|---|
| 1 | Image Quality Assessment | EfficientNetB0 (quality classifier) | Implemented, trained, verified — see `PROJECT_STRUCTURE.md`'s Stage 1 results. | **Completed** |
| 2 | Image Preprocessing | Gamma Correction, CLAHE only (RGB in/out, deterministic, generated once) | `image_preprocessing.py` implements exactly this. No green-channel extraction, Ben Graham, median denoise, or resize — all explicitly excluded from Stage 02 per the frozen architecture. | **Frozen / implementation-ready** |
| 3 | Vessel Segmentation | Pretrained LWNet, inference only | No code exists yet. Design finalized: Stage 03 integrates the externally-sourced, MIT-licensed `lwnet` checkpoint for inference only, not trained within this project (see `SEGMENTATION_ARCHITECTURE.md` §1.2/§2, and its design-history appendix for why an intermediate design trained a Baseline U-Net on DRIVE + CHASE_DB1 instead, and why that was reversed). | **Missing — design finalized, ready to implement** |
| 4 | Lesion Segmentation | Attention U-Net | Implemented and trained in Colab. Final experiment **2C (Weighted-Pooled Dice)**: Mean Dice 0.1314 / Mean IoU 0.0766 on the official 27-image IDRiD test set (per-class: MA 0.0165/0.0083, HE 0.1273/0.0680, EX 0.3574/0.2176, SE 0.0244/0.0123). See `SEGMENTATION_ARCHITECTURE.md` §3 and `RACAF_ARCHITECTURE.md` §1. No Experiment 2D is planned — Stage 4 is closed. | **Completed — FROZEN.** No further training, loss changes, or architecture changes to this stage. |
| 5 | Local Feature Extraction | Adaptive Multi-Kernel CNN | No dedicated "local" feature extractor exists. Input contract finalized: RGB image + vessel map + 4 lesion maps, concatenated into an 8-channel tensor (`SEGMENTATION_ARCHITECTURE.md` §4). | **Missing** |
| 6 | Global Feature Extraction | Dual-Scale Swin Transformer | Implemented: `swin_transformer.py`'s `create_dual_scale_swin_model()` — two parallel Swin branches (patch 4/8, `depths=[2,2,6,2]`/`[2,2,6]`) reusing the existing `PatchEmbed`/`BasicLayer`/`PatchMerging` classes, fused by concatenation only, output `(B,64,1152)`. `create_swin_tiny_model()`/`create_hybrid_model()` untouched. Not trained (no standalone objective — see `PROJECT_STRUCTURE.md` §6). | **Implemented, not trained** |
| 7 | Feature Fusion | Adaptive Cross-Attention | **Implemented, not trained.** `feature_fusion.py`'s `build_adaptive_cross_attention()` — one-way cross-attention, Global queries Local (`Q`=Global's 64 tokens, `K,V`=Local's 1024 tokens), `d_model=256`, 8 heads, pre-LN block + FFN (`256->1024->256`, GELU, dropout 0.1), factorized 2D positional embeddings, global-average-pooled to `E=(B,256)`. Verified against the real, already-implemented Stage 05/06 models (not just representative tensors) — see `PROJECT_STRUCTURE.md` §7 for the full specification. `feature_fusion.py` fully replaces the old baseline "fusion" (CNN → one Swin block → GlobalAveragePooling2D → Dense) reference — that baseline is not this design and was never a real implementation of this stage. | **Implemented, not trained** |
| 8 | Reliability-Aware Cross-Attention Fusion (RACAF) | RACAF — TTA-based reliability gate wrapping Stage 7's output | **Implemented, not trained.** `racaf.py`: `tta_views()` (frozen Stage 04, 4 deterministic transforms, called directly, never via `predict_lesion_mask()`), `compute_reliability()` (population-variance disagreement, per-class `kappa`, burden-weighted scalar `r` — fully deterministic, no labels), `get_or_compute_reliability()` (new disk cache, stores only `kappa`/`r`), `build_racaf_fusion()` (the only trainable piece: `gate=σ(w_g·r+b_g)`, `Ĝ=W_r·GAP(G)+b_r`, `F=gate·E+(1-gate)·Ĝ`, exactly 295,170 trainable params, measured). Verified against the real Stage 05/06/07 models end-to-end. | **Implemented, not trained** |
| 9 | Ordinal Classification | CORN | **Implemented, not trained.** `corn.py`: a single `Dense(256->4)` layer on RACAF's `F`, standard CORN conditional-subset loss (`corn_loss`), sigmoid+cumulative-product decoding (`decode_logits`), 1,028 trainable parameters (measured). `pipeline.classification.ClassificationStage` implementation (`CORNStage`). QWK — already implemented and reusable in `evaluation/metrics.py` / `training/metrics.py` — is the intended ordinal evaluation metric once trained. See `CORN_ARCHITECTURE.md`. | **Implemented, not trained** |
| 10 | Uncertainty Estimation | Monte Carlo Dropout | Fully implemented in `bayesian_inference.py`: MC sampling, mean/std, predictive entropy, uncertainty visualizations. Reliability diagram uses simulated labels (documented limitation, not a bug). Will need re-pointing at whatever model results from steps 5–9. | **Implemented** (needs integration once the classifier changes) |
| 11 | Explainability | Grad-CAM++, SHAP, Attention Rollout | Only vanilla Grad-CAM exists (`explainable_ai.py`), hard-coded to the `swin_refine` layer name from the current hybrid model. Grad-CAM++, SHAP, and Attention Rollout are all absent. | **Partially implemented** |
| 12 | Evaluation | — | Confusion matrix, classification report, accuracy/AUC exist per-script but are duplicated across files rather than a single evaluation module, and none compute ordinal-appropriate metrics (QWK) or real (non-simulated) calibration against the target architecture's models. | **Partially implemented** |

**Non-target component present in the repo:** `dr_gan.py` (conditional GAN for synthetic minority-class oversampling) is real and working but is not part of the 12-stage target pipeline. Per the "reuse existing components" rule, it should be kept and can still feed the ordinal classifier's training data, but it is not one of the roadmap's numbered modules.

**Dataset gap — resolved differently for each stage.** Lesion Segmentation (Stage 4) requires pixel-level mask ground truth and is trained within this project on IDRiD's segmentation subset. Vessel Segmentation (Stage 3) also needs pixel-level vessel ground truth, but rather than sourcing a dataset and training within this project, it integrates a pretrained external checkpoint (LWNet, trained by its own authors on DRIVE) for inference only — so Stage 3 needs no project dataset of its own. An intermediate design added DRIVE and CHASE_DB1 as project datasets to train a "Baseline U-Net" within this project instead; that design was itself superseded by the current pretrained-LWNet design — see `SEGMENTATION_ARCHITECTURE.md`'s design-history appendix for the full chronology.

---

## 4. Development Roadmap

Ordered to match the target pipeline's numbering, since each stage after preprocessing consumes the previous stage's output. A "Step 0" is added first for shared infrastructure every later step depends on.

### Step 0 — Shared Infrastructure & Config Extension
- **Why:** New datasets (IDRiD) and new model stages (segmentation, fusion, ordinal head) need new path/config variables before any of them can be built, following the repo's existing `.env`-driven convention.
- **Files to modify:** `.env_sample` (append new keys, don't remove existing ones); `config.py` (add `VESSEL_SEG_MODEL_DIR`, mirroring `IQA_MODEL_DIR`, pointed at the vendored LWNet checkpoint rather than a training-run output — no `VESSEL_SEG_RESULTS_DIR` needed since Stage 03 has no evaluation run of its own).
- **Expected output:** extended config documenting the additional variables Stage 3 needs. No behavior change to existing scripts.

### Step 1 — Image Quality Assessment (EfficientNetB0)
**Status: Completed, verified, trained, exported.** See `PROJECT_STRUCTURE.md` for full detail. No further action.

### Step 2 — Preprocessing (Gamma Correction + CLAHE)
- **Why:** `PROJECT_CODE.md` specifies Stage 02 as exactly Gamma Correction + CLAHE on RGB — no other transform.
- **Status:** `image_preprocessing.py` already implements this. Frozen and implementation-ready; no further architectural decision remains before Stage 02 is wired into `colab/notebooks/stage02_preprocessing.ipynb` and run once, per-dataset, per `PROJECT_CODE.md`'s Dataset Policy.

### Step 3 — Vessel Segmentation (pretrained LWNet integration)
- **Why:** Vessel maps are a prerequisite input for Lesion Segmentation (Stage 4) and Local Feature Extraction (Stage 5) in the target architecture. Unlike every other trainable stage, this one integrates an already-trained, externally-sourced model rather than training a new one.
- **Datasets:** None. LWNet's vendored checkpoint was trained by its original authors on DRIVE; this project stages, trains, or evaluates nothing for this stage. Stage 03 runs inference directly on Stage 02's RGB output.
- **New files:** `vessel_segmentation_model.py` (loads the vendored checkpoint, wraps LWNet's `wnet` architecture), `vessel_segmentation_inference.py` (FOV crop/resize/normalize + `predict`/`predict_batch`, adapted from the upstream repo's `predict_one_image.py`) — no dataset loader, training, or evaluation script, since none apply to a pretrained-only stage.
- **Colab notebook:** yes — `colab/notebooks/stage03_vessel_segmentation.ipynb`, but scoped to checkpoint integration and inference verification, not the Setup → Training → Evaluation → Export workflow every trainable stage otherwise follows.
- **Reuse:** `colab/common/setup.py`, `verify_environment.py` (environment checks only — no dataset staging needed, since Stage 03 has no dataset). `colab_config.py`'s existing `"VesselSegmentation"` entry in `PIPELINE_MODULES` still applies (to hold the vendored checkpoint under `exported_models/VesselSegmentation/`), even though no training run ever populates `experiments/VesselSegmentation/`. `training.Trainer` and `training.build_metrics("segmentation")` (TensorFlow/Keras-specific) do not apply to this stage at all — LWNet is PyTorch, and this stage is inference-only regardless of framework.
- **Expected output:** the vendored LWNet checkpoint (`models/vessel_segmentation/best_model.pth` + `config.cfg`) producing single-channel vessel probability maps, `(H, W, 1)`, values in `[0, 1]`, at Stage 02's native resolution — see `SEGMENTATION_ARCHITECTURE.md` §2/§6.

### Step 4 — Lesion Segmentation (Attention U-Net)
- **Status: Completed — FROZEN.** Final experiment **2C (Weighted-Pooled Dice)**: Mean Dice
  0.1314 / Mean IoU 0.0766 on the official 27-image IDRiD test set. No Experiment 2D and no
  further architecture, loss, or training changes to this stage — see `RACAF_ARCHITECTURE.md`
  §1 for the full per-class results, recorded there as evaluation history only. Stage 4 is
  frozen for every downstream stage, including RACAF (Step 7.5 below).
- **Why:** Same rationale as Step 3, for lesion (exudate/hemorrhage/microaneurysm) maps.
- **Depends on:** Step 3's vendored checkpoint, since Lesion Segmentation's training input requires a vessel-mask channel generated by running the pretrained LWNet model over every IDRiD/segmentation image first — a checkpoint-availability dependency, not a training-order one (Step 3 never trains).
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
- **Status: Implemented, unit-tested. Not trained, not frozen.** Two parallel Swin branches
  (Branch A: patch 4, full 4-stage `depths=[2,2,6,2]`; Branch B: patch 8, 3-stage `depths=[2,2,6]`)
  built from `PatchEmbed`/`BasicLayer`/`PatchMerging`, fused by channel-wise concatenation only —
  no projection, no cross-attention, no pooling, no classification head. Output `(B,64,1152)`. See
  `PROJECT_STRUCTURE.md` §6 for the full specification and the Stage 06 design-resolution record
  for the complete literature/engineering traceability.
- **Why:** The existing `swin_transformer.py` already provided every low-level building block needed for this — the gap was topology (single-scale block used for refinement) vs. target (a genuine dual-scale backbone).
- **Input:** the processed RGB image directly (parallel branch, not sequential with Local Feature Extraction, and with no dependency on Stage 3/4/5). Resized to 256x256 inside Stage 06's own pipeline — Stage 02 itself is not resized.
- **Files modified:** `swin_transformer.py` — extended with `create_dual_scale_swin_model()` and `GlobalFeatureExtractionStage`; the function bodies of `create_swin_tiny_model()`/`create_hybrid_model()` are unchanged (verified via `git diff` hunk boundaries — no edit touches either function). Three pre-existing defects in the shared `SwinTransformerBlock`/`PatchMerging` classes (a dtype mismatch, a `tf.Variable`-inside-graph-tracing error, and a symbolic-tensor-as-bool assertion — all unrelated to Stage 06's architecture, all required for any `shift_size>0` Swin block to build under the currently-installed Keras version) were fixed as the minimal changes needed for Stage 06 to build at all — independently re-verified (not just re-asserted) against the pre-session code, with the fixes confirmed to produce numerically identical values to what the original code would have computed had it not crashed. A fourth, unrelated pre-existing defect (`SwinTransformer.__init__`'s `self.layers = []` shadowing Keras 3's reserved `Model.layers`) was deliberately left unfixed — `git grep` confirms `create_swin_tiny_model()`/`SwinTransformer` have no active caller anywhere in this project, so fixing it was judged unnecessary scope expansion; pinned by a regression test rather than left undocumented. See `PROJECT_STRUCTURE.md` §6 for the full account.
- **New files:** `global_feature_extraction_dataset.py` (reuses `local_feature_extraction_dataset.py`'s Stage-02-application helpers, not duplicated).
- **Checkpoint format:** weights-only (`.weights.h5`), matching the shared `training/callbacks.py`/`training/trainer.py` framework's own default (`save_weights_only=True`) — Stage 4 is the one that opts into full `.keras`, not the project baseline. See `PROJECT_STRUCTURE.md` §6.

### Step 5/6 infrastructure update — authoritative split + Drive path resolution

Both stages' dataset loaders previously computed the same interim, non-stratified train/val split
independently (Stage 5 owned it, Stage 6 re-exported it) and had no way to resolve APTOS2019/IDRiD
paths from Google Drive in a real Colab session (`colab/common/setup.py`'s
`configure_environment_variables()` wired only `EYEQ_RAW_DIR`). Fixed as infrastructure, not an
architecture change (`local_feature_extraction_model.py`/`swin_transformer.py`'s model code and
tensor contracts are unchanged):
- **New file:** `downstream_split.py` — the ONE authoritative, stratified (by `diagnosis`), seeded
  (80/20, seed 42 — the prior interim ratio/seed, now promoted and stratified) train/val split,
  persisted to the committed manifest `dataset_splits/aptos2019_train_val_split.csv`.
  `local_feature_extraction_dataset.split_train_val_ids()` now delegates to it; `global_feature_extraction_dataset.py`'s
  re-export is unchanged and picks this up transparently. See `PROJECT_STRUCTURE.md`'s
  "Authoritative Downstream Classification Split" section.
- **`colab/common/drive_paths.py`/`colab_config.py`/`setup.py`:** extended to resolve and wire
  APTOS2019 raw/processed and IDRiD's three subsets (`grading`/`localization`/`segmentation`,
  each with its own raw/processed) individually — previously unresolvable from Drive in a fresh
  Colab session. `colab/common/verify_dataset.py` gained `verify_idrid_dataset_dir()`, an explicit,
  fail-clearly existence check (not a silent assumption) for this layout.
- **`local_feature_extraction_dataset.py`/`global_feature_extraction_dataset.py`:** new
  `_resolve_processed_rgb()` helper (Stage 05, reused by Stage 06) checks for an existing Stage 02
  processed-output file per image before falling back to live in-memory preprocessing — so a Drive
  environment with APTOS2019's Stage 02 output already batch-generated is used as-is, never
  silently regenerated. Both `_build_sample` functions gained an optional, defaulted `processed_dir`
  parameter; all existing callers (including the Stage 05/06 Colab notebooks' direct calls) are
  unaffected.
- **IDRiD's Disease Grading test split** (103 official-labeled images, verified this session to
  have no image overlap with Stage 4's own training images) remains an **optional external
  evaluation candidate for CORN only — PENDING USER APPROVAL**, not adopted into training, model
  selection, or documented as final anywhere in this repository.

### Step 7 — Feature Fusion (Adaptive Cross-Attention)
- **Status: Implemented, unit-tested. Not trained. Not frozen.** One-way cross-attention, Global
  queries Local (`Q`=Global's 64 tokens, `K,V`=Local's 1024 tokens), `d_model=256` (8 heads x 32
  dims/head), pre-LN block with residual + FFN (`256->1024->256`, GELU, dropout 0.1), factorized
  2D positional embeddings on both branches, global-average-pooled to a single fused embedding
  `E=(B,256)`. See `PROJECT_STRUCTURE.md` §7 for the complete specification, literature basis
  (Perceiver, Jaegle et al. ICML 2021; CrossViT, Chen et al. ICCV 2021), and traceability table.
- **Why:** Combines the Local (Step 5) and Global (Step 6) feature streams.
- **Why this direction:** RACAF's formulation (`RACAF_ARCHITECTURE.md` §5) consumes exactly one
  fused vector `E`; keeping the query side small (Global's 64 tokens, not Local's 1024) keeps
  every post-attention operation an order of magnitude cheaper than the reverse direction, for
  identical attention-score cost. Bidirectional attention was considered and rejected — it would
  produce a second stream RACAF's approved formula has no place to consume.
- **"Adaptive" clarified:** content-dependent attention weighting via learned Q/K/V projections
  and softmax — explicitly **not** reliability-, uncertainty-, TTA-, or confidence-aware. Those
  remain exclusive to RACAF (Step 7.5); Step 7 never reads Stage 4's output — verified structurally
  by regression tests, not just asserted (`tests/test_feature_fusion.py`'s `RACAFBoundaryTests`).
- **Global's real input shape:** Stage 06's real, implemented output is already flattened
  `(64, 1152)`, not the conceptually-described spatial `(8,8,1152)` — `create_dual_scale_swin_model()`
  performs that flatten internally. `feature_fusion.py`'s Global input matches Stage 06's real,
  verified output shape directly; Local's input remains spatial `(32,32,256)`, matching Stage 05's
  real output, and is flattened inside this module. Verified end-to-end against the real Stage
  05/06 models, not just representative tensors.
- **Serialization:** full `.keras` (`model.save()`/`load_model()`), not weights-only — every layer
  used (`MultiHeadAttention`, `Dense`, `LayerNormalization`, `Dropout`, `Add`,
  `GlobalAveragePooling1D`) is a built-in Keras layer with existing `get_config()` support; the one
  custom layer this module introduces (`Factorized2DPositionalEmbedding`) implements `get_config()`
  and is registered via `@register_keras_serializable()`. This deliberately does not repeat Stage
  06's weights-only fallback — no pre-existing, non-serializable classes are reused here.
- **New files:** `feature_fusion.py` (`build_adaptive_cross_attention()`,
  `Factorized2DPositionalEmbedding`, `AdaptiveCrossAttentionStage`).
- **Checkpoint format:** full `.keras`, per the serialization note above. See `PROJECT_STRUCTURE.md`
  §7.

### Step 7.5 — RACAF (Reliability-Aware Cross-Attention Fusion)
- **Status: Implemented, unit-tested (73 tests). Not trained. Not frozen.** `racaf.py` implements
  exactly `RACAF_ARCHITECTURE.md`'s Sec 4/5/6/7 formulation, in four independently testable
  pieces: frozen-Stage-04 TTA (`tta_views`), deterministic reliability computation
  (`compute_reliability` — population variance, per-class `kappa`, burden-weighted scalar `r`, no
  labels), a new per-image disk cache (`get_or_compute_reliability`, stores only `kappa`/`r`), and
  the trainable fusion model (`build_racaf_fusion` — gate + Global readout only, 295,170 trainable
  parameters, measured exactly). `RACAFStage` follows `AdaptiveCrossAttentionStage`'s identical
  `TrainableStage`/`InferenceStage` pattern.
- **Why:** The single approved downstream research innovation (`PROJECT_CODE.md`'s "Approved
  Research Innovation" section). Wraps Step 7's output with a per-image reliability gate derived
  from Stage 4's frozen, test-time-augmented output — never from Stage 4's recorded test-set
  Dice/IoU, and never requiring Stage 4 to be retrained or architecturally modified.
- **Depends on:** Steps 5, 6, and 7's output contracts (`L`'s shape, `G`'s shape, and Cross-
  Attention's own output dimensionality) all being finalized first — RACAF's Global-readout
  projection is defined relative to those shapes and cannot be built before they exist. **All
  three were satisfied** (`L=(B,1024,256)`, `G=(B,64,1152)`, `E=(B,256)`), which is what made this
  implementation possible; verified against the real Stage 05/06/07 models end-to-end.
- **Full specification:** `RACAF_ARCHITECTURE.md` — the authoritative design document, followed
  exactly; no equation, parameter, or tensor contract was changed during implementation.
- **New files:** `racaf.py` (`tta_views`, `compute_reliability`, `get_or_compute_reliability`,
  `build_racaf_fusion`, `RACAFStage`).

*(Note: this roadmap's Step numbers below no longer align 1:1 with the Target Architecture's
Stage numbers in §2, since RACAF was inserted as "Step 7.5" rather than triggering a renumbering
of every already-referenced roadmap step — Step 8 below corresponds to Stage 9 in §2, Step 9 to
Stage 10, Step 10 to Stage 11, and Step 11 to Stage 12.)*

### Step 8 — Ordinal Classification (CORN)
- **Status: Implemented, unit-tested (42 tests). Not trained. Not frozen.** `corn.py` implements
  exactly `CORN_ARCHITECTURE.md`'s formulation: `build_corn_model()` (`Dense(256->4)`, no hidden
  layer, no output activation, uncompiled, 1,028 trainable parameters measured exactly),
  `corn_loss()` (standard conditional-subset ordinal loss — no focal loss, class weighting, label
  smoothing, or Dice loss added), `decode_logits()` (sigmoid -> cumulative product -> thresholded
  grade -> per-class probability reconstruction), and `CORNStage`
  (`pipeline.classification.ClassificationStage`, `train()`/`evaluate()` raise
  `NotImplementedError`, mirroring `RACAFStage`'s identical pattern).
- **Why:** Replaces the current nominal softmax heads with a rank-consistent ordinal head appropriate for DR severity grading.
- **Input:** RACAF's `F=(B,256)` only — verified live against the real, current `racaf.py`. No
  Stage 4 mask, Stage 5/6 feature, Stage 7 raw `E`, or RACAF reliability vector `r` is consumed.
- **Dataset/split:** APTOS2019 only, via the already-committed authoritative manifest
  (`dataset_splits/aptos2019_train_val_split.csv`, `downstream_split.get_authoritative_split()`) —
  no second split created; the identical partition Stage 5/6 already use.
- **New files:** `corn.py` (CORN head + CORN loss + rank-to-class decoding). The end-to-end
  training script wiring Stage 05 -> Stage 06 -> Stage 07 -> RACAF -> CORN together does not exist
  yet — not implemented in this step, per the explicit no-training instruction for this stage.
- **Colab notebook:** infrastructure implemented, `RUN_TRAINING = False`. `colab/notebooks/stage08_corn_classifier.ipynb`
  is repurposed as the joint Stage 05-08+RACAF training notebook (no second, competing notebook) —
  see `JOINT_TRAINING_ARCHITECTURE.md` §27 and Step 8.5 below.

### Step 8.5 — Joint Training Pipeline (Stage 05-08 + RACAF)
- **Status: Implemented, unit-tested (43 tests, `tests/test_joint_training.py`, plus 11 more in
  `tests/test_corn.py` for the CORN-aware QWK metric below). NOT trained — no `model.fit()` call
  on real data, no epoch, no checkpoint exists anywhere in this repository or on Drive.**
  Full design: `JOINT_TRAINING_ARCHITECTURE.md`.
- **`corn.CORNQuadraticWeightedKappa`** — a post-implementation audit found that
  `compile_joint_model()` originally compiled with no metric at all, so `monitor="val_QWK"`
  (below) had no `"val_QWK"` key in Keras's `logs` dict to read — checkpoint
  selection/early-stopping/LR-reduction would each have silently no-op'd on the first real run.
  Fixed with a Keras `Metric` (`corn.py`) that decodes CORN's raw logits via EXACTLY
  `decode_logits`'s own sigmoid → cumulative-product → threshold-count rule (not an argmax — the
  project's existing generic `training.metrics.QuadraticWeightedKappa` cannot be attached to
  CORN's output directly, since argmaxing 4 conditional-task logits both computes the wrong grade
  and can never reach class index 4) and delegates confusion-matrix/kappa computation to
  `QuadraticWeightedKappa` unmodified. `compile_joint_model()` now compiles with
  `metrics=[corn.CORNQuadraticWeightedKappa()]` (named `"QWK"`) alongside the unchanged
  `loss=joint_corn_loss` — QWK is a metric only, never a second loss. See
  `JOINT_TRAINING_ARCHITECTURE.md` §23.
- **New files:**
  - `joint_training_dataset.py` — the joint data/cache pipeline. Builds `stage5_input`
    `(512,512,8)`, `stage6_input` `(256,256,3)`, `reliability` (scalar `r`), and `grade` per
    sample, using the SAME authoritative split (`downstream_split.get_authoritative_split()`) and
    the SAME Stage 3/4 canonical cache path convention (`local_feature_extraction_dataset._cache_path`)
    Stage 5's own loader already established. Resolves the previously-deferred Step 4 redundancy
    (`JOINT_TRAINING_ARCHITECTURE.md` §11.1): `_get_or_compute_joint_frozen_outputs()` calls
    `racaf.prepare_stage4_input()` + `racaf.tta_views()` exactly once per uncached image, deriving
    BOTH Stage 5's lesion-cache entry and RACAF's `kappa`/`r` from that one call — verified via a
    mocked call-count test and a direct numerical equality test against RACAF's own identity view.
    Synchronized spatial augmentation (Stage 6's input is a resize of Stage 5's OWN, possibly
    augmented, RGB channels — never an independent draw); RGB-only intensity jitter; RACAF's `r`
    always computed from the canonical, unaugmented Stage 4 output.
  - `joint_training_model.py` — `build_joint_model()` composes Stage 5/6/7/RACAF/CORN's own,
    unmodified `build_*()` functions into one functional `keras.Model`
    (`[stage5_input, stage6_input, reliability] -> logits`, measured 43,296,810 total / 393
    trainable variables). `compile_joint_model()` sets `corn.corn_loss` as the sole training
    objective (verified against `corn_loss` directly). `save_joint_model_weights()`/
    `load_joint_model_weights()` — weights-only, path-parameterized, no built-in Drive/local
    default (verified round-trip, `atol=1e-5`).
  - Stage 3/4 never appear in the joint model's graph at all (verified: no Stage 3/4 variable name
    appears in `trainable_variables`) — they run only inside `joint_training_dataset.py`'s data
    layer, producing plain NumPy arrays before anything reaches the model's `Input` tensors. A
    single `GradientTape` step confirms zero missing gradients across all 393 trainable variables.
- **`colab/notebooks/stage08_corn_classifier.ipynb`:** repurposed with real infrastructure cells
  (Drive/environment/dataset/checkpoint verification, authoritative split, joint model
  construction + compile, a synthetic-tensor smoke test) plus a training-configuration cell
  (`batch_size=2`, `mixed_precision=True`, `monitor="val_QWK"`, `mode="max"`) and
  dataset-loading/`Trainer` cells gated behind `RUN_TRAINING = False` — opening or running this
  notebook as committed does not start training.
- **Next step:** set `RUN_TRAINING = True` and actually run joint training on a real Colab T4 —
  not attempted by this step, per its own explicit no-training instruction.

### Step 9 — Uncertainty Estimation (Monte Carlo Dropout) — integration only
- **Why:** Already implemented and correct in `bayesian_inference.py`; this step re-points it at the new model.

### Step 10 — Explainability (Grad-CAM++, SHAP, Attention Rollout)
- **Why:** Current `explainable_ai.py` only implements vanilla Grad-CAM against a single hard-coded layer name; generalize and add the two missing techniques.

### Step 11 — Evaluation
- **Why:** Consolidate the currently-duplicated evaluation logic into one module, and add QWK / real calibration for the new end-to-end model.

---

## 5. Open Questions Before Implementation Begins

1. **Segmentation datasets (Steps 3–4) — resolved.** Stage 3 needs no project dataset — it integrates a pretrained external checkpoint (LWNet) for inference only; IDRiD's segmentation subset trains Stage 4. See `SEGMENTATION_ARCHITECTURE.md`.
2. **`dr_gan.py`'s role going forward:** keep feeding the new ordinal classifier's training set the same way it currently feeds `dr_classifier.py`, or treat it as legacy/optional — undecided, out of scope for this refactor.
3. **Local Feature Extraction's input (Step 5) — resolved.** RGB image + vessel map + lesion maps, concatenated into an 8-channel tensor (§4 of `SEGMENTATION_ARCHITECTURE.md`).
4. **LWNet integration details** — exact FOV-mask/resize/threshold behavior to vendor from the upstream `predict_one_image.py`, and how much of `external/lwnet/` (temporary, inspection-only) can be discarded once integration is complete, per `SEGMENTATION_ARCHITECTURE.md` §2.

---

## 6. Next Step

Stages 1–4 are complete. Stage 4 (Lesion Segmentation) is finalized as Experiment 2C and is now
**frozen** — no further training, loss, or architecture changes to it, and no Experiment 2D is
planned. Stage 5 (Local Feature Extraction), Stage 6 (Global Feature Extraction), Stage 7 (Feature
Fusion / Adaptive Cross-Attention), **RACAF**, and **CORN (Stage 08)** are all now **implemented
and unit-tested, but not trained and not frozen** — none has a standalone training objective; all
five await the joint Stage 05–08 + RACAF training script (still not implemented). RACAF's
implementation (`racaf.py`) follows exactly the approved, audited design
(`RACAF_ARCHITECTURE.md`): frozen-Stage-04 TTA disagreement, population variance, burden-weighted
scalar reliability `r`, and a 295,170-parameter gate + Global-readout fusion model. CORN's
implementation (`corn.py`) follows exactly the approved design (`CORN_ARCHITECTURE.md`): a
1,028-parameter `Dense(256->4)` head on RACAF's `F`, the standard CORN conditional-subset loss and
sigmoid/cumulative-product decoding — no second research innovation. Both verified against the
real Stage 05/06/07 models end-to-end. The downstream dataset infrastructure this joint training
will need — one authoritative, stratified APTOS2019 train/val split (`downstream_split.py`) shared
by every stage from Stage 5 through CORN, plus Drive path resolution for APTOS2019/IDRiD/the frozen
Stage 1/3/4 checkpoints/the persistent Stage 03-04/RACAF caches, plus a canonical-resolution
(rather than native-image-resolution) Stage 03/04 prediction cache — is in place; see
`JOINT_TRAINING_ARCHITECTURE.md` for the full design (dataset flow, caching, gradient boundary,
loss, checkpoint format, QWK-based model selection, T4 starting configuration, and the notebook
role). **The joint model builder (`joint_training_model.py`) and joint dataset loader
(`joint_training_dataset.py`) are now implemented and unit-tested (Step 8.5 above) — no training
was run at any point in this pipeline; no checkpoint exists anywhere in this repository or on
Drive.** Per the "one module at a time, wait for approval" rule, the next implementation target is
to actually **run** joint training on a real Colab T4 (set `RUN_TRAINING = True` in
`colab/notebooks/stage08_corn_classifier.ipynb`) — not attempted here.

RACAF is the one approved research innovation for this project (`PROJECT_CODE.md`'s "Approved
Research Innovation" section); no second, competing innovation should be introduced without a
deliberate revision of that decision — Stage 7's cross-attention mechanism, every individual
technique RACAF itself uses (TTA, predictive variance, sigmoid gating, linear projection), and
CORN's ordinal-regression methodology (Shi, Cao & Raschka, 2021) all remain established/
literature-derived engineering, not research contributions in their own right; only RACAF's
specific integration of them is.
