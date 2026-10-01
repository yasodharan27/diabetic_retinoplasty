"""Constants and locations for the v2 Stage 3-8 pipeline (research record §40,
`research/Grade3vs4_Architecture_Research/IMPLEMENTATION_SPEC_STAGE3_8_V2.md`).

The v2 pipeline is Stage 2 RGB -> Stage 3 LWNet vessel map (reused, parity-checked) + a NEW Stage 4
multi-label pathology segmenter -> Stage 5 prior encoder -> Stage 6/7 ConvNeXt-T with gated prior
injection -> Stage 8 CORN.

Nothing here touches the legacy Stage 3/4 code, checkpoints or caches. The legacy locations are
listed only so the v2 code can REFUSE to read from or write to them (`stage34_cache_v2`)."""
import os

import config

# ------------------------------------------------------------------ identities (verified 2026-10-01)
# Deny-listed: the old Stage 4 Attention U-Net (exported_models/LesionSegmentation/best_model.keras).
LEGACY_STAGE4_SHA256 = "64b3c0468bdc77204e7126c4ff06aa3fbbdab1430349217e5d6baaa69888ce76"
# Stage 3 LWNet (models/vessel_segmentation/best_model.pth == exported_models/VesselSegmentation/best_model.pth).
STAGE3_LWNET_SHA256 = "91f0cada4b26ece63464b05be60f9b3a51f1bcd1f764081d07d3a35961118de1"
# Authoritative APTOS train/validation split (LF-normalised CSV).
SPLIT_SHA256 = "bc80fd450340b09307fbd80a1b00553e70e34d64a3cdf94635162b6c1e99aca5"
# Stage 6 RGB trunk: pinned Keras ImageNet ConvNeXt-Tiny (as P).
CONVNEXT_TINY_WEIGHTS_SHA256 = "d547c096cabd03329d7be5562c5e14798aa39ed24b474157cef5e85ab9e49ef1"

# Stage 4 v2 encoder: ImageNet SE-ResNet-101 as re-hosted by segmentation_models_pytorch 0.5.0
# (original weights: http://data.lip6.fr/cadene/pretrainedmodels/se_resnet101-7e38fcc6.pth).
# The user decision (2026-10-01) is: use exactly these weights; NO silent fallback encoder.
SMP_VERSION = "0.5.0"
STAGE4_ENCODER = "se_resnet101"
STAGE4_ENCODER_WEIGHTS = "imagenet"
STAGE4_ENCODER_HF_REPO = "smp-hub/se_resnet101.imagenet"
STAGE4_ENCODER_HF_REVISION = "71fe95cc0a27f444cf83671f354de02dc741b18b"
STAGE4_ENCODER_HF_FILE = "model.safetensors"
STAGE4_ENCODER_SHA256 = "f4ff9b3dbf8e7bf3929b02918d64d09ceb76ce04ea95e0ddc937879c00851fc6"
STAGE4_ENCODER_BYTES = 197_790_168
STAGE4_DECODER_CHANNELS = (256, 128, 64, 32, 16)
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)

# ------------------------------------------------------------------ geometry and channels
CACHE_SIZE = 512                 # canonical frame: full native image directly resized (no crop)
STAGE4_INFERENCE_SIZE = 1536     # the same full-frame direct resize at exactly 3x
POOL_FACTOR = STAGE4_INFERENCE_SIZE // CACHE_SIZE
assert POOL_FACTOR * CACHE_SIZE == STAGE4_INFERENCE_SIZE
STAGE4_TRAIN_PATCH = 512
POOLINGS = ("mean", "max")       # per class, in this order
STAGE4_V2A_CLASSES = ("MA", "HE", "EX", "SE")   # K = 4 first; OD excluded (§40 decision 2)
CANONICAL_CLASSES = ("MA", "HE", "EX", "SE", "NV", "IRMA", "PRH", "VH", "FP", "LASER", "OD")

PREPROC_VERSION = "stage2-DR-profile/full-frame-direct-resize/v1"
CACHE_SCHEMA_VERSION = 1

# ------------------------------------------------------------------ v2 locations
# On Colab `config.LOCAL_FEATURE_RESULTS_DIR` is .../DiabeticRetinopathy/cache/LocalFeatureExtraction,
# so its parent is the Drive cache root; locally the same relation holds under results/.
CACHE_ROOT = os.path.dirname(config.LOCAL_FEATURE_RESULTS_DIR)
EXPORTED_MODELS_ROOT = os.path.dirname(config.LESION_SEG_MODEL_DIR)

STAGE2_RGB_GENERATION = "rgb-v1"
STAGE3_GENERATION = "s3-" + STAGE3_LWNET_SHA256[:8]
STAGE2_ROOT = os.path.join(CACHE_ROOT, "Stage2")
STAGE3_ROOT = os.path.join(CACHE_ROOT, "Stage3")
STAGE4_ROOT = os.path.join(CACHE_ROOT, "Stage4")
BUNDLE_ROOT = os.path.join(CACHE_ROOT, "Bundle")
STAGE4_V2_MODEL_ROOT = os.path.join(EXPORTED_MODELS_ROOT, "LesionSegmentation_v2")
LOCAL_V2_ROOT = "/content/cache_v2"

# ------------------------------------------------------------------ TJDR (Stage-4 v2 training data)
# Verified in research record §42 (TJDR_PRETRAINING_AUDIT.md) and re-verified by
# tjdr_dataset.verify_tjdr() before `DatasetSpec("TJDR").verified` was set (record §43).
TJDR_DATASET_NAME = "TJDR"                         # config.dataset_raw_dir("TJDR"); Colab: TJDR_RAW_DIR
TJDR_DRIVE_RAW_DIR = "DiabeticRetinopathy/datasets/TJDR/raw"   # relative to the Drive root (MyDrive)
TJDR_SOURCE = "github.com/NekoPii/TJDR -> Google Drive folder 1RBAtPPAvX1KXiJrNsrJz7jSMCL3AgAJT"
TJDR_SOURCE_LISTING_SHA256 = "31cdefa2fcfdaab74da4b5672167ab299eb6f0ca30b33712138e845c93d02e06"
TJDR_MASK_MODE = "P"                               # palette-indexed PNG, never .convert()-ed
TJDR_MASK_VALUES = (0, 1, 2, 3, 4)                 # 0 = background (a true negative for all 4 classes)
TJDR_MASK_CODES = {"MA": 3, "HE": 2, "EX": 1, "SE": 4}     # paper: EX(1) HE(2) MA(3) SE(4); verified §42
TJDR_OFFICIAL_COUNTS = {"train": 448, "test": 113}
# §42 R1: exact duplicates (different masks) and same-eye near-duplicates; the lower id is kept.
TJDR_EXCLUDED = {
    "train": ("TJDR_train_041", "TJDR_train_042", "TJDR_train_091", "TJDR_train_174", "TJDR_train_105"),
    "test": ("TJDR_test_021", "TJDR_test_023", "TJDR_test_003"),
}
TJDR_EXCLUSION_REASONS = {
    "TJDR_train_041": "byte-identical to train_039", "TJDR_train_042": "byte-identical to train_040",
    "TJDR_train_091": "byte-identical to train_089", "TJDR_train_174": "byte-identical to train_173",
    "TJDR_train_105": "same-eye near-duplicate of train_100 (r=0.96)",
    "TJDR_test_021": "byte-identical to train_092 (cross-split leak)",
    "TJDR_test_023": "byte-identical to train_101 (cross-split leak)",
    "TJDR_test_003": "same-eye near-duplicate of test_002 (r=0.82)",
}
TJDR_USABLE_COUNTS = {"train": 443, "test": 110}
# Images containing each class (measured, §42 / §43): official full set and the usable subset.
TJDR_CLASS_IMAGE_COUNTS_OFFICIAL = {"train": {"MA": 137, "HE": 249, "EX": 255, "SE": 151},
                                    "test": {"MA": 37, "HE": 69, "EX": 67, "SE": 41}}
TJDR_CLASS_IMAGE_COUNTS_USABLE = {"train": {"MA": 135, "HE": 245, "EX": 251, "SE": 147},   # §43 verify_tjdr
                                  "test": {"MA": 37, "HE": 67, "EX": 64, "SE": 39}}

# ------------------------------------------------------------------ IDRiD (Stage-4 v2 training data)
# Only the 54-image lesion-segmentation TRAINING set is used for Stage-4 training/validation. The 27-image
# test set (IDRiD_55..IDRiD_81) is reachable only through stage4_v2_gate (one-time gate); the grading
# set (IDRiD_001..IDRiD_516, datasets/IDRiD/grading) never enters Stage 4.
IDRID_SEG_TRAIN_IDS = tuple(f"IDRiD_{i:02d}" for i in range(1, 55))
IDRID_SEG_TEST_IDS = tuple(f"IDRiD_{i:02d}" for i in range(55, 82))
# Pinned internal validation split: stratified by SE presence (26/54 SE+ -> 5 SE+ + 5 SE-), drawn with
# numpy default_rng(20261001) from the sorted SE+ / SE- lists; recomputed and asserted by
# stage4_v2_data.idrid_split().
IDRID_V2_SPLIT_SEED = 20261001
IDRID_V2_VAL_IDS = ("IDRiD_05", "IDRiD_10", "IDRiD_11", "IDRiD_14", "IDRiD_15",
                    "IDRiD_18", "IDRiD_24", "IDRiD_30", "IDRiD_35", "IDRiD_48")
IDRID_V2_VAL_SE_POSITIVE = ("IDRiD_14", "IDRiD_18", "IDRiD_30", "IDRiD_35", "IDRiD_48")
IDRID_V2_SPLIT_SHA256 = "2e5bf1c36f6126e68d0356c6252665bbb6871aab0d725bff08efb11439142f30"

# ------------------------------------------------------------------ Stage-4 v2 training-input cache
# Stage-2 DR output (existing Drive `processed/` for IDRiD; TJDR/processed created once) -> direct
# full-frame resize to 1536^2 (uint8) + the 4 binary class masks at 1536 (current rule, threshold 0.5).
# NOT a prediction cache; never read by Stage 5-8. Roles: train / val only (no test images).
STAGE4_TRAIN_CACHE_GENERATION = "s4train-v1"
STAGE4_TRAIN_CACHE_ROOT = os.path.join(CACHE_ROOT, "Stage4Train")
STAGE4_TRAIN_ROLES = ("train", "val")
STAGE4_TRAIN_EXPECTED = {("IDRiD", "train"): 44, ("IDRiD", "val"): 10,
                         ("TJDR", "train"): 443, ("TJDR", "val"): 110}

# ------------------------------------------------------------------ Stage-4 v2 training configuration
# Fixed by §40: K=4, smp Unet + pinned SE-ResNet-101, encoder LR = 0.1x decoder, partial-label loss,
# w+ = clip(sqrt(neg/pos), 1, 20), lambda = 1 (core), grad clip 1.0, EMA 0.999, 512 patches with
# p=0.5 class-aware centring, dataset-balanced batches. Set here (not fixed by §40): the optimiser
# schedule / step budget below, sized for about 3 T4-h.
STAGE4_TRAIN_DEFAULTS = {
    "seed": 42, "classes": STAGE4_V2A_CLASSES, "patch": STAGE4_TRAIN_PATCH, "p_lesion": 0.5,
    "batch_size": 8, "datasets_per_batch": ("IDRiD", "TJDR"), "iterations": 12000, "warmup_steps": 500,
    "decoder_lr": 3e-4, "encoder_lr_factor": 0.1, "weight_decay": 1e-4, "min_lr_factor": 0.01,
    "grad_clip": 1.0, "ema_decay": 0.999, "amp": True, "flip_augmentation": True,
    "val_every": 1000, "checkpoint_every": 500, "num_workers": 2,
    "selection_metric": "mean over {IDRiD-val, TJDR-val-TRC50DX, TJDR-val-CLARUS500} of the 4-class mean AUPR (EMA model)",
}
STAGE4_AP_BINS = 10000        # streaming histogram AUPR (probability resolution 1e-4)

# ------------------------------------------------------------------ one-time IDRiD test gate (§40 step 3)
# Documented old Stage-4 IDRiD-test Dice (SEGMENTATION_ARCHITECTURE.md / stage04 notebook output), old
# protocol: dataset-pooled soft Dice (smooth 1) at 512^2, targets resized with the old `> 0` rule.
STAGE4_GATE_REFERENCE_DICE = {"MA": 0.0165, "HE": 0.1273, "EX": 0.3574, "SE": 0.0244}
STAGE4_GATE_MIN_MEAN_AUPR = 0.55
STAGE4_GATE_CONFIRM_TOKEN = "RUN_THE_ONE_TIME_IDRID_TEST_GATE"

# ------------------------------------------------------------------ APTOS population (downstream, §46)
# The authoritative split (multiseed_runs.verify_split: 2929/733, sha SPLIT_SHA256) minus the 11 pinned
# empty-field-of-view images (experiments/ImprovedTraining/improved_multiseed_2026_09/experiment_manifest.json)
# = 2921 train + 730 val = 3651, the population every Stage 5-8 experiment so far used.
APTOS_EMPTY_FOV_IDS = ("14ee87d6cc42", "188a9323be03", "2241b7e90782", "262ad704319c", "26453eb7e989",
                       "3a122851e526", "453a1e2754b2", "6c315ad3d07f", "7356dd08b0ae", "9785805af1b8",
                       "a6c9e96a10d7")
APTOS_EXPECTED_COUNTS = {"train": 2921, "val": 730}
APTOS_POPULATION = 3651

# ------------------------------------------------------------------ APTOS cache generation (§46)
STAGE3_PARITY_N = 25            # fixed-seed parity ids (spec §2)
STAGE3_PARITY_TOL = 1e-4
RGB_PARITY_TOL = 1e-6           # canonical RGB recomputed from the Stage-2 output must match the cache
FRESHNESS_CANARY_N = 8          # recompute 8 Stage-4 maps after generation; max |delta| <= 2/255
CANARY_TOL_COUNTS = 2
PARITY_SEED = 20261001
STAGE4_GATE_OVERRIDE_TOKEN = "PROCEED_WITH_A_STAGE4_MODEL_THAT_FAILED_THE_GATE"

# ------------------------------------------------------------------ C2 information screen (§40 §10, pre-registered)
C2_R_FEATURES_REL = "HR_Screen_ImageNet/v1/hr_screen_features.npz"     # under Drive experiments/
C2_R_FEATURES_SHA256 = "0f8addb8382ae60a5be3a1bfb4f1e01def47c60a4dbc9373272a1c9cb3c0a245"
C2_K_TOTAL = 64                 # every arm has 64 probe dimensions (two-block arms: 32 + 32)
C2_N_REPEATS = 10               # outer stratified 5-fold x 10 repeats, identical folds for all arms
C2_N_FOLDS = 5
C2_CS = (0.01, 0.1, 1.0, 10.0)
C2_N_BOOT = 2000                # paired, grade-stratified bootstrap
C2_SEED = 20261001
C2_CUTS = (1, 2, 3, 4)          # cumulative cuts grade >= c
C2_MIN_DELTA = 0.005            # PASS iff R+Q - R >= +0.005 AND bootstrap CI lower > 0 AND R+Q >= Q

# ------------------------------------------------------------------ experiment roots (Drive experiments/)
EXPERIMENT_DIRS = {"stage4": "Stage4V2", "c2": "C2", "arch1": "Architecture1"}

# ------------------------------------------------------------------ legacy locations (never used)
LEGACY_DIRS = (
    config.LOCAL_FEATURE_RESULTS_DIR,                              # loose legacy lesion/vessel/rgb cache
    os.path.join(config.LOCAL_FEATURE_RESULTS_DIR, "stage03_stage04_cache"),
    config.RACAF_RESULTS_DIR,                                      # legacy RACAF reliability cache
    os.path.join(CACHE_ROOT, "cache_archive"),                     # legacy mixed archive shards
    config.LESION_SEG_MODEL_DIR,                                   # legacy Stage 4 checkpoint
    "/content/cache/local_feature_extraction",
    "/content/cache/racaf",
)
