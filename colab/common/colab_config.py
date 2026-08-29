"""
Single source of truth for every path used by the Colab training workflow --
shared by every stage notebook (`colab/notebooks/stage01_iqa.ipynb` today;
`stage02_preprocessing.ipynb` onward once each stage is implemented).

Pure configuration only -- no side effects, no `google.colab` import, no
filesystem access. This means it can be imported and inspected (including
outside Colab, e.g. for static validation) without mounting Drive or
cloning anything; `setup.py` is where those actions actually happen.

Every other `colab/common/` module, and every notebook, should import paths
from here rather than hardcoding them.
"""

import posixpath

import drive_paths

# --- Repository (Colab VM local disk; ephemeral, code only) ---
REPO_URL = "https://github.com/yasodharan27/diabetic_retinoplasty.git"
REPO_BRANCH = "main"
REPO_DIR = "/content/diabetic_retinoplasty"
REPOSITORY_ROOT = REPO_DIR

# --- Google Drive (persistent; datasets + all training outputs) ---
DRIVE_MOUNT_POINT = "/content/drive"

DRIVE = drive_paths.build_drive_paths(DRIVE_MOUNT_POINT)

DRIVE_ROOT = DRIVE.project_root
DATASET_ROOT = DRIVE.datasets_root
EXPERIMENT_ROOT = DRIVE.experiments_root
TENSORBOARD_ROOT = DRIVE.tensorboard_root
EXPORTED_MODELS_ROOT = DRIVE.exported_models_root
LOGS_ROOT = DRIVE.logs_root

# --- Dataset roots -- available to every stage, not just IQA ---
EYEQ_RAW_DIR = DRIVE.eyeq_raw_dir
EYEQ_PROCESSED_DIR = DRIVE.eyeq_processed_dir

APTOS2019_DATASET_DIR = DRIVE.aptos2019_dataset_dir
APTOS2019_RAW_DIR = DRIVE.aptos2019_raw_dir
APTOS2019_PROCESSED_DIR = DRIVE.aptos2019_processed_dir

# IDRiD is not flat -- see drive_paths.py's docstring for the epistemic
# status of these per-subset paths (verified locally, inferred-by-convention
# on Drive until a real Colab session confirms it via
# verify_dataset.verify_idrid_dataset_dir()).
IDRID_DATASET_DIR = DRIVE.idrid_dataset_dir
IDRID_GRADING_RAW_DIR = DRIVE.idrid_grading_raw_dir
IDRID_GRADING_PROCESSED_DIR = DRIVE.idrid_grading_processed_dir
IDRID_LOCALIZATION_RAW_DIR = DRIVE.idrid_localization_raw_dir
IDRID_LOCALIZATION_PROCESSED_DIR = DRIVE.idrid_localization_processed_dir
IDRID_SEGMENTATION_RAW_DIR = DRIVE.idrid_segmentation_raw_dir
IDRID_SEGMENTATION_PROCESSED_DIR = DRIVE.idrid_segmentation_processed_dir

# --- Per-module experiment / export directories ---
# `drive_paths.PIPELINE_MODULES` (currently IQA, VesselSegmentation,
# LesionSegmentation, FinalClassification) is the set of *independently
# trained* models per PROJECT_CODE.md's Models table -- this deliberately
# does not have a 1:1 entry per pipeline *stage* (11 stages; see
# PROJECT_STRUCTURE.md's Pipeline Overview). Stages that are not
# independently trained (e.g. Uncertainty Estimation, Explainability) reuse
# an upstream module's exported model rather than needing their own
# experiment bucket -- resolve those via `DRIVE.experiment_dir(module)` /
# `DRIVE.exported_model_dir(module)` directly once that stage's training
# approach is finalized, rather than adding a placeholder constant here.

# --- Stage 01 -- Image Quality Assessment (the only fully implemented stage) ---
IQA_MODULE_NAME = "IQA"
IQA_EXPERIMENTS_DIR = DRIVE.experiment_dir(IQA_MODULE_NAME)
IQA_EXPORTED_MODELS_DIR = DRIVE.exported_model_dir(IQA_MODULE_NAME)

# Fixed, load-bearing constant, not a knob: image_quality_dataset.py's
# load_eyeq_datasets() always splits with this fraction; recorded here only
# so verify_dataset.py can report the *expected* split sizes ahead of time.
EYEQ_VAL_SPLIT = 0.15

# --- Frozen Stage 1/3/4 checkpoint locations (Drive) ---
# config.py's IQA_MODEL_DIR / VESSEL_SEG_MODEL_DIR / LESION_SEG_MODEL_DIR
# each default to a path inside the LOCAL repository checkout when their
# matching env var is unset -- correct for local development, but wrong
# inside a fresh Colab session, where that checkout is the just-cloned,
# ephemeral VM copy that never contains these already-trained checkpoints.
# These resolve the SAME already-existing, already-verified Drive paths
# `exported_models/<Module>/` (via drive_paths.PIPELINE_MODULES) already
# uses for Stage 1's checkpoint -- extended here to also cover Stage 3/4,
# which were not previously wired into any Colab-facing constant at all.
IQA_MODEL_DIR = IQA_EXPORTED_MODELS_DIR
VESSEL_SEG_MODEL_DIR = DRIVE.exported_model_dir("VesselSegmentation")
LESION_SEG_MODEL_DIR = DRIVE.exported_model_dir("LesionSegmentation")

# --- Persistent, per-image derived caches (Stage 03/04 predictions, RACAF
# reliability) for the joint Stage 05-08+RACAF training design -- see
# drive_paths.py's docstring for why these live under a new `cache/` Drive
# root rather than `experiments/` (per-run, would defeat cross-run reuse)
# or `exported_models/` (final trained weights, not a derived cache). ---
LOCAL_FEATURE_CACHE_DIR = DRIVE.cache_dir("LocalFeatureExtraction")
RACAF_CACHE_DIR = DRIVE.cache_dir("RACAF")

# --- Stage 05-08 + RACAF joint training ("FinalClassification") ---
# These five stages train jointly, as ONE model, under the already-reserved
# "FinalClassification" PIPELINE_MODULES entry (PROJECT_STRUCTURE.md's own
# note: "a single experiments/FinalClassification/ bucket, not four
# separate ones") -- so each stage's own exported best_model is nested
# under that single reserved directory, rather than adding five new
# PIPELINE_MODULES entries for stages with no independent checkpoint of
# their own.
FINAL_CLASSIFICATION_EXPORTED_DIR = DRIVE.exported_model_dir("FinalClassification")
LOCAL_FEATURE_MODEL_DIR = posixpath.join(FINAL_CLASSIFICATION_EXPORTED_DIR, "local_feature_extraction")
GLOBAL_FEATURE_MODEL_DIR = posixpath.join(FINAL_CLASSIFICATION_EXPORTED_DIR, "global_feature_extraction")
FEATURE_FUSION_MODEL_DIR = posixpath.join(FINAL_CLASSIFICATION_EXPORTED_DIR, "feature_fusion")
RACAF_MODEL_DIR = posixpath.join(FINAL_CLASSIFICATION_EXPORTED_DIR, "racaf")
CORN_MODEL_DIR = posixpath.join(FINAL_CLASSIFICATION_EXPORTED_DIR, "corn")
