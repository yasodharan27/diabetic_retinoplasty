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
