"""
Single source of truth for every path used by the Colab training workflow.

Pure configuration only -- no side effects, no `google.colab` import, no
filesystem access. This means it can be imported and inspected (including
outside Colab, e.g. for static validation) without mounting Drive or
cloning anything; `setup.py` is where those actions actually happen.

Every other `colab/` module, and the notebook itself, should import paths
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

# --- Image Quality Assessment (this notebook's module) ---
IQA_MODULE_NAME = "IQA"
EYEQ_RAW_DIR = DRIVE.eyeq_raw_dir
IQA_EXPERIMENTS_DIR = DRIVE.experiment_dir(IQA_MODULE_NAME)
IQA_EXPORTED_MODELS_DIR = DRIVE.exported_model_dir(IQA_MODULE_NAME)

# --- Fixed, load-bearing constant, not a knob: image_quality_dataset.py's
# load_eyeq_datasets() always splits with this fraction; recorded here only
# so verify_dataset.py can report the *expected* split sizes ahead of time.
EYEQ_VAL_SPLIT = 0.15
