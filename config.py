"""
Centralized configuration for the diabetic retinopathy detection pipeline.

Loads the .env file exactly once (here) and exposes every environment-variable-backed
path used across the repository, grouped by purpose. Every script that previously did:

    from dotenv import load_dotenv
    load_dotenv()
    BASE_PATH = os.environ.get('BASE_PATH')
    ...

should instead do:

    from config import BASE_PATH, ...

The flat names below are identical to the environment variable names each script
already used, so this is a drop-in replacement -- no downstream code needs to change.

Note: per-script training hyperparameters (BATCH_SIZE, EPOCHS, IMAGE_SIZE, NUM_CLASSES,
CHANNELS, LATENT_DIM, TARGET_SIZE, etc.) are intentionally NOT centralized here. They are
not read from the environment anywhere in the current codebase -- each script hardcodes
its own value locally, and those values differ by design (e.g. BATCH_SIZE is 4 in
train_hybrid_model.py, 8 in efficientnet_model.py, 16 in dr_classifier.py; EPOCHS is 10,
20, or 30 depending on the script). Centralizing them would change behavior, which is out
of scope for this module.
"""

import os
from dataclasses import dataclass
from typing import Optional

from dotenv import load_dotenv

load_dotenv()


@dataclass(frozen=True)
class DatasetPaths:
    """Raw dataset inputs: source CSV + source images (APTOS-style)."""
    csv_path: Optional[str]
    image_dir: Optional[str]
    test_image_dir: Optional[str]


@dataclass(frozen=True)
class OutputPaths:
    """Preprocessing outputs: dataset splits, processed images, GAN samples."""
    output_dir: Optional[str]
    processed_images_dir: Optional[str]
    processed_test_dir: Optional[str]
    gan_images_dir: Optional[str]


@dataclass(frozen=True)
class ModelPaths:
    """Where trained models are saved to / loaded from."""
    model_save_path: Optional[str]
    model_path: Optional[str]
    retrained_model_path: Optional[str]


@dataclass(frozen=True)
class ResultsPaths:
    """Where evaluation/inference outputs (CSVs, plots) are written."""
    results_path: Optional[str]
    results_dir: Optional[str]


@dataclass(frozen=True)
class EyeQPaths:
    """Image Quality Assessment (EyeQ) dataset, model, and results paths.

    Unlike the flat env-var-only fields above, these default to fixed
    locations inside the repository (datasets/EyeQ/raw already ships with
    the repo, so no .env entry is required to get started) while still being
    overridable via the matching environment variable.
    """
    raw_dir: str
    model_dir: str
    results_dir: str


# --- Base ---
BASE_PATH: Optional[str] = os.environ.get('BASE_PATH')

_REPO_ROOT = os.path.dirname(os.path.abspath(__file__))

# --- Grouped, typed access ---
DATASET = DatasetPaths(
    csv_path=os.environ.get('CSV_PATH'),
    image_dir=os.environ.get('IMAGE_DIR'),
    test_image_dir=os.environ.get('TEST_IMAGE_DIR'),
)

OUTPUT = OutputPaths(
    output_dir=os.environ.get('OUTPUT_DIR'),
    processed_images_dir=os.environ.get('PROCESSED_IMAGES_DIR'),
    processed_test_dir=os.environ.get('PROCESSED_TEST_DIR'),
    gan_images_dir=os.environ.get('GAN_IMAGES_DIR'),
)

MODELS = ModelPaths(
    model_save_path=os.environ.get('MODEL_SAVE_PATH'),
    model_path=os.environ.get('MODEL_PATH'),
    retrained_model_path=os.environ.get('RETRAINED_MODEL_PATH'),
)

RESULTS = ResultsPaths(
    results_path=os.environ.get('RESULTS_PATH'),
    results_dir=os.environ.get('RESULTS_DIR'),
)

EYEQ = EyeQPaths(
    raw_dir=os.environ.get('EYEQ_RAW_DIR') or os.path.join(_REPO_ROOT, 'datasets', 'EyeQ', 'raw'),
    model_dir=os.environ.get('IQA_MODEL_DIR') or os.path.join(_REPO_ROOT, 'models', 'image_quality_assessment'),
    results_dir=os.environ.get('IQA_RESULTS_DIR') or os.path.join(_REPO_ROOT, 'results', 'image_quality_assessment'),
)

# --- Flat, backward-compatible names ---
# Match the exact variable names every script previously assigned via
# `os.environ.get(...)`, so existing code can switch to `from config import X`
# without renaming anything or changing behavior.
CSV_PATH = DATASET.csv_path
IMAGE_DIR = DATASET.image_dir
TEST_IMAGE_DIR = DATASET.test_image_dir

OUTPUT_DIR = OUTPUT.output_dir
PROCESSED_IMAGES_DIR = OUTPUT.processed_images_dir
PROCESSED_TEST_DIR = OUTPUT.processed_test_dir
GAN_IMAGES_DIR = OUTPUT.gan_images_dir

MODEL_SAVE_PATH = MODELS.model_save_path
MODEL_PATH = MODELS.model_path
RETRAINED_MODEL_PATH = MODELS.retrained_model_path

RESULTS_PATH = RESULTS.results_path
RESULTS_DIR = RESULTS.results_dir

EYEQ_RAW_DIR = EYEQ.raw_dir
IQA_MODEL_DIR = EYEQ.model_dir
IQA_RESULTS_DIR = EYEQ.results_dir
