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
from typing import Optional, Tuple

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
class VesselSegmentationPaths:
    """Vessel Segmentation (Stage 03) model and results paths.

    Unlike `EyeQPaths`/`ModelPaths` above, `model_dir` holds a vendored
    pretrained checkpoint (LWNet), not this project's own training output --
    see SEGMENTATION_ARCHITECTURE.md Sec 1.2/2/6. `results_dir` is used only
    for smoke-test / qualitative-verification artifacts (sample predictions,
    visualizations), not a training evaluation report, since this stage has
    no training or benchmark run of its own yet."""
    model_dir: str
    results_dir: str


@dataclass(frozen=True)
class LesionSegmentationPaths:
    """Lesion Segmentation (Stage 04) model and results paths.

    Unlike `VesselSegmentationPaths`, `model_dir` holds this project's own
    trained `.keras` output, not a vendored checkpoint -- Lesion
    Segmentation has a full training workflow of its own within this
    project (SEGMENTATION_ARCHITECTURE.md Sec 3/6/7.0)."""
    model_dir: str
    results_dir: str


@dataclass(frozen=True)
class LocalFeatureExtractionPaths:
    """Local Feature Extraction (Stage 05) model and results paths.

    Mirrors `LesionSegmentationPaths` exactly: `model_dir` will hold this
    project's own trained `.keras` output once Stage 05 is trained (not a
    vendored checkpoint), and `results_dir` additionally holds the cached
    Stage 03/04 inference outputs Stage 05's dataset loader computes over
    APTOS 2019 (see `local_feature_extraction_dataset.py`) -- a derived,
    regenerable artifact, not raw dataset content, so it lives here rather
    than under `datasets/`."""
    model_dir: str
    results_dir: str


@dataclass(frozen=True)
class GlobalFeatureExtractionPaths:
    """Global Feature Extraction (Stage 06) model and results paths.

    Mirrors `LocalFeatureExtractionPaths` exactly. Unlike Stage 05, Stage
    06 has no frozen upstream (Stage 03/04) inference to cache -- its
    `results_dir` exists only for parity with the per-stage
    model_dir/results_dir convention every other trainable stage in this
    project already uses."""
    model_dir: str
    results_dir: str


@dataclass(frozen=True)
class FeatureFusionPaths:
    """Adaptive Cross-Attention / Feature Fusion (Stage 07) model and
    results paths. Mirrors `GlobalFeatureExtractionPaths` exactly -- Stage
    07, like Stage 06, has no frozen upstream inference to cache; this
    exists only for parity with the per-stage model_dir/results_dir
    convention every other trainable stage in this project already uses."""
    model_dir: str
    results_dir: str


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

VESSEL_SEGMENTATION = VesselSegmentationPaths(
    model_dir=os.environ.get('VESSEL_SEG_MODEL_DIR') or os.path.join(_REPO_ROOT, 'models', 'vessel_segmentation'),
    results_dir=os.environ.get('VESSEL_SEG_RESULTS_DIR') or os.path.join(_REPO_ROOT, 'results', 'vessel_segmentation'),
)

LESION_SEGMENTATION = LesionSegmentationPaths(
    model_dir=os.environ.get('LESION_SEG_MODEL_DIR') or os.path.join(_REPO_ROOT, 'models', 'lesion_segmentation'),
    results_dir=os.environ.get('LESION_SEG_RESULTS_DIR') or os.path.join(_REPO_ROOT, 'results', 'lesion_segmentation'),
)

LOCAL_FEATURE_EXTRACTION = LocalFeatureExtractionPaths(
    model_dir=os.environ.get('LOCAL_FEATURE_MODEL_DIR') or os.path.join(_REPO_ROOT, 'models', 'local_feature_extraction'),
    results_dir=os.environ.get('LOCAL_FEATURE_RESULTS_DIR') or os.path.join(_REPO_ROOT, 'results', 'local_feature_extraction'),
)

GLOBAL_FEATURE_EXTRACTION = GlobalFeatureExtractionPaths(
    model_dir=os.environ.get('GLOBAL_FEATURE_MODEL_DIR') or os.path.join(_REPO_ROOT, 'models', 'global_feature_extraction'),
    results_dir=os.environ.get('GLOBAL_FEATURE_RESULTS_DIR') or os.path.join(_REPO_ROOT, 'results', 'global_feature_extraction'),
)

FEATURE_FUSION = FeatureFusionPaths(
    model_dir=os.environ.get('FEATURE_FUSION_MODEL_DIR') or os.path.join(_REPO_ROOT, 'models', 'feature_fusion'),
    results_dir=os.environ.get('FEATURE_FUSION_RESULTS_DIR') or os.path.join(_REPO_ROOT, 'results', 'feature_fusion'),
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

VESSEL_SEG_MODEL_DIR = VESSEL_SEGMENTATION.model_dir
VESSEL_SEG_RESULTS_DIR = VESSEL_SEGMENTATION.results_dir

LESION_SEG_MODEL_DIR = LESION_SEGMENTATION.model_dir
LESION_SEG_RESULTS_DIR = LESION_SEGMENTATION.results_dir

LOCAL_FEATURE_MODEL_DIR = LOCAL_FEATURE_EXTRACTION.model_dir
LOCAL_FEATURE_RESULTS_DIR = LOCAL_FEATURE_EXTRACTION.results_dir

GLOBAL_FEATURE_MODEL_DIR = GLOBAL_FEATURE_EXTRACTION.model_dir
GLOBAL_FEATURE_RESULTS_DIR = GLOBAL_FEATURE_EXTRACTION.results_dir

FEATURE_FUSION_MODEL_DIR = FEATURE_FUSION.model_dir
FEATURE_FUSION_RESULTS_DIR = FEATURE_FUSION.results_dir


# --- Generic per-dataset path helpers (Step 2 preprocessing: EyePACS, APTOS2019, ...) ---
# Every dataset under datasets/ follows the same raw/ (read-only) -> processed/
# (preprocessing output) convention already scaffolded on disk. Rather than
# adding a fixed field per dataset, these resolve the pair generically by
# name, each overridable via a `<NAME>_RAW_DIR` / `<NAME>_PROCESSED_DIR`
# environment variable (e.g. EYEPACS_RAW_DIR, APTOS2019_PROCESSED_DIR).

def dataset_raw_dir(dataset_name):
    """Path to datasets/<dataset_name>/raw. Read-only -- never written to."""
    return os.environ.get(f'{dataset_name.upper()}_RAW_DIR') or os.path.join(
        _REPO_ROOT, 'datasets', dataset_name, 'raw'
    )


def dataset_processed_dir(dataset_name):
    """Path to datasets/<dataset_name>/processed -- where preprocessing outputs go."""
    return os.environ.get(f'{dataset_name.upper()}_PROCESSED_DIR') or os.path.join(
        _REPO_ROOT, 'datasets', dataset_name, 'processed'
    )


# --- Preprocessing configuration (Step 2: Gamma Correction + CLAHE) ---

def _env_float(name, default):
    """Parse an environment variable as float; `default` when unset/empty."""
    value = os.environ.get(name)
    return default if value is None or value == '' else float(value)


def _env_int(name, default):
    """Parse an environment variable as int; `default` when unset/empty."""
    value = os.environ.get(name)
    return default if value is None or value == '' else int(value)


def _env_bool(name, default):
    """Parse an environment variable as bool; `default` when unset/empty."""
    value = os.environ.get(name)
    if value is None or value == '':
        return default
    return value.strip().lower() in ('1', 'true', 'yes', 'on')


@dataclass(frozen=True)
class PreprocessingConfig:
    """Default Gamma Correction / CLAHE parameters used by
    image_preprocessing.py. Each field is read from its environment
    variable when set, otherwise falls back to the documented default
    below -- these are real, working values meant to be experimented with,
    not placeholders. image_preprocessing.py's own function arguments
    always take precedence over these: every function there defaults to
    `None`, which resolves to the matching field here only when the caller
    doesn't supply an explicit value."""
    DEFAULT_GAMMA: float
    DEFAULT_CLAHE_CLIP_LIMIT: float
    DEFAULT_CLAHE_TILE_GRID_SIZE: Tuple[int, int]


PREPROCESSING = PreprocessingConfig(
    DEFAULT_GAMMA=_env_float('DEFAULT_GAMMA', 1.2),
    DEFAULT_CLAHE_CLIP_LIMIT=_env_float('DEFAULT_CLAHE_CLIP_LIMIT', 2.0),
    DEFAULT_CLAHE_TILE_GRID_SIZE=(
        _env_int('DEFAULT_CLAHE_TILE_GRID_WIDTH', 8),
        _env_int('DEFAULT_CLAHE_TILE_GRID_HEIGHT', 8),
    ),
)


# --- Preprocessing profiles ---
# Named recipes consumed by image_preprocessing.py's `profile=` argument --
# different pipeline consumers need different preprocessing behavior from
# the same module, without each caller having to restate every parameter.

@dataclass(frozen=True)
class PreprocessingProfile:
    """A single named preprocessing recipe: which parameters to use, and
    whether CLAHE runs at all (`clahe_enabled`)."""
    gamma: float
    clahe_enabled: bool
    clip_limit: float
    tile_grid_size: Tuple[int, int]


@dataclass(frozen=True)
class PreprocessingProfiles:
    """The two preprocessing profiles currently defined:

    - IQA: no preprocessing (gamma=1.0 is an identity transform, CLAHE
      disabled) -- the Image Quality Assessment model (Step 1) is trained
      on, and must continue to see, the original unprocessed RGB EyeQ
      images.
    - DR: the full approved Step 2 pipeline (Gamma Correction + CLAHE),
      using PREPROCESSING's centralized defaults.
    """
    IQA: PreprocessingProfile
    DR: PreprocessingProfile


PREPROCESSING_PROFILES = PreprocessingProfiles(
    IQA=PreprocessingProfile(
        gamma=1.0,
        clahe_enabled=False,
        clip_limit=PREPROCESSING.DEFAULT_CLAHE_CLIP_LIMIT,
        tile_grid_size=PREPROCESSING.DEFAULT_CLAHE_TILE_GRID_SIZE,
    ),
    DR=PreprocessingProfile(
        gamma=PREPROCESSING.DEFAULT_GAMMA,
        clahe_enabled=True,
        clip_limit=PREPROCESSING.DEFAULT_CLAHE_CLIP_LIMIT,
        tile_grid_size=PREPROCESSING.DEFAULT_CLAHE_TILE_GRID_SIZE,
    ),
)


# --- Vessel Segmentation configuration (Step 3: pretrained LWNet inference) ---
# Only test-time augmentation is exposed here -- everything else about this
# stage (architecture, checkpoint, FOV-crop/resize/threshold procedure) is
# fixed by the vendored checkpoint itself (see vessel_segmentation_model.py /
# vessel_segmentation_inference.py), not a tunable knob.

# Default True: matches the LWNet authors' own default (`tta='from_preds'`
# in their predict_one_image.py) -- 4-view flip-averaged prediction. Set
# VESSEL_SEG_TTA=false to run a single forward pass per image instead
# (faster, e.g. for large-batch pipeline runs), without editing code.
VESSEL_SEG_TTA = _env_bool('VESSEL_SEG_TTA', True)
