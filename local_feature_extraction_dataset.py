"""
APTOS 2019 dataset loader for Local Feature Extraction (pipeline Stage 05).

Reads `datasets/APTOS2019/raw/train.csv` + `raw/train_images/` (never
modified -- `raw/` is read-only per PROJECT_CODE.md's Dataset Policy) and
builds `(input, label)` `tf.data.Dataset` pairs: an 8-channel,
segmentation-aware tensor per image, and its APTOS DR grade (0-4).

Per-sample input tensor -- `(H, W, 8)` float32, channel order fixed:
  - channels 0-2: Stage 02 processed RGB (Gamma Correction + CLAHE, via
    `image_preprocessing.preprocess_array`'s unmodified "DR" profile,
    applied live since `datasets/APTOS2019/processed/` has no precomputed
    output yet -- unlike `lesion_segmentation_dataset.py`, which reads
    IDRiD's already-processed files from disk), scaled to [0, 1].
  - channel 3: Stage 03 vessel probability map (frozen LWNet, via
    `vessel_segmentation_inference.predict_vessel_mask`, unmodified), [0, 1].
  - channels 4-7: Stage 04 four-class lesion probability maps (frozen
    Attention U-Net, Experiment 2C, via
    `lesion_segmentation_model.predict_lesion_mask`, unmodified), [0, 1],
    channel order == `lesion_segmentation_dataset.LESION_CLASSES`
    (Microaneurysm, Haemorrhage, HardExudate, SoftExudate).

All 8 channels are stacked at each image's own native resolution (Stage
03/04's inference functions already return their maps resized back to
match their input's native shape) and resized together, in a single
`skimage.transform.resize` call, to `image_size` -- never per-channel
independently -- so RGB/vessel/lesion content stays pixel-aligned, exactly
mirroring `lesion_segmentation_dataset.py`'s `_resize_input` convention.

Label: the `diagnosis` column of `train.csv`, 0-4, returned as a plain
int32 scalar -- not one-hot, not CORN's cumulative-target encoding (that
transform belongs to CORN's own module, not implemented here).

APTOS2019's `raw/test.csv` is never read by this module -- it ships with
no `diagnosis` column (a held-out Kaggle-competition split), so it cannot
support supervised training or evaluation in this project. `train.csv`'s
3662 labeled images are split into train/val via `split_train_val_ids`,
which now delegates to `downstream_split.get_authoritative_split` -- the
ONE authoritative, stratified, cross-environment split shared by every
downstream trainable stage (Stage 05, Stage 06, and eventually Stage 07 /
RACAF / CORN's joint training), not a per-stage interim choice. See
`downstream_split.py`'s module docstring for the full rationale; this
module is a consumer of that split, not its owner.

IDRiD is never read by this module in any capacity -- Stage 05 does not
train on it (only 81 total images, no DR-grade labels at all); its only
role relative to Stage 05 is already fully realized by the frozen Stage
04 checkpoint this module calls into. No IDRiD ground-truth mask is ever
read here, and none could be, since APTOS images have no such masks to
begin with.

Stage 03/04 inference is expensive (a full LWNet + Attention U-Net
forward pass per image) but deterministic per image, so each is computed
once and cached to disk as a `.npy` file under `cache_dir`, via
`_get_or_compute_stage3_stage4_maps` -- both maps are cached together
because Stage 04's own inference needs the vessel map at Stage 03's native
resolution. Stage 3/4's OWN inference (FOV detection, working resolution,
thresholds) is unmodified; only their already-computed output is resized
down to `image_size` (the canonical downstream resolution -- matching
Stage 04's own native working resolution, `DEFAULT_IMAGE_SIZE`) before
being written to disk, so the cache never stores a native-image-resolution
array (previously several MB-to-tens-of-MB per image at APTOS2019's
typical resolution; canonical resolution is a small, fixed size instead).
Caching never touches `datasets/*/raw`, and the cached, canonical-resolution
value is otherwise never altered once written.
"""

import csv
import logging
import os

import cv2
import numpy as np
import tensorflow as tf
from PIL import Image
from skimage.transform import resize as sk_resize

import config
import downstream_split
from image_preprocessing import preprocess_array
from lesion_segmentation_dataset import LESION_CLASSES
from lesion_segmentation_model import (
    DEFAULT_MODEL_PATH as DEFAULT_LESION_MODEL_PATH,
    load_lesion_model,
    predict_lesion_mask,
)
from vessel_segmentation_inference import (
    DEFAULT_MODEL_PATH as DEFAULT_VESSEL_MODEL_PATH,
    EmptyFieldOfViewError,
    _load_rgb_array,
    load_vessel_model,
    predict_vessel_mask,
)

logger = logging.getLogger(__name__)

NUM_CHANNELS = 8  # 3 RGB + 1 vessel probability + 4 lesion probabilities

# datasets/APTOS2019/raw -- reuses config.py's existing generic per-dataset
# helper unmodified, matching lesion_segmentation_dataset.py's identical use
# of dataset_raw_dir/dataset_processed_dir for IDRiD.
APTOS_RAW_DIR = config.dataset_raw_dir("APTOS2019")
DEFAULT_TRAIN_CSV = os.path.join(APTOS_RAW_DIR, "train.csv")
DEFAULT_TRAIN_IMAGE_DIR = os.path.join(APTOS_RAW_DIR, "train_images")

# datasets/APTOS2019/processed -- reuses config.py's existing generic helper.
# Empty in this project's local copy as of this module's original
# implementation (see module docstring), but a real Colab/Drive environment
# may already contain Stage 02's batch-processed output here; see
# _resolve_processed_rgb.
DEFAULT_PROCESSED_DIR = config.dataset_processed_dir("APTOS2019")

# Cached Stage 03/04 outputs -- derived, regenerable artifacts, not raw
# dataset content, so they live under this stage's own results directory
# (config.LOCAL_FEATURE_EXTRACTION.results_dir), matching
# lesion_segmentation_dataset.DEFAULT_VESSEL_CACHE_DIR's identical convention.
DEFAULT_CACHE_DIR = os.path.join(config.LOCAL_FEATURE_RESULTS_DIR, "stage03_stage04_cache")

# Not centralized in config.py -- per config.py's own module docstring,
# per-script training hyperparameters are deliberately left as plain
# function defaults, not environment-backed settings.
DEFAULT_IMAGE_SIZE = (512, 512)
DEFAULT_VAL_SPLIT = 0.2
DEFAULT_BATCH_SIZE = 4
DEFAULT_SEED = 42


# --- Filesystem discovery -- APTOS2019's own labeled train.csv ---

def _list_labeled_images(csv_path=DEFAULT_TRAIN_CSV):
    """Reads `csv_path` (APTOS2019's train.csv: `id_code`, `diagnosis`
    columns), returning a sorted list of `(id_code, diagnosis)` with
    `diagnosis` already cast to `int`. This is the only labeled APTOS split
    available in this project -- see this module's docstring for why
    `test.csv` cannot be used the same way."""
    if not os.path.exists(csv_path):
        raise FileNotFoundError(
            f"APTOS2019 labeled CSV not found at {csv_path}. Expected the raw APTOS2019 "
            "train.csv to already be present under datasets/APTOS2019/raw/ "
            "(PROJECT_CODE.md's Dataset Handling rule -- datasets are never auto-downloaded)."
        )
    entries = []
    with open(csv_path, newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            entries.append((row["id_code"], int(row["diagnosis"])))
    if not entries:
        raise FileNotFoundError(f"No labeled rows found in {csv_path}.")
    return sorted(entries)


def split_train_val_ids(csv_path=DEFAULT_TRAIN_CSV, val_split=DEFAULT_VAL_SPLIT, seed=DEFAULT_SEED):
    """Thin delegation to `downstream_split.get_authoritative_split` -- kept
    here, under this same name and signature, so every existing caller
    (`global_feature_extraction_dataset.py`'s re-export, this module's own
    `load_local_feature_extraction_datasets`, and any test) keeps working
    unchanged. This is no longer where the split is actually computed or
    owned -- see `downstream_split.py`'s module docstring for why that
    module, not this one, is now the single source of truth (Stage 05 was
    the first stage to need this split, not its conceptual owner). For the
    default `csv_path`/`val_split`/`seed`, this now returns the committed,
    stratified, cross-environment authoritative manifest rather than a
    freshly (and non-stratified) computed interim split."""
    return downstream_split.get_authoritative_split(csv_path, val_split=val_split, seed=seed)


# --- Per-sample loading ---

def _load_raw_bgr(image_dir, id_code):
    """Loads one raw APTOS image via `cv2.imread` (BGR channel order --
    the convention `image_preprocessing.py`'s CLAHE step requires, since it
    calls `cv2.cvtColor(image, cv2.COLOR_BGR2LAB)` internally)."""
    path = os.path.join(image_dir, f"{id_code}.png")
    if not os.path.exists(path):
        raise FileNotFoundError(f"APTOS2019 image not found at {path}.")
    image = cv2.imread(path)
    if image is None:
        raise ValueError(f"Failed to load image: {path}")
    return image


def _stage02_processed_rgb(raw_bgr_image):
    """Applies Stage 02's frozen Gamma Correction + CLAHE pipeline
    (`image_preprocessing.preprocess_array`, unmodified, `profile="DR"` --
    the same profile every other DR-facing stage in this project uses) to
    one raw BGR image, then converts the result to true RGB channel order.

    The conversion matters: `cv2.imwrite`-based pipelines elsewhere in this
    project (e.g. IDRiD's already-processed files, read back via
    `lesion_segmentation_dataset._load_rgb_image`'s
    `Image.open(path).convert("RGB")`) round-trip through a real image file,
    which silently fixes channel order for the reader. Applying Stage 02
    in-memory here has no such round trip, so the BGR-to-RGB conversion is
    made explicit instead -- the result is numerically identical to what a
    save-then-reload of the same array would produce. Every downstream
    consumer of this array (Stage 03/04's inference functions, and this
    module's own channel 0-2 output) expects true RGB, matching
    `vessel_segmentation_inference._load_rgb_array`'s documented contract.
    """
    processed_bgr = preprocess_array(raw_bgr_image, profile="DR")
    return cv2.cvtColor(processed_bgr, cv2.COLOR_BGR2RGB)


def _cache_path(cache_dir, id_code, kind, image_size):
    """Cache filename includes `image_size` so a cache built for one
    canonical resolution can never be silently reused for another --
    load-bearing since (unlike before this cache stored the resize-down
    result, not the native-resolution prediction) the cached bytes
    themselves now depend on `image_size` (see `_get_or_compute_stage3_stage4_maps`)."""
    height, width = image_size
    return os.path.join(cache_dir, f"APTOS_{id_code}_{kind}_{height}x{width}.npy")


def _resolve_processed_rgb(raw_bgr_image, processed_dir, id_code):
    """Reuses an already-generated Stage 02 processed output for `id_code`
    if `processed_dir` (default `DEFAULT_PROCESSED_DIR`) already contains
    one -- the same "read the existing file, never regenerate it"
    convention `lesion_segmentation_dataset.py` already relies on for IDRiD
    (PROJECT_CODE.md's Stage 02 Preprocessing Policy: "No downstream stage
    should regenerate deterministic preprocessing outputs"). Falls back to
    live in-memory Stage 02 application (`_stage02_processed_rgb`) only
    when no precomputed file exists for this `id_code` -- true for every
    environment as of this module's original implementation
    (`datasets/APTOS2019/processed/` ships empty locally), but not
    necessarily true in every environment this code runs in (e.g. a Google
    Drive copy where Stage 02's batch preprocessing has already run) -- see
    this module's docstring. Matches the raw image's own filename exactly
    (`preprocess_folder()` preserves filenames, per `image_preprocessing.py`),
    read the same way `lesion_segmentation_dataset._load_rgb_image` already
    reads IDRiD's processed files."""
    processed_path = os.path.join(processed_dir, f"{id_code}.png")
    if os.path.exists(processed_path):
        return np.array(Image.open(processed_path).convert("RGB"), dtype=np.uint8)
    return _stage02_processed_rgb(raw_bgr_image)


def _resize_map(map_array, image_size):
    """Resizes one Stage 03/04 probability map (`(H, W)` or `(H, W, C)`,
    native image resolution) down to `image_size` -- the canonical
    downstream resolution Stage 05's tensor and RACAF's own TTA input both
    actually need (`DEFAULT_IMAGE_SIZE`, matching Stage 04's own native
    working resolution) -- BEFORE it is written to the on-disk cache. Same
    resize convention as `_resize_input` (order=1, reflect, anti-aliasing,
    clipped to [0,1]), applied to a single map rather than the full 8-channel
    stack. Introduced to fix the pre-existing cache design storing
    native-image-resolution arrays (up to tens of MB per image at APTOS2019's
    typical resolution) instead of the small, fixed-size arrays every
    consumer actually reads."""
    output_shape = (*image_size, map_array.shape[-1]) if map_array.ndim == 3 else image_size
    resized = sk_resize(
        map_array, output_shape, order=1, mode="reflect", anti_aliasing=True, preserve_range=True,
    )
    return np.clip(resized, 0.0, 1.0).astype(np.float32)


def _resize_rgb(rgb_uint8, image_size):
    """Resizes a native-resolution uint8 RGB image down to `image_size`,
    preserving its [0, 255] uint8 encoding (never normalizing here --
    `build_local_feature_input`'s own `/255.0` normalization still owns
    that step, unchanged). Exists only so the RGB passed into
    `build_local_feature_input` already matches the canonical resolution
    Stage 03/04's caches are now stored at (`_get_or_compute_stage3_stage4_maps`),
    satisfying that function's existing, UNMODIFIED native-shape-match
    validation without altering it at all."""
    resized = sk_resize(
        rgb_uint8, (*image_size, 3), order=1, mode="reflect", anti_aliasing=True, preserve_range=True,
    )
    return np.clip(resized, 0, 255).astype(np.uint8)


def _get_or_compute_stage3_stage4_maps(rgb_image, vessel_cache_path, lesion_cache_path,
                                        vessel_model, lesion_model, image_size):
    """Returns `(vessel_map, lesion_maps)`, both resized to the canonical
    `image_size` resolution -- `vessel_map` `(*image_size,)` float32,
    `lesion_maps` `(*image_size, 4)` float32 -- populating (or reusing) both
    on-disk caches together.

    Stage 03/04's OWN inference is completely unchanged: `predict_vessel_mask`/
    `predict_lesion_mask` still run on the full, native-resolution
    `rgb_image` exactly as before (same FOV-detection, same working
    resolution, same threshold conventions) -- only their ALREADY-COMPUTED
    output is resized down, once, before being written to disk. This is a
    downstream caching optimization only; Stage 3/4's official inference
    behavior is not altered.

    The two caches are populated together (not by two independent
    functions, as before) because Stage 04's own inference inherently needs
    the vessel map at the SAME NATIVE resolution as `rgb_image` for its
    internal RGB+vessel concatenation (`predict_lesion_mask`'s existing,
    unmodified contract) -- the small, canonical-resolution vessel map
    alone is not enough to reconstruct that input. So: if both caches
    already exist, neither Stage 3 nor Stage 4 runs at all (the common
    case, every epoch after the first). Otherwise Stage 3's native-resolution
    vessel map is (re)computed in memory -- and, if the vessel cache itself
    was already populated but the lesion cache was not (a rare, partial-cache
    edge case), this recomputes Stage 3 once more rather than persisting a
    second, native-resolution copy purely to avoid it. The native-resolution
    vessel map itself is never written to disk -- only its canonical,
    resized-down copy is, exactly like the lesion map.
    """
    vessel_cached = os.path.exists(vessel_cache_path)
    lesion_cached = os.path.exists(lesion_cache_path)

    if vessel_cached and lesion_cached:
        return np.load(vessel_cache_path), np.load(lesion_cache_path)

    native_vessel_map = predict_vessel_mask(rgb_image, model=vessel_model)["probability_map"].astype(np.float32)

    if vessel_cached:
        canonical_vessel_map = np.load(vessel_cache_path)
    else:
        canonical_vessel_map = _resize_map(native_vessel_map, image_size)
        os.makedirs(os.path.dirname(vessel_cache_path), exist_ok=True)
        np.save(vessel_cache_path, canonical_vessel_map)

    if lesion_cached:
        lesion_maps = np.load(lesion_cache_path)
    else:
        result = predict_lesion_mask(
            rgb_image, vessel_probability_map=native_vessel_map, model=lesion_model, vessel_model=vessel_model,
        )
        lesion_maps = _resize_map(result["probability_maps"].astype(np.float32), image_size)
        os.makedirs(os.path.dirname(lesion_cache_path), exist_ok=True)
        np.save(lesion_cache_path, lesion_maps)

    return canonical_vessel_map, lesion_maps


def _resize_input(input_array, image_size):
    """Resizes the full, already-concatenated 8-channel tensor in one call
    -- never per-channel independently -- so RGB/vessel/lesion content stays
    pixel-aligned. No aspect-ratio-preserving padding (direct resize is the
    approved convention). Identical mechanism to
    `lesion_segmentation_dataset._resize_input`."""
    resized = sk_resize(
        input_array, (*image_size, input_array.shape[-1]),
        order=1, mode="reflect", anti_aliasing=True, preserve_range=True,
    )
    return np.clip(resized, 0.0, 1.0).astype(np.float32)


def build_local_feature_input(image, vessel_probability_map=None, lesion_probability_maps=None,
                               vessel_model=None, lesion_model=None,
                               vessel_model_path=DEFAULT_VESSEL_MODEL_PATH,
                               lesion_model_path=DEFAULT_LESION_MODEL_PATH,
                               image_size=DEFAULT_IMAGE_SIZE):
    """
    Builds one `(*image_size, 8)` float32 Stage 05 input tensor for a
    single **already Stage-02-processed** RGB image (a file path, PIL
    Image, or `(H, W, 3)` uint8 RGB array -- see
    `vessel_segmentation_inference._load_rgb_array`). Unlike
    `load_local_feature_extraction_datasets`, this does not apply Stage 02
    preprocessing itself and does not read/return a label -- it exists as a
    single-image entry point for inference or a sanity-check forward pass
    (mirroring `lesion_segmentation_model.predict_lesion_mask`'s identical
    "caller supplies an already-processed image" contract), not for
    building the APTOS training corpus.

    Passing a precomputed `vessel_probability_map`/`lesion_probability_maps`
    skips the corresponding frozen Stage 03/04 forward pass entirely
    (matching `predict_lesion_mask`'s own `vessel_probability_map=`
    parameter) -- useful for tests and for reusing already-cached maps.
    Otherwise each is computed here via the unmodified Stage 03/04
    inference functions.

    Returns the resized `(*image_size, 8)` float32 array; channel order is
    fixed as channels 0-2 = RGB ([0, 1]), channel 3 = vessel probability
    ([0, 1]), channels 4-7 = lesion probabilities in `LESION_CLASSES` order
    ([0, 1]).
    """
    rgb = _load_rgb_array(image)
    native_shape = rgb.shape[:2]

    if vessel_probability_map is None:
        resolved_vessel_model = vessel_model if vessel_model is not None else load_vessel_model(vessel_model_path)
        vessel_probability_map = predict_vessel_mask(rgb, model=resolved_vessel_model)["probability_map"]
    vessel_probability_map = np.asarray(vessel_probability_map, dtype=np.float32)
    if vessel_probability_map.ndim == 2:
        vessel_probability_map = vessel_probability_map[..., np.newaxis]
    if vessel_probability_map.shape[:2] != native_shape:
        raise RuntimeError(
            f"Vessel probability map shape {vessel_probability_map.shape[:2]} does not "
            f"match input image shape {native_shape}."
        )

    if lesion_probability_maps is None:
        resolved_lesion_model = lesion_model if lesion_model is not None else load_lesion_model(lesion_model_path)
        lesion_probability_maps = predict_lesion_mask(
            rgb, vessel_probability_map=vessel_probability_map,
            model=resolved_lesion_model, vessel_model=vessel_model,
        )["probability_maps"]
    lesion_probability_maps = np.asarray(lesion_probability_maps, dtype=np.float32)
    if lesion_probability_maps.shape[:2] != native_shape:
        raise RuntimeError(
            f"Lesion probability maps shape {lesion_probability_maps.shape[:2]} does not "
            f"match input image shape {native_shape}."
        )
    if lesion_probability_maps.shape[-1] != len(LESION_CLASSES):
        raise ValueError(
            f"Expected {len(LESION_CLASSES)} lesion probability channels "
            f"({LESION_CLASSES}), got {lesion_probability_maps.shape[-1]}."
        )

    input_array = np.concatenate(
        [rgb.astype(np.float32) / 255.0, vessel_probability_map, lesion_probability_maps], axis=-1,
    )
    return _resize_input(input_array, image_size)


def _build_sample(id_code, diagnosis, image_dir, cache_dir, vessel_model, lesion_model, image_size,
                   processed_dir=DEFAULT_PROCESSED_DIR):
    """Builds one `(input, label)` pair for one APTOS training-set image:
    Stage 02 output resolved via `_resolve_processed_rgb` (an existing
    `processed_dir` file if present, live application otherwise), Stage
    03/04 outputs resolved via the disk cache (computed once per id_code,
    reused on every subsequent call/epoch), concatenated and resized via
    `build_local_feature_input`. `input` is `(*image_size, 8)` float32;
    `label` is the plain `int` APTOS DR grade (0-4). `processed_dir`
    defaults to `DEFAULT_PROCESSED_DIR` so every existing positional caller
    (this module's own `load_local_feature_extraction_datasets`, and
    `colab/notebooks/stage05_local_feature_extraction.ipynb`'s direct call)
    keeps working unchanged."""
    raw_bgr = _load_raw_bgr(image_dir, id_code)
    rgb_native = _resolve_processed_rgb(raw_bgr, processed_dir, id_code)

    vessel_cache = _cache_path(cache_dir, id_code, "vessel", image_size)
    lesion_cache = _cache_path(cache_dir, id_code, "lesion", image_size)
    vessel_map, lesion_maps = _get_or_compute_stage3_stage4_maps(
        rgb_native, vessel_cache, lesion_cache, vessel_model, lesion_model, image_size,
    )

    # Stage 3/4 ran on the full native-resolution rgb_native above (their
    # own inference is unchanged); only their cached output is canonical
    # resolution. Resize the RGB itself down to the same resolution here so
    # build_local_feature_input's existing native-shape-match validation
    # (unmodified) is satisfied without altering that function at all.
    rgb = _resize_rgb(rgb_native, image_size)

    input_array = build_local_feature_input(
        rgb, vessel_probability_map=vessel_map, lesion_probability_maps=lesion_maps, image_size=image_size,
    )
    return input_array, diagnosis


def _augment_spatial(input_array, rng):
    """Random horizontal flip, vertical flip, and 90-degree rotation,
    applied identically across all 8 channels -- so RGB/vessel/lesion
    spatial correspondence is always preserved. Same mechanism as
    `lesion_segmentation_dataset._augment`'s spatial half, applied here to
    a single 8-channel input tensor (Stage 05 has no per-pixel target
    array of its own to keep synchronized against)."""
    if rng.random() < 0.5:
        input_array = input_array[:, ::-1, :]
    if rng.random() < 0.5:
        input_array = input_array[::-1, :, :]
    k = int(rng.integers(0, 4))
    if k:
        input_array = np.rot90(input_array, k=k, axes=(0, 1))
    return np.ascontiguousarray(input_array)


def _augment_intensity_rgb(input_array, rng, brightness_range=0.1, contrast_range=0.1):
    """Mild brightness/contrast jitter applied ONLY to channels 0-2 (RGB).
    Channels 3-7 (vessel/lesion probabilities) are returned completely
    unmodified -- jittering an already-computed probability value would
    corrupt its meaning as a calibrated-ish confidence, per the approved
    design's explicit augmentation rule."""
    output = np.array(input_array, copy=True)
    rgb = output[..., :3]
    brightness = 1.0 + rng.uniform(-brightness_range, brightness_range)
    contrast = 1.0 + rng.uniform(-contrast_range, contrast_range)
    mean = rgb.mean()
    rgb = (rgb - mean) * contrast + mean
    rgb = rgb * brightness
    output[..., :3] = np.clip(rgb, 0.0, 1.0)
    return output


def _augment(input_array, rng):
    input_array = _augment_spatial(input_array, rng)
    input_array = _augment_intensity_rgb(input_array, rng)
    return input_array


def _resolve_vessel_model(vessel_model, vessel_model_path):
    if vessel_model is not None:
        return vessel_model
    return load_vessel_model(vessel_model_path)


def _resolve_lesion_model(lesion_model, lesion_model_path):
    if lesion_model is not None:
        return lesion_model
    return load_lesion_model(lesion_model_path)


def _make_dataset(entries, image_dir, cache_dir, vessel_model, lesion_model,
                   image_size, batch_size, shuffle, augment, seed, processed_dir=DEFAULT_PROCESSED_DIR):
    entries = list(entries)

    def gen():
        rng = np.random.default_rng(seed) if augment else None
        for id_code, diagnosis in entries:
            try:
                x, y = _build_sample(
                    id_code, diagnosis, image_dir, cache_dir, vessel_model, lesion_model, image_size,
                    processed_dir=processed_dir,
                )
            except EmptyFieldOfViewError:
                # Same real, rare condition documented in EmptyFieldOfViewError: Stage 03's FOV
                # circle-fit found no fundus disk for this image. No project-sanctioned fallback
                # exists, so the image is skipped/quarantined for this epoch, not fabricated or
                # allowed to crash the whole run. The authoritative split manifest is unchanged.
                logger.warning(
                    "Skipping image_id=%s: empty Stage 03 field-of-view (no fundus disk "
                    "detected). This image cannot be processed by the frozen vessel "
                    "segmentation FOV-crop step and is excluded from this epoch.",
                    id_code,
                )
                continue
            if augment:
                x = _augment(x, rng)
            yield x, y

    output_signature = (
        tf.TensorSpec(shape=(*image_size, NUM_CHANNELS), dtype=tf.float32),
        tf.TensorSpec(shape=(), dtype=tf.int32),
    )
    ds = tf.data.Dataset.from_generator(gen, output_signature=output_signature)
    if shuffle:
        ds = ds.shuffle(buffer_size=max(len(entries), 1), seed=seed, reshuffle_each_iteration=True)
    return ds.batch(batch_size).prefetch(tf.data.AUTOTUNE)


# --- Public API ---

def load_local_feature_extraction_datasets(
    csv_path=DEFAULT_TRAIN_CSV,
    image_dir=DEFAULT_TRAIN_IMAGE_DIR,
    cache_dir=DEFAULT_CACHE_DIR,
    vessel_model=None,
    vessel_model_path=DEFAULT_VESSEL_MODEL_PATH,
    lesion_model=None,
    lesion_model_path=DEFAULT_LESION_MODEL_PATH,
    image_size=DEFAULT_IMAGE_SIZE,
    val_split=DEFAULT_VAL_SPLIT,
    batch_size=DEFAULT_BATCH_SIZE,
    seed=DEFAULT_SEED,
    augment_train=True,
    processed_dir=DEFAULT_PROCESSED_DIR,
):
    """
    Train/val `tf.data.Dataset` pipelines built from APTOS2019's labeled
    `train.csv` (3662 images), split per `split_train_val_ids` -- for the
    default `csv_path`/`val_split`/`seed`, this is now the authoritative,
    stratified, cross-environment split (`downstream_split.py`), not an
    interim per-stage one. See this module's docstring for why `test.csv`
    is never used.

    Pass already-loaded `vessel_model`/`lesion_model` to reuse them across
    many calls/datasets; otherwise each is loaded once (not per-sample, not
    per-epoch) from `vessel_model_path`/`lesion_model_path`. `processed_dir`
    is checked for an existing Stage 02 output per image before falling
    back to live preprocessing (`_resolve_processed_rgb`).

    Returns `(train_ds, val_ds)`, each yielding batches of `((batch, H, W,
    8) float32 input, (batch,) int32 label)`.
    """
    train_entries, val_entries = split_train_val_ids(csv_path, val_split=val_split, seed=seed)

    resolved_vessel_model = _resolve_vessel_model(vessel_model, vessel_model_path)
    resolved_lesion_model = _resolve_lesion_model(lesion_model, lesion_model_path)

    train_ds = _make_dataset(
        train_entries, image_dir, cache_dir, resolved_vessel_model, resolved_lesion_model,
        image_size, batch_size, shuffle=True, augment=augment_train, seed=seed, processed_dir=processed_dir,
    )
    val_ds = _make_dataset(
        val_entries, image_dir, cache_dir, resolved_vessel_model, resolved_lesion_model,
        image_size, batch_size, shuffle=False, augment=False, seed=seed, processed_dir=processed_dir,
    )
    return train_ds, val_ds
