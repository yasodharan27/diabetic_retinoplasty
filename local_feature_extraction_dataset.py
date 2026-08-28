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
support supervised training or evaluation in this project. Only
`train.csv`'s 3662 labeled images are split (deterministically,
`train_test_split`) into train/val here, mirroring the exact mechanism
`lesion_segmentation_dataset.split_train_val_ids` /
`image_quality_dataset.load_eyeq_datasets` already use. This is a
reasonable interim split for exercising this module -- it is NOT yet the
authoritative Stage 08 joint-training split (no target-architecture APTOS
dataset loader existed anywhere in this project before this module), so
Stage 05 must adopt whatever split Stage 08's own loader eventually fixes
once written, since Stages 05-08 + RACAF are documented to train in one
shared graph (`RACAF_ARCHITECTURE.md` Sec 7).

IDRiD is never read by this module in any capacity -- Stage 05 does not
train on it (only 81 total images, no DR-grade labels at all); its only
role relative to Stage 05 is already fully realized by the frozen Stage
04 checkpoint this module calls into. No IDRiD ground-truth mask is ever
read here, and none could be, since APTOS images have no such masks to
begin with.

Stage 03/04 inference is expensive (a full LWNet + Attention U-Net
forward pass per image) but deterministic per image, so each is computed
once and cached to disk as a `.npy` file under `cache_dir` -- mirroring
`lesion_segmentation_dataset.py`'s `_get_or_compute_vessel_map` pattern
exactly, extended here to also cache Stage 04's lesion probability maps.
Caching only ever stores an already-computed derived array unchanged; it
never alters a numerical value, and never touches `datasets/*/raw`.
"""

import csv
import os

import cv2
import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
from skimage.transform import resize as sk_resize

import config
from image_preprocessing import preprocess_array
from lesion_segmentation_dataset import LESION_CLASSES
from lesion_segmentation_model import (
    DEFAULT_MODEL_PATH as DEFAULT_LESION_MODEL_PATH,
    load_lesion_model,
    predict_lesion_mask,
)
from vessel_segmentation_inference import (
    DEFAULT_MODEL_PATH as DEFAULT_VESSEL_MODEL_PATH,
    _load_rgb_array,
    load_vessel_model,
    predict_vessel_mask,
)

NUM_CHANNELS = 8  # 3 RGB + 1 vessel probability + 4 lesion probabilities

# datasets/APTOS2019/raw -- reuses config.py's existing generic per-dataset
# helper unmodified, matching lesion_segmentation_dataset.py's identical use
# of dataset_raw_dir/dataset_processed_dir for IDRiD.
APTOS_RAW_DIR = config.dataset_raw_dir("APTOS2019")
DEFAULT_TRAIN_CSV = os.path.join(APTOS_RAW_DIR, "train.csv")
DEFAULT_TRAIN_IMAGE_DIR = os.path.join(APTOS_RAW_DIR, "train_images")

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
    """Deterministic train/val split of APTOS2019's labeled `train.csv`
    (same `train_test_split` mechanism `lesion_segmentation_dataset.
    split_train_val_ids` / `image_quality_dataset.load_eyeq_datasets`
    already use, with a fixed `seed` so the split is reproducible run to
    run). See this module's docstring for why this is an interim split,
    not yet the authoritative Stage 08 joint-training split."""
    entries = _list_labeled_images(csv_path)
    train_entries, val_entries = train_test_split(entries, test_size=val_split, random_state=seed)
    return sorted(train_entries), sorted(val_entries)


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


def _cache_path(cache_dir, id_code, kind):
    return os.path.join(cache_dir, f"APTOS_{id_code}_{kind}.npy")


def _get_or_compute_vessel_map(rgb_image, cache_path, vessel_model):
    """Returns the `(H, W, 1)` float32 Stage 03 vessel probability map for
    `rgb_image`, from `cache_path` if already computed, otherwise via
    Stage 03's unmodified `predict_vessel_mask()` -- cached to disk
    afterward. Mirrors `lesion_segmentation_dataset._get_or_compute_vessel_map`
    exactly (same cache-then-reuse behavior, same never-recompute-if-cached
    guarantee), just keyed by an APTOS id instead of an IDRiD one."""
    if os.path.exists(cache_path):
        return np.load(cache_path)
    result = predict_vessel_mask(rgb_image, model=vessel_model)
    vessel_map = result["probability_map"].astype(np.float32)
    os.makedirs(os.path.dirname(cache_path), exist_ok=True)
    np.save(cache_path, vessel_map)
    return vessel_map


def _get_or_compute_lesion_maps(rgb_image, vessel_map, cache_path, lesion_model, vessel_model):
    """Returns the `(H, W, 4)` float32 Stage 04 lesion probability maps for
    `rgb_image`, from `cache_path` if already computed, otherwise via Stage
    04's unmodified `predict_lesion_mask()` (frozen Experiment 2C) -- cached
    to disk afterward. `vessel_map`, if given, is passed straight through as
    `predict_lesion_mask`'s own `vessel_probability_map=`, so Stage 03 is
    never re-run here when a vessel map was already computed/cached by
    `_get_or_compute_vessel_map`."""
    if os.path.exists(cache_path):
        return np.load(cache_path)
    result = predict_lesion_mask(
        rgb_image, vessel_probability_map=vessel_map, model=lesion_model, vessel_model=vessel_model,
    )
    lesion_maps = result["probability_maps"].astype(np.float32)
    os.makedirs(os.path.dirname(cache_path), exist_ok=True)
    np.save(cache_path, lesion_maps)
    return lesion_maps


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


def _build_sample(id_code, diagnosis, image_dir, cache_dir, vessel_model, lesion_model, image_size):
    """Builds one `(input, label)` pair for one APTOS training-set image:
    Stage 02 preprocessing applied live, Stage 03/04 outputs resolved via
    the disk cache (computed once per id_code, reused on every subsequent
    call/epoch), concatenated and resized via `build_local_feature_input`.
    `input` is `(*image_size, 8)` float32; `label` is the plain `int`
    APTOS DR grade (0-4)."""
    raw_bgr = _load_raw_bgr(image_dir, id_code)
    rgb = _stage02_processed_rgb(raw_bgr)

    vessel_cache = _cache_path(cache_dir, id_code, "vessel")
    vessel_map = _get_or_compute_vessel_map(rgb, vessel_cache, vessel_model)

    lesion_cache = _cache_path(cache_dir, id_code, "lesion")
    lesion_maps = _get_or_compute_lesion_maps(rgb, vessel_map, lesion_cache, lesion_model, vessel_model)

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
                   image_size, batch_size, shuffle, augment, seed):
    entries = list(entries)

    def gen():
        rng = np.random.default_rng(seed) if augment else None
        for id_code, diagnosis in entries:
            x, y = _build_sample(id_code, diagnosis, image_dir, cache_dir, vessel_model, lesion_model, image_size)
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
):
    """
    Train/val `tf.data.Dataset` pipelines built from APTOS2019's labeled
    `train.csv` (3662 images), split deterministically per
    `split_train_val_ids`. See this module's docstring for why `test.csv`
    is never used, and why this split is an interim choice, not yet the
    authoritative Stage 08 joint-training split.

    Pass already-loaded `vessel_model`/`lesion_model` to reuse them across
    many calls/datasets; otherwise each is loaded once (not per-sample, not
    per-epoch) from `vessel_model_path`/`lesion_model_path`.

    Returns `(train_ds, val_ds)`, each yielding batches of `((batch, H, W,
    8) float32 input, (batch,) int32 label)`.
    """
    train_entries, val_entries = split_train_val_ids(csv_path, val_split=val_split, seed=seed)

    resolved_vessel_model = _resolve_vessel_model(vessel_model, vessel_model_path)
    resolved_lesion_model = _resolve_lesion_model(lesion_model, lesion_model_path)

    train_ds = _make_dataset(
        train_entries, image_dir, cache_dir, resolved_vessel_model, resolved_lesion_model,
        image_size, batch_size, shuffle=True, augment=augment_train, seed=seed,
    )
    val_ds = _make_dataset(
        val_entries, image_dir, cache_dir, resolved_vessel_model, resolved_lesion_model,
        image_size, batch_size, shuffle=False, augment=False, seed=seed,
    )
    return train_ds, val_ds
