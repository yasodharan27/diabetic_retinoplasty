"""
IDRiD segmentation-subset dataset loader for Lesion Segmentation (pipeline
Stage 04).

Reads `datasets/IDRiD/segmentation/raw/` (the fixed IDRiD 54-image Training
Set / 27-image Testing Set split; ground-truth masks; never modified here)
and the corresponding already-generated Stage 02 output under
`datasets/IDRiD/segmentation/processed/` (read-only, never regenerated --
per PROJECT_CODE.md's Stage 02 Preprocessing Policy, "no downstream stage
should regenerate deterministic preprocessing outputs"). Builds `(image,
mask)` `tf.data.Dataset` pairs -- the first dataset loader in this repo with
that shape, rather than `(image, label)` (SEGMENTATION_ARCHITECTURE.md
Sec 5).

Per-sample tensors:
  - input:  (H, W, 4) float32 in [0, 1] -- Stage 02 processed RGB (3
            channels, scaled to [0, 1]) concatenated with Stage 03's vessel
            probability map (1 channel), via vessel_segmentation_inference's
            unmodified `predict_vessel_mask()` (SEGMENTATION_ARCHITECTURE.md
            Sec 3.1).
  - target: (H, W, 4) float32 in {0, 1} -- one binary channel per lesion
            class, in this fixed order: Microaneurysm, Haemorrhage, Hard
            Exudate, Soft Exudate (Sec 3.3). Optic Disc is never read or
            included (Sec 3.4/Appendix A.4) -- it is not one of this
            stage's output classes.

Vessel probability maps are expensive (a full LWNet forward pass) but
deterministic per image, so each one is computed once and cached to disk
as a `.npy` file under `vessel_cache_dir` -- every subsequent epoch (and
every subsequent process run, e.g. resuming training) reads the cached
array instead of re-running inference.

Missing Soft Exudate ground truth (IDRiD ships no mask file for images that
simply contain no soft exudates -- verified: 28 of the 54 training images)
is treated as an all-zero channel for that image, not an error -- this is
IDRiD's own convention for "this lesion class is absent here", not missing
data.

Does not require the real dataset to import or unit-test: every path is a
function parameter with a locally-resolved default (mirroring
`image_quality_dataset.py`'s `raw_dir=EYEQ_RAW_DIR` pattern), so a caller
(e.g. the Stage 04 Colab notebook, or a test using a small temporary
directory) can point this module at any IDRiD-segmentation-shaped folder
tree without editing it. Processed IDRiD images live on Google Drive during
real Colab runs, not in this local repository -- see
`colab/notebooks/stage04_lesion_segmentation.ipynb` for how they get staged
locally there.
"""

import os

import numpy as np
import tensorflow as tf
from PIL import Image
from sklearn.model_selection import train_test_split
from skimage.transform import resize as sk_resize

import config
from vessel_segmentation_inference import (
    DEFAULT_MODEL_PATH as DEFAULT_VESSEL_MODEL_PATH,
    load_vessel_model,
    predict_vessel_mask,
)

LESION_CLASSES = ("Microaneurysm", "Haemorrhage", "HardExudate", "SoftExudate")

# (class name, IDRiD groundtruth subfolder, IDRiD mask filename suffix).
# Order fixes the target tensor's channel order -- must match LESION_CLASSES.
# "5. Optic Disc" / "OD" is deliberately absent: SEGMENTATION_ARCHITECTURE.md
# Sec 3.4/Appendix A.4 explicitly excludes Optic Disc from this stage's output.
_MASK_SPECS = (
    ("Microaneurysm", "1. Microaneurysms", "MA"),
    ("Haemorrhage", "2. Haemorrhages", "HE"),
    ("HardExudate", "3. Hard Exudates", "EX"),
    ("SoftExudate", "4. Soft Exudates", "SE"),
)

_ORIGINAL_IMAGES_SUBDIR = "1. Original Images"
_GROUNDTRUTHS_SUBDIR = "2. All Segmentation Groundtruths"
TRAIN_SPLIT_DIR = "a. Training Set"
TEST_SPLIT_DIR = "b. Testing Set"

_IMAGE_EXTENSIONS = (".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff")

# datasets/IDRiD/segmentation/{raw,processed} -- reuses config.py's existing
# generic per-dataset helpers unmodified (dataset_raw_dir/dataset_processed_dir
# already join `datasets/<name>/{raw,processed}`; passing "IDRiD/segmentation"
# resolves the exact nested path this subtask lives at without needing a new
# `subtask=` parameter -- see SEGMENTATION_ARCHITECTURE.md Sec 6/Appendix A.5
# for why that parameter doesn't exist yet). Callers needing a different
# location (e.g. the Colab notebook, pointing at a Drive-staged local SSD
# copy) pass raw_dir=/processed_dir= explicitly instead of relying on an
# environment-variable override here.
IDRID_SEGMENTATION_RAW_DIR = config.dataset_raw_dir("IDRiD/segmentation")
IDRID_SEGMENTATION_PROCESSED_DIR = config.dataset_processed_dir("IDRiD/segmentation")

# Cached Stage 03 vessel probability maps -- a derived, regenerable
# artifact (not raw data, not a trained model), so it lives under this
# stage's own results directory rather than under datasets/ (which
# PROJECT_CODE.md's Dataset Policy reserves for raw/processed dataset
# content only).
DEFAULT_VESSEL_CACHE_DIR = os.path.join(config.LESION_SEG_RESULTS_DIR, "vessel_cache")

# Not centralized in config.py -- per config.py's own module docstring,
# per-script training hyperparameters (image size, batch size, val split)
# are deliberately left as plain function defaults, not environment-backed
# settings, so each caller can override them without a config.py edit.
DEFAULT_IMAGE_SIZE = (512, 512)
DEFAULT_VAL_SPLIT = 0.2
DEFAULT_BATCH_SIZE = 4
DEFAULT_SEED = 42


# --- Filesystem discovery -- IDRiD's own fixed train/test split ---

def _list_original_images(raw_dir, split_dir):
    """Discovers every original fundus image directly under
    `raw_dir/1. Original Images/<split_dir>/`, returning a sorted list of
    (image_id, filename) -- e.g. ("01", "IDRiD_01.jpg"). Filenames are read
    from disk, not assumed, so this only depends on IDRiD's own
    `IDRiD_<id>.<ext>` naming, not a hardcoded extension. `image_id` is
    whatever follows the last "_" before the extension, preserving its
    original zero-padding so mask filenames (built from the same id) match
    exactly."""
    folder = os.path.join(raw_dir, _ORIGINAL_IMAGES_SUBDIR, split_dir)
    if not os.path.isdir(folder):
        raise FileNotFoundError(
            f"IDRiD segmentation split not found: {folder}. Expected the "
            "IDRiD segmentation subset's raw images to already be present "
            "under datasets/IDRiD/segmentation/raw/ (PROJECT_CODE.md's "
            "Dataset Handling rule -- datasets are never auto-downloaded)."
        )
    entries = []
    for filename in sorted(os.listdir(folder)):
        if not filename.lower().endswith(_IMAGE_EXTENSIONS):
            continue
        stem = os.path.splitext(filename)[0]
        image_id = stem.rsplit("_", 1)[-1]
        entries.append((image_id, filename))
    if not entries:
        raise FileNotFoundError(f"No images found under {folder}.")
    return entries


def split_train_val_ids(raw_dir=IDRID_SEGMENTATION_RAW_DIR, val_split=DEFAULT_VAL_SPLIT,
                         seed=DEFAULT_SEED):
    """Deterministic train/val split of IDRiD segmentation's 54-image
    Training Set only (same `train_test_split` mechanism
    `image_quality_dataset.load_eyeq_datasets` already uses for EyeQ, with
    a fixed `seed` so the split is reproducible run to run). The 27-image
    official Testing Set is a separate, fixed split -- see
    `load_lesion_segmentation_test_dataset` -- and is never touched by this
    function, so it can never leak into training or validation."""
    entries = _list_original_images(raw_dir, TRAIN_SPLIT_DIR)
    ids = [image_id for image_id, _ in entries]
    train_ids, val_ids = train_test_split(ids, test_size=val_split, random_state=seed)
    return sorted(train_ids), sorted(val_ids)


# --- Per-sample loading ---

def _mask_path(raw_dir, split_dir, category_folder, suffix, image_id):
    return os.path.join(
        raw_dir, _GROUNDTRUTHS_SUBDIR, split_dir, category_folder,
        f"IDRiD_{image_id}_{suffix}.tif",
    )


def _load_rgb_image(path):
    return np.array(Image.open(path).convert("RGB"), dtype=np.uint8)


def _load_binary_mask(path, shape):
    """Loads one IDRiD lesion mask, binarized to {0, 1} uint8 (verified:
    IDRiD's own `.tif` masks already store exactly {0, 1} palette indices,
    but `> 0` is applied regardless rather than assumed, for robustness).
    Returns an all-zero `shape` array when `path` doesn't exist -- IDRiD's
    own convention for "this lesion class is not present in this image"
    (confirmed: Soft Exudates masks are absent for many images), not an
    error."""
    if not os.path.exists(path):
        return np.zeros(shape, dtype=np.uint8)
    mask = np.array(Image.open(path))
    if mask.shape != shape:
        raise RuntimeError(
            f"Mask shape {mask.shape} at {path} does not match its image's shape {shape}."
        )
    return (mask > 0).astype(np.uint8)


def _vessel_cache_path(cache_dir, split_dir, image_id):
    safe_split = split_dir.split(". ", 1)[-1].replace(" ", "_")  # "a. Training Set" -> "Training_Set"
    return os.path.join(cache_dir, f"IDRiD_{image_id}_{safe_split}_vessel.npy")


def _get_or_compute_vessel_map(image_array, cache_path, vessel_model):
    """Returns the (H, W, 1) float32 vessel probability map for
    `image_array`, from `cache_path` if already computed, otherwise via
    Stage 03's unmodified `predict_vessel_mask()` -- cached to disk
    afterward so no later call (another epoch, another process) recomputes
    it. This is the only place this module calls into Stage 03."""
    if os.path.exists(cache_path):
        return np.load(cache_path)
    result = predict_vessel_mask(image_array, model=vessel_model)
    vessel_map = result["probability_map"].astype(np.float32)
    os.makedirs(os.path.dirname(cache_path), exist_ok=True)
    np.save(cache_path, vessel_map)
    return vessel_map


def _resize_input(input_array, image_size):
    resized = sk_resize(
        input_array, (*image_size, input_array.shape[-1]),
        order=1, mode="reflect", anti_aliasing=True, preserve_range=True,
    )
    return np.clip(resized, 0.0, 1.0).astype(np.float32)


def _resize_target(target_array, image_size):
    # order=1 + anti_aliasing (area-weighted downsampling) rather than
    # order=0 nearest: IDRiD's native resolution is ~4288x2848 and
    # microaneurysms are only a few pixels wide, so nearest-neighbor point
    # sampling would drop most of them outright depending on sampling luck.
    # Anti-aliased linear resizing accumulates each output pixel's coverage
    # from every source pixel underneath it (including tiny lesions), so a
    # near-zero-but-nonzero result still survives the resize; re-binarizing
    # at `> 0` (rather than 0.5) then treats "any surviving trace of this
    # lesion" as present, preserving far more small lesions than a hard
    # nearest-neighbor downsample would. The target tensor is still exactly
    # {0, 1} after this -- never a soft/probabilistic value.
    resized = sk_resize(
        target_array, (*image_size, target_array.shape[-1]),
        order=1, mode="constant", cval=0.0, anti_aliasing=True, preserve_range=True,
    )
    return (resized > 0).astype(np.float32)


def _build_sample(image_id, filename, raw_dir, processed_dir, split_dir,
                   vessel_cache_dir, vessel_model, image_size):
    """Builds one (input, target) pair, both already resized to
    `image_size`. `input` is (H, W, 4) float32 in [0, 1]; `target` is
    (H, W, 4) float32 in {0, 1}, channel order == LESION_CLASSES."""
    processed_path = os.path.join(processed_dir, _ORIGINAL_IMAGES_SUBDIR, split_dir, filename)
    if not os.path.exists(processed_path):
        raise FileNotFoundError(
            f"Stage 02 processed image not found at {processed_path}. Lesion "
            "Segmentation consumes Stage 02's output and never regenerates it "
            "(PROJECT_CODE.md's Stage 02 Preprocessing Policy) -- run Stage 02 "
            "(colab/notebooks/stage02_preprocessing.ipynb) first, or pass "
            "processed_dir= pointing at wherever it was staged (e.g. the "
            "Colab local SSD copy staged from Google Drive)."
        )
    rgb = _load_rgb_image(processed_path)

    cache_path = _vessel_cache_path(vessel_cache_dir, split_dir, image_id)
    vessel_map = _get_or_compute_vessel_map(rgb, cache_path, vessel_model)
    if vessel_map.shape[:2] != rgb.shape[:2]:
        raise RuntimeError(
            f"Vessel probability map shape {vessel_map.shape[:2]} does not match "
            f"processed image shape {rgb.shape[:2]} for {filename}."
        )

    input_array = np.concatenate([rgb.astype(np.float32) / 255.0, vessel_map], axis=-1)

    mask_channels = [
        _load_binary_mask(_mask_path(raw_dir, split_dir, category_folder, suffix, image_id), rgb.shape[:2])
        for _, category_folder, suffix in _MASK_SPECS
    ]
    target_array = np.stack(mask_channels, axis=-1).astype(np.float32)

    return _resize_input(input_array, image_size), _resize_target(target_array, image_size)


def _augment(input_array, target_array, rng):
    """Random horizontal flip, vertical flip, and 90-degree rotation,
    applied identically to `input_array` and `target_array` so spatial
    correspondence between the two is always preserved. Training split
    only -- never applied to validation or test data."""
    if rng.random() < 0.5:
        input_array = input_array[:, ::-1, :]
        target_array = target_array[:, ::-1, :]
    if rng.random() < 0.5:
        input_array = input_array[::-1, :, :]
        target_array = target_array[::-1, :, :]
    k = int(rng.integers(0, 4))
    if k:
        input_array = np.rot90(input_array, k=k, axes=(0, 1))
        target_array = np.rot90(target_array, k=k, axes=(0, 1))
    return np.ascontiguousarray(input_array), np.ascontiguousarray(target_array)


def _resolve_vessel_model(vessel_model, vessel_model_path):
    """Resolves the Stage 03 model to use exactly once per dataset build
    (not once per epoch/sample) -- reused unmodified via
    `vessel_segmentation_inference.load_vessel_model`, never reimplemented
    here."""
    if vessel_model is not None:
        return vessel_model
    return load_vessel_model(vessel_model_path)


def _make_dataset(entries, raw_dir, processed_dir, split_dir, vessel_cache_dir,
                   vessel_model, image_size, batch_size, shuffle, augment, seed):
    entries = list(entries)

    def gen():
        rng = np.random.default_rng(seed) if augment else None
        for image_id, filename in entries:
            x, y = _build_sample(
                image_id, filename, raw_dir, processed_dir, split_dir,
                vessel_cache_dir, vessel_model, image_size,
            )
            if augment:
                x, y = _augment(x, y, rng)
            yield x, y

    output_signature = (
        tf.TensorSpec(shape=(*image_size, 4), dtype=tf.float32),
        tf.TensorSpec(shape=(*image_size, 4), dtype=tf.float32),
    )
    ds = tf.data.Dataset.from_generator(gen, output_signature=output_signature)
    if shuffle:
        ds = ds.shuffle(buffer_size=max(len(entries), 1), seed=seed, reshuffle_each_iteration=True)
    return ds.batch(batch_size).prefetch(tf.data.AUTOTUNE)


# --- Public API ---

def load_lesion_segmentation_datasets(
    raw_dir=IDRID_SEGMENTATION_RAW_DIR,
    processed_dir=IDRID_SEGMENTATION_PROCESSED_DIR,
    vessel_cache_dir=DEFAULT_VESSEL_CACHE_DIR,
    vessel_model=None,
    vessel_model_path=DEFAULT_VESSEL_MODEL_PATH,
    image_size=DEFAULT_IMAGE_SIZE,
    val_split=DEFAULT_VAL_SPLIT,
    batch_size=DEFAULT_BATCH_SIZE,
    seed=DEFAULT_SEED,
    augment_train=True,
):
    """
    Train/val `tf.data.Dataset` pipelines built from IDRiD segmentation's
    54-image Training Set only, split deterministically per
    `split_train_val_ids`. The 27-image official Testing Set is never
    included here -- see `load_lesion_segmentation_test_dataset`.

    Pass an already-loaded `vessel_model` (`vessel_segmentation_inference.
    load_vessel_model()`) to reuse it across many calls/datasets;
    otherwise it is loaded once (not per-sample, not per-epoch) from
    `vessel_model_path`.

    Returns (train_ds, val_ds), each yielding batches of
    ((batch, H, W, 4) float32 input, (batch, H, W, 4) float32 target).
    """
    train_ids, val_ids = split_train_val_ids(raw_dir, val_split=val_split, seed=seed)
    filenames_by_id = dict(_list_original_images(raw_dir, TRAIN_SPLIT_DIR))
    train_entries = [(i, filenames_by_id[i]) for i in train_ids]
    val_entries = [(i, filenames_by_id[i]) for i in val_ids]

    resolved_vessel_model = _resolve_vessel_model(vessel_model, vessel_model_path)

    train_ds = _make_dataset(
        train_entries, raw_dir, processed_dir, TRAIN_SPLIT_DIR, vessel_cache_dir,
        resolved_vessel_model, image_size, batch_size, shuffle=True,
        augment=augment_train, seed=seed,
    )
    val_ds = _make_dataset(
        val_entries, raw_dir, processed_dir, TRAIN_SPLIT_DIR, vessel_cache_dir,
        resolved_vessel_model, image_size, batch_size, shuffle=False,
        augment=False, seed=seed,
    )
    return train_ds, val_ds


def load_lesion_segmentation_test_dataset(
    raw_dir=IDRID_SEGMENTATION_RAW_DIR,
    processed_dir=IDRID_SEGMENTATION_PROCESSED_DIR,
    vessel_cache_dir=DEFAULT_VESSEL_CACHE_DIR,
    vessel_model=None,
    vessel_model_path=DEFAULT_VESSEL_MODEL_PATH,
    image_size=DEFAULT_IMAGE_SIZE,
    batch_size=DEFAULT_BATCH_SIZE,
):
    """
    The held-out, official 27-image IDRiD segmentation Testing Set --
    for `LesionSegmentationStage.evaluate()` only. Never used for training
    or validation (kept entirely separate from `split_train_val_ids`,
    which only ever sees the 54-image Training Set).
    """
    entries = _list_original_images(raw_dir, TEST_SPLIT_DIR)
    resolved_vessel_model = _resolve_vessel_model(vessel_model, vessel_model_path)
    return _make_dataset(
        entries, raw_dir, processed_dir, TEST_SPLIT_DIR, vessel_cache_dir,
        resolved_vessel_model, image_size, batch_size, shuffle=False,
        augment=False, seed=None,
    )
