"""
APTOS 2019 dataset loader for Global Feature Extraction (pipeline Stage 06).

Reads the same `datasets/APTOS2019/raw/train.csv` + `raw/train_images/`
Stage 05 already reads (never modified -- `raw/` is read-only per
PROJECT_CODE.md's Dataset Policy) and builds `(input, label)` `tf.data.Dataset`
pairs: a Stage-02-processed RGB tensor per image, and its APTOS DR grade
(0-4).

Per-sample input tensor -- `(256, 256, 3)` float32, [0, 1] -- Stage 02
processed RGB only. No vessel probability, no lesion probability maps, no
Stage 05 `local_features` -- Stage 06 is a parallel, segmentation-independent
branch (`PROJECT_STRUCTURE.md` Sec 6: "directly ... parallel to, not
sequential with, Local Feature Extraction").

This module deliberately reuses `local_feature_extraction_dataset.py`'s
existing Stage-02-application and CSV-reading helpers (imported, not
duplicated) rather than re-implementing "load a raw APTOS image, apply
Stage 02 live" a second time -- both stages read the exact same raw files
and must apply the exact same frozen Gamma+CLAHE pipeline, so having two
independent implementations of that step would risk them silently drifting
apart. `local_feature_extraction_dataset.py` itself is not modified by this
module -- Stage 05 is unchanged.

Label: the `diagnosis` column of `train.csv`, 0-4, returned as a plain
int32 scalar -- not one-hot, not CORN's cumulative-target encoding.

Split: reuses `local_feature_extraction_dataset.split_train_val_ids`
unmodified, which itself now delegates to `downstream_split.
get_authoritative_split` -- the ONE authoritative, stratified split shared
by every downstream stage, not a Stage-05-owned one. So Stage 05 and Stage
06 see the identical train/val partition of APTOS 2019 -- required once
they are trained jointly (a future, not-yet-implemented step), since a
mismatched split between the two parallel branches would make "the same
training example" ambiguous.

Augmentation: reuses `local_feature_extraction_dataset._augment_spatial`
and `_augment_intensity_rgb` unmodified. Both are channel-count-agnostic
(spatial) or operate on the first 3 channels only (intensity) -- since
Stage 06's entire input *is* 3-channel RGB, both apply correctly here
without any adaptation.

IDRiD and EyeQ are never read by this module in any capacity -- Stage 06
uses APTOS 2019 only, per the approved design.
"""

import numpy as np
import tensorflow as tf

import local_feature_extraction_dataset as lfed

NUM_CHANNELS = 3  # Stage 02 RGB only -- no vessel, no lesion channels

DEFAULT_TRAIN_CSV = lfed.DEFAULT_TRAIN_CSV
DEFAULT_TRAIN_IMAGE_DIR = lfed.DEFAULT_TRAIN_IMAGE_DIR

DEFAULT_IMAGE_SIZE = (256, 256)
DEFAULT_VAL_SPLIT = lfed.DEFAULT_VAL_SPLIT
DEFAULT_BATCH_SIZE = lfed.DEFAULT_BATCH_SIZE
DEFAULT_SEED = lfed.DEFAULT_SEED


def split_train_val_ids(csv_path=DEFAULT_TRAIN_CSV, val_split=DEFAULT_VAL_SPLIT, seed=DEFAULT_SEED):
    """Identical to `local_feature_extraction_dataset.split_train_val_ids`
    -- re-exported here (not reimplemented) so Stage 06 uses the exact same
    deterministic train/val partition of APTOS 2019 as Stage 05, required
    for their eventual joint training run."""
    return lfed.split_train_val_ids(csv_path, val_split=val_split, seed=seed)


def build_global_feature_input(image, image_size=DEFAULT_IMAGE_SIZE):
    """
    Builds one `(*image_size, 3)` float32 Stage 06 input tensor for a
    single **already Stage-02-processed** RGB image (a file path, PIL
    Image, or `(H, W, 3)` uint8 RGB array -- see
    `vessel_segmentation_inference._load_rgb_array`, reused transitively
    via `local_feature_extraction_dataset`). Mirrors
    `local_feature_extraction_dataset.build_local_feature_input`'s
    single-image entry-point contract, minus the vessel/lesion channels
    Stage 06 never consumes.
    """
    from vessel_segmentation_inference import _load_rgb_array
    rgb = _load_rgb_array(image)
    input_array = rgb.astype(np.float32) / 255.0
    return lfed._resize_input(input_array, image_size)


DEFAULT_PROCESSED_DIR = lfed.DEFAULT_PROCESSED_DIR


def _build_sample(id_code, diagnosis, image_dir, image_size, processed_dir=DEFAULT_PROCESSED_DIR):
    """Builds one `(input, label)` pair for one APTOS training-set image:
    Stage 02 output resolved via `local_feature_extraction_dataset`'s
    existing `_resolve_processed_rgb` helper (unmodified, reused not
    duplicated -- an existing `processed_dir` file if present, live
    application otherwise), resized. `input` is `(*image_size, 3)` float32;
    `label` is the plain `int` APTOS DR grade (0-4). `processed_dir`
    defaults to the same `DEFAULT_PROCESSED_DIR` Stage 05 uses, so every
    existing positional caller (this module's own
    `load_global_feature_extraction_datasets`, and
    `colab/notebooks/stage06_global_feature_extraction.ipynb`'s direct
    call) keeps working unchanged."""
    raw_bgr = lfed._load_raw_bgr(image_dir, id_code)
    rgb = lfed._resolve_processed_rgb(raw_bgr, processed_dir, id_code)
    input_array = build_global_feature_input(rgb, image_size=image_size)
    return input_array, diagnosis


def _augment(input_array, rng):
    """Synchronized spatial (flip/rotate) + RGB intensity augmentation,
    reusing `local_feature_extraction_dataset`'s existing functions
    unmodified -- both apply correctly to a pure-RGB tensor as-is."""
    input_array = lfed._augment_spatial(input_array, rng)
    input_array = lfed._augment_intensity_rgb(input_array, rng)
    return input_array


def _make_dataset(entries, image_dir, image_size, batch_size, shuffle, augment, seed,
                   processed_dir=DEFAULT_PROCESSED_DIR):
    entries = list(entries)

    def gen():
        rng = np.random.default_rng(seed) if augment else None
        for id_code, diagnosis in entries:
            x, y = _build_sample(id_code, diagnosis, image_dir, image_size, processed_dir=processed_dir)
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


def load_global_feature_extraction_datasets(
    csv_path=DEFAULT_TRAIN_CSV,
    image_dir=DEFAULT_TRAIN_IMAGE_DIR,
    image_size=DEFAULT_IMAGE_SIZE,
    val_split=DEFAULT_VAL_SPLIT,
    batch_size=DEFAULT_BATCH_SIZE,
    seed=DEFAULT_SEED,
    augment_train=True,
    processed_dir=DEFAULT_PROCESSED_DIR,
):
    """
    Train/val `tf.data.Dataset` pipelines built from APTOS2019's labeled
    `train.csv`, split per `split_train_val_ids` (the same authoritative
    split Stage 05's own loader produces, given the same `val_split`/
    `seed`). `processed_dir` is checked for an existing Stage 02 output per
    image before falling back to live preprocessing.

    Returns `(train_ds, val_ds)`, each yielding batches of `((batch, H, W,
    3) float32 input, (batch,) int32 label)`.
    """
    train_entries, val_entries = split_train_val_ids(csv_path, val_split=val_split, seed=seed)

    train_ds = _make_dataset(
        train_entries, image_dir, image_size, batch_size, shuffle=True, augment=augment_train, seed=seed,
        processed_dir=processed_dir,
    )
    val_ds = _make_dataset(
        val_entries, image_dir, image_size, batch_size, shuffle=False, augment=False, seed=seed,
        processed_dir=processed_dir,
    )
    return train_ds, val_ds
