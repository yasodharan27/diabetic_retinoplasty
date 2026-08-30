"""
Joint dataset loader for the Stage 05-08 + RACAF joint training run.

Authoritative design: `JOINT_TRAINING_ARCHITECTURE.md`. This module implements exactly that
design's dataset/cache/augmentation sections (§9-13, §18, §20) -- it does not redefine any
stage's architecture, RACAF's mathematics, or CORN's loss.

Per sample, this module builds:
  - `stage5_input`: `(512, 512, 8)` float32 -- canonical processed RGB(3) + vessel(1) + lesion(4),
    matching `local_feature_extraction_model.py`'s existing input contract exactly.
  - `stage6_input`: `(256, 256, 3)` float32 -- the SAME (possibly augmented) RGB, resized
    independently, matching `swin_transformer.create_dual_scale_swin_model()`'s existing input
    contract exactly.
  - `reliability`: a scalar float `r` -- RACAF's precomputed, cached image-level reliability,
    matching `racaf.build_racaf_fusion()`'s existing `r` input contract exactly.
  - `grade`: the plain int APTOS DR grade (0-4), from `train.csv`'s `diagnosis` column via the
    authoritative split -- never a second, independently computed split
    (`downstream_split.get_authoritative_split()`, `dataset_splits/aptos2019_train_val_split.csv`).

Step 4/5 redundancy fix (`JOINT_TRAINING_ARCHITECTURE.md` §11.1): this is the shared per-image
computation point that module deferred building -- `_get_or_compute_joint_frozen_outputs()` calls
`racaf.prepare_stage4_input()` + `racaf.tta_views()` exactly ONCE per (uncached) image, and derives
BOTH the canonical Stage 03/04 cache (consumed by Stage 05) AND RACAF's own `kappa`/`r` cache from
that single call's four TTA views -- eliminating the previously-unavoidable extra Stage 04 forward
pass `predict_lesion_mask()` would otherwise cost. `racaf.py` itself is not modified; every
function this module calls into it (`prepare_stage4_input`, `tta_views`, `compute_reliability`,
`TTA_TRANSFORMS`, `load_frozen_stage4_model`) is reused completely unmodified, per RACAF's own
"independently testable pieces" design (`racaf.py`'s module docstring). This module depends on
`racaf.py` -- not the reverse -- so no Stage 05 -> RACAF dependency is introduced into Stage 05's
own module (`local_feature_extraction_model.py`/`local_feature_extraction_dataset.py` are both
untouched by this file).

Cache reuse: the canonical Stage 03/04 vessel/lesion cache uses the EXACT SAME path convention
`local_feature_extraction_dataset._cache_path()` already established (Step 3 of the prior
infrastructure task) -- so a cache entry populated by either loader is found and reused by the
other; no second, competing cache is introduced. RACAF's reliability cache reuses
`racaf.reliability_cache_path()`/`racaf.DEFAULT_CACHE_DIR` unmodified.

Stage 1 (IQA) is never imported or called anywhere in this module -- per
`JOINT_TRAINING_ARCHITECTURE.md` §3.1 (locked), Stage 1 does not gate APTOS2019 downstream
training. No ground-truth segmentation mask, IDRiD label, or `test.csv` value is ever read here --
only Stage 03/04's own frozen, predicted outputs and `train.csv`'s `diagnosis` column via the
authoritative split.

Empty-FOV handling: a small number of real APTOS images make Stage 03's FOV circle-fit
(`vessel_segmentation_inference.compute_fov_mask`) degenerate to an empty result (see
`EmptyFieldOfViewError`'s docstring there for the exact mechanism). There is no project-sanctioned
fallback for this -- no full-image FOV substitute, no synthetic vessel/lesion map -- so
`_make_joint_dataset`'s generator explicitly catches `EmptyFieldOfViewError` per-image, logs the
skipped `image_id`, and excludes just that sample from the epoch. This does NOT modify the
authoritative split manifest on disk (`dataset_splits/aptos2019_train_val_split.csv` keeps every
id, unchanged) -- it only means a `tf.data.Dataset` built here may yield slightly fewer than the
manifest's nominal 2929 (train) / 733 (val) samples per epoch, by however many ids are affected.
"""

import logging
import os

import numpy as np
import tensorflow as tf
from skimage.transform import resize as sk_resize

import config
import downstream_split
import local_feature_extraction_dataset as lfed
import racaf
from vessel_segmentation_inference import (
    DEFAULT_MODEL_PATH as DEFAULT_VESSEL_MODEL_PATH,
    EmptyFieldOfViewError,
    load_vessel_model,
    predict_vessel_mask,
)

logger = logging.getLogger(__name__)

# --- Fixed by the approved joint design -- not free parameters of this module ---
STAGE5_IMAGE_SIZE = lfed.DEFAULT_IMAGE_SIZE  # (512, 512), also Stage 04's own native resolution
STAGE6_IMAGE_SIZE = (256, 256)  # matches global_feature_extraction_dataset.DEFAULT_IMAGE_SIZE

DEFAULT_TRAIN_CSV = lfed.DEFAULT_TRAIN_CSV
DEFAULT_TRAIN_IMAGE_DIR = lfed.DEFAULT_TRAIN_IMAGE_DIR
DEFAULT_PROCESSED_DIR = lfed.DEFAULT_PROCESSED_DIR

# Reuses Stage 05's existing Stage 03/04 cache directory/path convention exactly (not a second,
# competing cache) and RACAF's existing reliability cache directory/path convention exactly.
DEFAULT_CACHE_DIR = lfed.DEFAULT_CACHE_DIR
DEFAULT_RACAF_CACHE_DIR = racaf.DEFAULT_CACHE_DIR

DEFAULT_VAL_SPLIT = lfed.DEFAULT_VAL_SPLIT
DEFAULT_BATCH_SIZE = lfed.DEFAULT_BATCH_SIZE
DEFAULT_SEED = lfed.DEFAULT_SEED


def split_train_val_ids(csv_path=DEFAULT_TRAIN_CSV, val_split=DEFAULT_VAL_SPLIT, seed=DEFAULT_SEED):
    """Thin delegation to `downstream_split.get_authoritative_split` -- the SAME authoritative
    split Stage 05/Stage 06 already use, not a new one (`JOINT_TRAINING_ARCHITECTURE.md` §6)."""
    return downstream_split.get_authoritative_split(csv_path, val_split=val_split, seed=seed)


# --- Canonical (unaugmented) RGB resize -- never cached, cheap, mirrors lfed._resize_rgb's
# convention but keeps [0,1] float (not uint8), matching racaf.prepare_stage4_input's own RGB
# normalization so the two independent resizes stay numerically consistent. ---

def _resize_rgb_01(rgb_native_uint8, image_size):
    rgb = np.asarray(rgb_native_uint8, dtype=np.float32) / 255.0
    resized = sk_resize(
        rgb, (*image_size, 3), order=1, mode="reflect", anti_aliasing=True, preserve_range=True,
    )
    return np.clip(resized, 0.0, 1.0).astype(np.float32)


# --- Shared, per-image, cache-populating computation -- Step 4/5 redundancy fix ---

def _get_or_compute_joint_frozen_outputs(rgb_native, vessel_cache_path, lesion_cache_path,
                                          reliability_cache_path, vessel_model, stage4_model,
                                          image_size=STAGE5_IMAGE_SIZE):
    """Returns `(canonical_rgb, vessel_map, lesion_maps, kappa, r)` for one image --
    `canonical_rgb`/`vessel_map`: `(*image_size, {3,1})`, `lesion_maps`: `(*image_size, 4)`,
    `kappa`: `(4,)`, `r`: scalar float. `canonical_rgb` is always freshly resized (cheap, never
    cached -- mirrors Stage 02's own "resize is per-consumer" convention). The other four are
    read from their existing on-disk caches if all three are already present; otherwise computed
    together from ONE `racaf.tta_views()` call:

      - Stage 03 (frozen) runs once, at native resolution, via the unmodified
        `predict_vessel_mask()` -- exactly as Stage 05's own standalone loader already does.
      - `racaf.prepare_stage4_input()` (unmodified) resizes the native RGB+vessel to Stage 04's
        canonical `(512,512,4)` working resolution ONCE -- its RGB(3)+vessel(1) channels ARE the
        canonical vessel-map cache value (no second resize needed for vessel).
      - `racaf.tta_views()` (unmodified) runs the frozen Stage 04 model 4 times (identity + 3
        deterministic transforms) on that single prepared tensor -- the `"identity"`-indexed view
        IS Stage 05's canonical lesion-map cache value; ALL FOUR views feed
        `racaf.compute_reliability()` (unmodified) for `kappa`/`r`. This is the one Stage 04
        forward-pass workflow this image needs -- no separate, redundant identity-only prediction
        is ever computed.

    Stage 04 stays frozen throughout: `stage4_model` must be loaded via
    `racaf.load_frozen_stage4_model()` (`trainable=False`), and `racaf.tta_views()` already wraps
    every prediction in `tf.stop_gradient` -- neither is altered here.
    """
    all_cached = (
        os.path.exists(vessel_cache_path)
        and os.path.exists(lesion_cache_path)
        and os.path.exists(reliability_cache_path)
    )
    if all_cached:
        vessel_map = np.load(vessel_cache_path)
        lesion_maps = np.load(lesion_cache_path)
        reliability_cached = np.load(reliability_cache_path)
        return vessel_map, lesion_maps, reliability_cached["kappa"], float(reliability_cached["r"])

    native_vessel_map = predict_vessel_mask(rgb_native, model=vessel_model)["probability_map"].astype(np.float32)

    prepared = racaf.prepare_stage4_input(rgb_native, native_vessel_map, model_input_shape=image_size)
    aligned = racaf.tta_views(stage4_model, prepared)
    aligned_np = aligned.numpy()

    identity_index = racaf.TTA_TRANSFORMS.index("identity")
    vessel_map = np.asarray(prepared)[0, ..., 3:4].astype(np.float32)
    lesion_maps = aligned_np[0, identity_index, ...].astype(np.float32)

    reliability = racaf.compute_reliability(aligned_np)
    kappa = reliability["kappa"][0]
    r = float(reliability["r"][0])

    if not os.path.exists(vessel_cache_path):
        os.makedirs(os.path.dirname(vessel_cache_path), exist_ok=True)
        np.save(vessel_cache_path, vessel_map)
    if not os.path.exists(lesion_cache_path):
        os.makedirs(os.path.dirname(lesion_cache_path), exist_ok=True)
        np.save(lesion_cache_path, lesion_maps)
    if not os.path.exists(reliability_cache_path):
        os.makedirs(os.path.dirname(reliability_cache_path), exist_ok=True)
        np.savez(reliability_cache_path, kappa=kappa, r=np.float32(r))

    return vessel_map, lesion_maps, kappa, r


# --- Synchronized spatial/intensity augmentation, reusing Stage 05's existing functions ---

def _augment(stage5_input, rng):
    """Applies Stage 05's existing, unmodified spatial (`_augment_spatial`) and RGB-only
    intensity (`_augment_intensity_rgb`) augmentation to the FULL, already-concatenated
    `(512,512,8)` tensor -- so RGB/vessel/lesion stay spatially synchronized by construction, and
    Stage 06's input (derived from this same tensor's RGB channels afterward, see
    `_build_joint_sample`) automatically receives the identical spatial transform, never an
    independent one. Intensity jitter never touches channels 3-7 (vessel/lesion) -- unchanged
    `_augment_intensity_rgb` behavior, reused not reimplemented."""
    stage5_input = lfed._augment_spatial(stage5_input, rng)
    stage5_input = lfed._augment_intensity_rgb(stage5_input, rng)
    return stage5_input


def _build_joint_sample(id_code, diagnosis, image_dir, cache_dir, racaf_cache_dir,
                         vessel_model, stage4_model, augment, rng,
                         processed_dir=DEFAULT_PROCESSED_DIR, image_size=STAGE5_IMAGE_SIZE):
    """Builds one joint training sample:
      `{"image_id", "stage5_input", "stage6_input", "reliability", "grade"}`.

    `stage5_input`: `(512,512,8)` float32 -- possibly augmented.
    `stage6_input`: `(256,256,3)` float32 -- derived from `stage5_input`'s OWN (possibly
    augmented) RGB channels, resized independently -- never a second, independently-augmented RGB
    (Step 6 synchronization requirement).
    `reliability`: RACAF's cached scalar `r`, from the CANONICAL, UNAUGMENTED Stage 04 output --
    computed before augmentation and never itself augmented (`JOINT_TRAINING_ARCHITECTURE.md`
    §20).
    `grade`: the plain int APTOS DR grade, from the authoritative split.
    """
    raw_bgr = lfed._load_raw_bgr(image_dir, id_code)
    rgb_native = lfed._resolve_processed_rgb(raw_bgr, processed_dir, id_code)

    vessel_cache = lfed._cache_path(cache_dir, id_code, "vessel", image_size)
    lesion_cache = lfed._cache_path(cache_dir, id_code, "lesion", image_size)
    reliability_cache = racaf.reliability_cache_path(racaf_cache_dir, id_code)

    vessel_map, lesion_maps, _kappa, r = _get_or_compute_joint_frozen_outputs(
        rgb_native, vessel_cache, lesion_cache, reliability_cache, vessel_model, stage4_model,
        image_size=image_size,
    )

    canonical_rgb = _resize_rgb_01(rgb_native, image_size)
    stage5_input = np.concatenate([canonical_rgb, vessel_map, lesion_maps], axis=-1)

    if augment:
        stage5_input = _augment(stage5_input, rng)

    stage6_input = lfed._resize_input(stage5_input[..., :3], STAGE6_IMAGE_SIZE)

    return {
        "image_id": id_code,
        "stage5_input": stage5_input.astype(np.float32),
        "stage6_input": stage6_input.astype(np.float32),
        "reliability": np.float32(r),
        "grade": int(diagnosis),
    }


# --- tf.data construction -- mirrors local_feature_extraction_dataset.py's /
# global_feature_extraction_dataset.py's own _make_dataset pattern exactly ---

def _make_joint_dataset(entries, image_dir, cache_dir, racaf_cache_dir, vessel_model, stage4_model,
                         batch_size, shuffle, augment, seed, processed_dir=DEFAULT_PROCESSED_DIR):
    entries = list(entries)

    def gen():
        rng = np.random.default_rng(seed) if augment else None
        for id_code, diagnosis in entries:
            try:
                sample = _build_joint_sample(
                    id_code, diagnosis, image_dir, cache_dir, racaf_cache_dir,
                    vessel_model, stage4_model, augment, rng, processed_dir=processed_dir,
                )
            except EmptyFieldOfViewError:
                # A real but exceptional APTOS image: Stage 03's FOV circle-fit found no
                # fundus disk at all (see `EmptyFieldOfViewError`'s docstring). There is no
                # project-sanctioned fallback (no full-image FOV, no synthetic vessel/lesion
                # map) for this case, so the image is explicitly skipped/quarantined for this
                # epoch rather than crashing the whole run or fabricating a result. The
                # authoritative split manifest on disk is NOT modified -- this id remains
                # listed there; it is simply not yielded by this generator.
                logger.warning(
                    "Skipping image_id=%s: empty Stage 03 field-of-view (no fundus disk "
                    "detected). This image cannot be processed by the frozen vessel "
                    "segmentation FOV-crop step and is excluded from this epoch.",
                    id_code,
                )
                continue
            yield (
                (sample["stage5_input"], sample["stage6_input"], sample["reliability"]),
                sample["grade"],
            )

    output_signature = (
        (
            tf.TensorSpec(shape=(*STAGE5_IMAGE_SIZE, lfed.NUM_CHANNELS), dtype=tf.float32),
            tf.TensorSpec(shape=(*STAGE6_IMAGE_SIZE, 3), dtype=tf.float32),
            tf.TensorSpec(shape=(), dtype=tf.float32),
        ),
        tf.TensorSpec(shape=(), dtype=tf.int32),
    )
    ds = tf.data.Dataset.from_generator(gen, output_signature=output_signature)
    if shuffle:
        ds = ds.shuffle(buffer_size=max(len(entries), 1), seed=seed, reshuffle_each_iteration=True)
    return ds.batch(batch_size).prefetch(tf.data.AUTOTUNE)


def load_joint_training_datasets(
    csv_path=DEFAULT_TRAIN_CSV,
    image_dir=DEFAULT_TRAIN_IMAGE_DIR,
    cache_dir=DEFAULT_CACHE_DIR,
    racaf_cache_dir=DEFAULT_RACAF_CACHE_DIR,
    vessel_model=None,
    vessel_model_path=DEFAULT_VESSEL_MODEL_PATH,
    stage4_model=None,
    val_split=DEFAULT_VAL_SPLIT,
    batch_size=DEFAULT_BATCH_SIZE,
    seed=DEFAULT_SEED,
    augment_train=True,
    processed_dir=DEFAULT_PROCESSED_DIR,
):
    """Train/val `tf.data.Dataset` pipelines for the joint Stage 05-08+RACAF training run, built
    from the SAME authoritative split (`downstream_split.get_authoritative_split()`) Stage 05/06
    already use -- no second split. Each batch yields `((stage5_input, stage6_input, reliability),
    grade)`, directly consumable by `joint_training_model.build_joint_model()`'s
    `[local_input, global_input, r_input] -> logits` contract.

    Pass already-loaded `vessel_model`/`stage4_model` to reuse them across calls; otherwise each is
    loaded once here -- `stage4_model` via `racaf.load_frozen_stage4_model()` (`trainable=False`),
    never `lesion_segmentation_model.load_lesion_model()` directly, so Stage 04 stays frozen for
    every consumer of this loader.
    """
    train_entries, val_entries = split_train_val_ids(csv_path, val_split=val_split, seed=seed)

    resolved_vessel_model = vessel_model if vessel_model is not None else load_vessel_model(vessel_model_path)
    resolved_stage4_model = stage4_model if stage4_model is not None else racaf.load_frozen_stage4_model()

    train_ds = _make_joint_dataset(
        train_entries, image_dir, cache_dir, racaf_cache_dir, resolved_vessel_model, resolved_stage4_model,
        batch_size, shuffle=True, augment=augment_train, seed=seed, processed_dir=processed_dir,
    )
    val_ds = _make_joint_dataset(
        val_entries, image_dir, cache_dir, racaf_cache_dir, resolved_vessel_model, resolved_stage4_model,
        batch_size, shuffle=False, augment=False, seed=seed, processed_dir=processed_dir,
    )
    return train_ds, val_ds
