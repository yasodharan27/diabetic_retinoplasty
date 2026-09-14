"""
Epoch-indexed, position-independent training data for the multi-seed improved-training
experiment -- an ADDITIVE module. `joint_training_dataset.py` is not modified.

Fixes the pre-existing augmentation defect documented in `JOINT_TRAINING_ARCHITECTURE.md` Sec 49
("`gen()` re-creates `rng = np.random.default_rng(seed)` on every dataset iteration, so every
image receives an identical augmentation in every epoch") WITHOUT changing the augmentation
policy itself: `local_feature_extraction_dataset._augment_spatial()` and `_augment_intensity_rgb()`
(both unmodified) are still the only two operations applied, in the same order, to the same
8-channel Stage 05 tensor, via `joint_training_dataset._build_joint_sample()` (also unmodified --
it already accepts an externally supplied `rng`).

Design (audited; see the improved-training audit's Part A/B3):

  - **Per-image augmentation RNG**: a function of `(run_seed, epoch, image_id)` ONLY --
    independent of iteration position, shuffle order, prefetch timing, or which other images were
    skipped for an empty field of view this epoch. The same triple always produces the same
    augmentation; a different epoch or run seed always produces a different one.
  - **Per-epoch training order**: a full permutation of the training ids, a function of
    `(run_seed, epoch)` ONLY -- replacing `joint_training_dataset`'s 256-element `tf.data`
    shuffle buffer (which only approximates shuffling and is not resume-aware). This also removes
    that shuffle buffer's ~2 GB of held, fully-materialized samples.
  - **Resume correctness**: because both of the above are pure functions of `(run_seed, epoch,
    image_id)` and never of "how many batches have been consumed so far", rebuilding epoch `e`'s
    dataset from scratch -- whether because Colab died mid-epoch and this same epoch is being
    retried, or because a brand-new runtime is continuing a later epoch -- reproduces EXACTLY the
    same stream every time. No batch-level or iterator-level state needs to be saved or restored
    for this to hold.

The keying uses a plain SHA-256-derived integer seed rather than `numpy.random.SeedSequence`'s
own entropy-mixing (whose accepted integer range/spawning semantics have changed across NumPy
versions): `hashlib.sha256("|".join(...))` is stable across NumPy/Python versions and trivially
auditable by hand.
"""

import hashlib

import numpy as np
import tensorflow as tf

import joint_training_dataset as jtd
import local_feature_extraction_dataset as lfed
from vessel_segmentation_inference import EmptyFieldOfViewError

#: Distinguishes the augmentation-RNG stream from the order-RNG stream so the two never
#: accidentally collide on the same seed for the same (run_seed, epoch, id_code)-shaped key.
_AUGMENTATION_TAG = "improved_training_data.augmentation.v1"
_ORDER_TAG = "improved_training_data.order.v1"


def _seed_from_key(tag, *parts):
    """A deterministic, position-independent, 63-bit-safe seed derived from `tag` and `parts`.

    `np.random.default_rng()` accepts any non-negative Python int, so the full 64 bits of the
    SHA-256 digest's first 8 bytes are used directly (masked to 63 bits purely so the value is
    also a valid, unsurprising plain non-negative int on every platform)."""
    key = "|".join(str(p) for p in (tag,) + parts)
    digest = hashlib.sha256(key.encode("utf-8")).digest()
    return int.from_bytes(digest[:8], "big") & 0x7FFF_FFFF_FFFF_FFFF


def per_image_augmentation_rng(run_seed, epoch, id_code):
    """The `numpy.random.Generator` `joint_training_dataset._build_joint_sample()`'s `rng`
    argument must receive for one image in one epoch of one run -- a pure function of
    `(run_seed, epoch, id_code)`, nothing else."""
    return np.random.default_rng(_seed_from_key(_AUGMENTATION_TAG, run_seed, epoch, id_code))


def epoch_training_order(entries, run_seed, epoch):
    """A full permutation of `entries` (a list of `(id_code, diagnosis)` pairs), deterministic in
    `(run_seed, epoch)` alone -- never in `entries`' own starting order beyond WHICH ids it lists,
    so the identical sequence is produced however `entries` happens to be assembled, as long as it
    lists the same ids.

    Sorts by `id_code` FIRST to fix a canonical starting order, THEN permutes that canonical
    sequence -- permuting `entries` in whatever order it happened to arrive in would make the
    RESULT depend on that arrival order even though the same `(run_seed, epoch)` always produces
    the same permutation of *positions*: position `k` of two differently-ordered input lists holds
    a different id, so `[entries[i] for i in rng.permutation(n)]` differs unless `entries` was
    already in some agreed-upon order. Sorting first removes that dependency entirely."""
    canonical = sorted(entries, key=lambda entry: entry[0])
    rng = np.random.default_rng(_seed_from_key(_ORDER_TAG, run_seed, epoch))
    order = rng.permutation(len(canonical))
    return [canonical[i] for i in order]


def make_epoch_dataset(entries, epoch, run_seed, image_dir, cache_dir, racaf_cache_dir,
                       vessel_model, stage4_model, batch_size, augment,
                       processed_dir=jtd.DEFAULT_PROCESSED_DIR, persistent_cache_dir=None,
                       persistent_racaf_cache_dir=None, rgb_cache_dir=None,
                       image_size=jtd.STAGE5_IMAGE_SIZE):
    """One epoch's `tf.data.Dataset`, built fresh every call -- never reused across epochs and
    never `.repeat()`-ed, so a batch can never span two epochs.

    `augment=True` (training): `entries` is reordered by `epoch_training_order(entries, run_seed,
    epoch)` and each sample is built with `per_image_augmentation_rng(run_seed, epoch, id_code)`.
    `augment=False` (validation): `entries`' own order is used unchanged and no `rng` is passed --
    identical to `joint_training_dataset._make_joint_dataset(..., shuffle=False, augment=False)`'s
    existing, unmodified validation behaviour.

    Reuses `joint_training_dataset._build_joint_sample()` UNCHANGED for the actual per-sample
    construction (Stage 03/04/RACAF-cache reads, canonical RGB, augmentation application, Stage 06
    resize) -- this module only supplies WHICH order and WHICH `rng` that function receives.
    `EmptyFieldOfViewError` is caught and the image skipped, exactly as
    `joint_training_dataset._make_joint_dataset()`'s own generator does."""
    ordered_entries = epoch_training_order(entries, run_seed, epoch) if augment else list(entries)

    def gen():
        for id_code, diagnosis in ordered_entries:
            rng = per_image_augmentation_rng(run_seed, epoch, id_code) if augment else None
            try:
                sample = jtd._build_joint_sample(
                    id_code, diagnosis, image_dir, cache_dir, racaf_cache_dir,
                    vessel_model, stage4_model, augment, rng,
                    processed_dir=processed_dir, image_size=image_size,
                    persistent_cache_dir=persistent_cache_dir,
                    persistent_racaf_cache_dir=persistent_racaf_cache_dir,
                    rgb_cache_dir=rgb_cache_dir,
                )
            except EmptyFieldOfViewError:
                continue
            yield (
                (sample["stage5_input"], sample["stage6_input"], sample["reliability"]),
                sample["grade"],
            )

    output_signature = (
        (
            tf.TensorSpec(shape=(*image_size, lfed.NUM_CHANNELS), dtype=tf.float32),
            tf.TensorSpec(shape=(*jtd.STAGE6_IMAGE_SIZE, 3), dtype=tf.float32),
            tf.TensorSpec(shape=(), dtype=tf.float32),
        ),
        tf.TensorSpec(shape=(), dtype=tf.int32),
    )
    ds = tf.data.Dataset.from_generator(gen, output_signature=output_signature)
    return ds.batch(batch_size, drop_remainder=False).prefetch(tf.data.AUTOTUNE)


def count_cached_entries(entries, cache_dir, racaf_cache_dir, image_size=jtd.STAGE5_IMAGE_SIZE,
                         persistent_cache_dir=None, persistent_racaf_cache_dir=None):
    """The number of `entries` whose Stage 03/04/RACAF cache is already fully populated -- a
    cheap `os.path.exists`/`os.stat`-only scan (no image load, no model call, no Stage 03/04
    inference), used to derive the EXACT number of samples a training/validation epoch will yield
    once the empty-field-of-view images are excluded (`docs/`'s improved-training audit Part 12:
    "derive the actual number of training samples yielded ... from the real cache. Do not
    hard-code a guessed value.").

    This assumes the cache is already fully populated for every non-empty-FOV entry (true once
    `precompute_authoritative_joint_caches`/the finalized RACAF and NO-RACAF experiments' own
    Phase 1 have run against the full APTOS2019 manifest, as they already have on this project's
    Drive) -- an entry with NO cache anywhere is then, by elimination, one of the known empty-FOV
    ids, not an uncached-but-valid one. If the cache is only partially populated, this undercounts
    validly-yielding entries; callers should treat a mismatch against a PRIOR call's count as a
    signal to re-verify the cache (`[6]`-equivalent), not silently trust either number."""
    return sum(
        1 for id_code, _diagnosis in entries
        if jtd._cache_entry_exists(id_code, cache_dir, racaf_cache_dir, image_size)
        or (persistent_cache_dir is not None and jtd._cache_entry_exists(
            id_code, persistent_cache_dir, persistent_racaf_cache_dir, image_size, persistent=True))
    )
