"""
Kappa-yielding training data for the C3 experiment -- an ADDITIVE module.

`improved_training_data.py` and `joint_training_dataset.py` are NOT modified. This module is a
thin wrapper that adds ONE field to each sample: the per-lesion-class reliability vector `kappa`,
`(4,)`, which the existing pipeline already loads from disk and then throws away.

Why a wrapper rather than an edit. `joint_training_dataset._build_joint_sample` reads the
reliability `.npz` (via `_get_or_compute_joint_frozen_outputs`) and discards `kappa` on line 694:

    vessel_map, lesion_maps, _kappa, r = _get_or_compute_joint_frozen_outputs(...)
    ...
    return {..., "reliability": np.float32(r), ...}

That function is on `multiseed_runs.TRAINING_BEHAVIOR_SOURCES`, so editing it would change the
training-behaviour fingerprint of the already-completed RACAF and NO_RACAF runs. Instead,
`load_cached_sample_kappa()` calls `improved_training_data.load_cached_sample()` COMPLETELY
UNCHANGED -- inheriting its augmentation, its empty-field-of-view handling and its
`UncachedEntryError` refusal to fall back to a raw image -- and then reads `kappa` from the SAME
already-cached `.npz`, resolved through RACAF's own `racaf.reliability_cache_path()`. No second
cache, no second path convention, no second augmentation implementation.

The double read is deliberate and cheap. `_build_joint_sample` opens that `.npz` internally and
this module opens it again, but training reads the LOCAL extracted cache (`/content/cache/...`),
not Drive, and the file holds 4 float32 plus one scalar -- a few hundred bytes, already in the OS
page cache from the read moments earlier. Avoiding it would require either editing the frozen
builder or reimplementing augmentation, both of which cost far more than the read.

NO Stage 04 inference, NO cache regeneration and NO cache writes happen anywhere in this module.
`kappa` is read-only, exactly as `racaf.compute_reliability` originally wrote it.

KAPPA CHANNEL ORDER is `lesion_segmentation_dataset.LESION_CLASSES` order -- **(MA, HE, EX, SE)**.
This module never reorders it.

The ONLY intentional difference from `improved_training_data.make_epoch_dataset`'s output
signature is the third element of the input tuple: `TensorSpec(shape=())` (the scalar `r`) becomes
`TensorSpec(shape=(4,))` (`kappa`). Ordering, the epoch/run-seed augmentation RNG, batching,
`drop_remainder=False` and prefetching are the existing functions, called directly.
"""

import numpy as np
import tensorflow as tf

import improved_training_data as itd
import joint_training_dataset as jtd
import local_feature_extraction_dataset as lfed
import racaf

#: `kappa`'s dimensionality, reused from RACAF rather than hardcoded as 4.
KAPPA_DIM = racaf.NUM_LESION_CLASSES


class MissingKappaError(RuntimeError):
    """A cached reliability entry exists but does not hold a usable `(4,)` kappa."""


def load_kappa(id_code, racaf_cache_dir):
    """`kappa` for one image, `(4,)` float32, read from the existing RACAF reliability cache.

    Never computes, never writes, never falls back: a missing or malformed entry raises. The path
    comes from `racaf.reliability_cache_path()` -- the same builder the pipeline itself uses, not
    a filename template duplicated here."""
    path = racaf.reliability_cache_path(racaf_cache_dir, id_code)
    try:
        with np.load(path) as cached:
            kappa = np.asarray(cached["kappa"], dtype=np.float32)
    except Exception as error:  # noqa: BLE001 -- any failure here means an unusable entry
        raise MissingKappaError(
            f"{id_code}: could not read kappa from {path} ({error!r}). C3 reads the existing "
            "reliability cache read-only and never recomputes Stage 04.") from error
    if kappa.shape != (KAPPA_DIM,):
        raise MissingKappaError(
            f"{id_code}: expected kappa shape ({KAPPA_DIM},) at {path}, found {kappa.shape}.")
    return kappa


def load_cached_sample_kappa(id_code, diagnosis, cache_dir, racaf_cache_dir, augment, rng,
                             image_size=jtd.STAGE5_IMAGE_SIZE):
    """One joint sample with `kappa` added.

    `improved_training_data.load_cached_sample()` is called unchanged and supplies every existing
    field (`image_id`, `stage5_input`, `stage6_input`, `reliability`, `grade`) with the identical
    augmentation and identical local-cache-only guarantees; this function only adds `"kappa"`.

    The original keys are deliberately preserved rather than renamed, so anything already written
    against `_build_joint_sample`'s contract keeps working; `"reliability"` is left in place
    (unused by C3) rather than stripped, so the two paths stay diffable."""
    sample = itd.load_cached_sample(id_code, diagnosis, cache_dir, racaf_cache_dir, augment, rng,
                                    image_size=image_size)
    sample["kappa"] = load_kappa(id_code, racaf_cache_dir)
    return sample


def make_epoch_dataset_kappa(entries, epoch, run_seed, cache_dir, racaf_cache_dir, batch_size,
                             augment, image_size=jtd.STAGE5_IMAGE_SIZE):
    """One epoch's `tf.data.Dataset` yielding `((stage5, stage6, kappa), grade)`.

    Structurally identical to `improved_training_data.make_epoch_dataset` -- it calls that
    module's own `epoch_training_order()` and `per_image_augmentation_rng()` directly, so the
    per-epoch ordering and the `(run_seed, epoch, image_id)` augmentation RNG are the SAME
    functions, not copies -- with exactly one difference: the third input `TensorSpec` is
    `(KAPPA_DIM,)` instead of the scalar reliability's `()`."""
    ordered_entries = itd.epoch_training_order(entries, run_seed, epoch) if augment else list(entries)

    def gen():
        for id_code, diagnosis in ordered_entries:
            rng = itd.per_image_augmentation_rng(run_seed, epoch, id_code) if augment else None
            sample = load_cached_sample_kappa(id_code, diagnosis, cache_dir, racaf_cache_dir,
                                              augment, rng, image_size=image_size)
            yield (
                (sample["stage5_input"], sample["stage6_input"], sample["kappa"]),
                sample["grade"],
            )

    output_signature = (
        (
            tf.TensorSpec(shape=(*image_size, lfed.NUM_CHANNELS), dtype=tf.float32),
            tf.TensorSpec(shape=(*jtd.STAGE6_IMAGE_SIZE, 3), dtype=tf.float32),
            tf.TensorSpec(shape=(KAPPA_DIM,), dtype=tf.float32),
        ),
        tf.TensorSpec(shape=(), dtype=tf.int32),
    )
    ds = tf.data.Dataset.from_generator(gen, output_signature=output_signature)
    return ds.batch(batch_size, drop_remainder=False).prefetch(tf.data.AUTOTUNE)


def evaluate_c3_from_disk(model, entries, cache_dir, racaf_cache_dir, batch_size=8):
    """The C3 twin of `multiseed_runs.evaluate_arm_from_disk()`: runs `model` over `entries`
    deterministically (no augmentation, manifest order), feeding `kappa` `(N, 4)` where that
    function feeds reliability `(N, 1)`, and returning rows in the SAME 27-column schema via
    `multiseed_runs.build_per_sample_rows()` so every existing metric/audit tool reads them
    unchanged. `entries` must already be cached locally."""
    import corn
    import multiseed_runs as msr

    ids, grades, s5, s6, kap = [], [], [], [], []
    all_logits, all_true, all_ids = [], [], []

    def flush():
        if not ids:
            return
        logits = model.predict_on_batch(
            [np.stack(s5), np.stack(s6), np.stack(kap).reshape(-1, KAPPA_DIM)]
        )
        all_logits.append(np.asarray(logits, dtype=np.float64))
        all_true.extend(grades)
        all_ids.extend(ids)
        ids.clear(); grades.clear(); s5.clear(); s6.clear(); kap.clear()

    for id_code, diagnosis in entries:
        sample = load_cached_sample_kappa(id_code, diagnosis, cache_dir, racaf_cache_dir, False, None)
        ids.append(id_code)
        grades.append(int(diagnosis))
        s5.append(sample["stage5_input"])
        s6.append(sample["stage6_input"])
        kap.append(sample["kappa"])
        if len(ids) >= batch_size:
            flush()
    flush()

    logits = (np.concatenate(all_logits, axis=0) if all_logits
             else np.zeros((0, corn.NUM_THRESHOLDS), dtype=np.float64))
    true_grades = np.asarray(all_true, dtype=int)
    decoded = corn.decode_logits(logits)
    return msr.build_per_sample_rows(all_ids, true_grades, logits, decoded)
