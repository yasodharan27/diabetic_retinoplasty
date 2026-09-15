"""
Epoch-indexed, position-independent, CACHE-ONLY training data for the multi-seed improved-training
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
    independent of iteration position, shuffle order, or prefetch timing. The same triple always
    produces the same augmentation; a different epoch or run seed always produces a different one.
  - **Per-epoch training order**: a full permutation of the training ids, a function of
    `(run_seed, epoch)` ONLY -- replacing `joint_training_dataset`'s 256-element `tf.data`
    shuffle buffer (which only approximates shuffling and is not resume-aware).
  - **Resume correctness**: both of the above are pure functions of `(run_seed, epoch, image_id)`,
    never of "how many batches have been consumed so far", so rebuilding epoch `e`'s dataset from
    scratch reproduces EXACTLY the same stream every time.

--- Data path: cache only ------------------------------------------------------------------

The augmentation operates on the cached, already-extracted representation -- canonical Stage 02
RGB + frozen Stage 03 vessel map + frozen Stage 04 lesion maps, concatenated -- exactly as the
finalized RACAF/NO-RACAF runs augmented it. RACAF's reliability `r` is itself a cached value. So
nothing in training needs a raw pixel, and the finalized experiments' persistent Drive cache (its
`cache_archive/` shards, extracted once per runtime) already holds every artifact for every image
that can be represented at all.

`load_cached_sample()` therefore reads ONLY the local cache: it refuses (`UncachedEntryError`)
rather than falling back to a raw image, a Drive read, or Stage 02/03/04 inference, and passes
no model into `_build_joint_sample()` at all -- a cache miss can never silently turn a training
epoch into an upstream-pipeline run.

The entries that have no cached representation are the known empty-field-of-view images (Stage 03
finds no fundus disk). The finalized runs never trained or evaluated on them either -- their
generator skipped them every epoch after re-running Stage 02/03 on each. Here they are excluded
once, up front (`locally_cached_entries()`), which yields the identical population without that
per-epoch recomputation.

`complete_local_cache()` is the ONE-TIME step that runs before training when the extracted local
cache is not already complete: it mirrors any entry that exists only in the loose Drive cache, and
only for an entry missing everywhere (and not already pinned as empty-FOV) does it stage that
entry's raw image and run the established Phase 1 generator (`precompute_joint_frozen_caches`),
persisting the result to Drive. Once an experiment's empty-FOV ids are pinned, a complete cache
never triggers raw staging, a Drive listing, or a model load again.

The keying uses a plain SHA-256-derived integer seed rather than `numpy.random.SeedSequence`'s
own entropy-mixing (whose accepted integer range/spawning semantics have changed across NumPy
versions): `hashlib.sha256("|".join(...))` is stable across NumPy/Python versions and trivially
auditable by hand.
"""

import hashlib
import os

import numpy as np
import tensorflow as tf

import joint_cache_diagnostics as jcd
import joint_cache_staging as jcs
import joint_training_dataset as jtd
import local_feature_extraction_dataset as lfed

#: Distinguishes the augmentation-RNG stream from the order-RNG stream so the two never
#: accidentally collide on the same seed for the same (run_seed, epoch, id_code)-shaped key.
_AUGMENTATION_TAG = "improved_training_data.augmentation.v1"
_ORDER_TAG = "improved_training_data.order.v1"


class UncachedEntryError(RuntimeError):
    """A training/evaluation entry has no complete LOCAL cache. Training never recomputes it."""


def _seed_from_key(tag, *parts):
    """A deterministic, position-independent, 63-bit-safe seed derived from `tag` and `parts`."""
    key = "|".join(str(p) for p in (tag,) + parts)
    digest = hashlib.sha256(key.encode("utf-8")).digest()
    return int.from_bytes(digest[:8], "big") & 0x7FFF_FFFF_FFFF_FFFF


def per_image_augmentation_rng(run_seed, epoch, id_code):
    """The `numpy.random.Generator` one image receives in one epoch of one run -- a pure function
    of `(run_seed, epoch, id_code)`, nothing else."""
    return np.random.default_rng(_seed_from_key(_AUGMENTATION_TAG, run_seed, epoch, id_code))


def epoch_training_order(entries, run_seed, epoch):
    """A full permutation of `entries` (a list of `(id_code, diagnosis)` pairs), deterministic in
    `(run_seed, epoch)` and the SET of ids alone. Sorts by `id_code` first so the result never
    depends on the order `entries` happened to arrive in."""
    canonical = sorted(entries, key=lambda entry: entry[0])
    rng = np.random.default_rng(_seed_from_key(_ORDER_TAG, run_seed, epoch))
    order = rng.permutation(len(canonical))
    return [canonical[i] for i in order]


# --- Local cache ---------------------------------------------------------------------------

def missing_local_artifacts(id_code, cache_dir, racaf_cache_dir, image_size=jtd.STAGE5_IMAGE_SIZE):
    """Which of the four cached artifacts (vessel, lesion, reliability, rgb) are absent locally.
    Local `os.path.exists` only -- never a Drive path."""
    paths = jcd.artifact_paths(id_code, cache_dir, racaf_cache_dir, image_size)
    return [artifact for artifact, path in paths.items() if not os.path.exists(path)]


def locally_cached_entries(entries, cache_dir, racaf_cache_dir, image_size=jtd.STAGE5_IMAGE_SIZE):
    """`entries` restricted to those whose four artifacts are all present locally, order kept."""
    return [entry for entry in entries
            if not missing_local_artifacts(entry[0], cache_dir, racaf_cache_dir, image_size)]


def load_cached_sample(id_code, diagnosis, cache_dir, racaf_cache_dir, augment, rng,
                       image_size=jtd.STAGE5_IMAGE_SIZE):
    """One joint sample built from the LOCAL cache only, via the unmodified
    `joint_training_dataset._build_joint_sample()` (same concatenation, augmentation and Stage 06
    resize as every prior run). No raw-image directory, no persistent cache and no Stage 03/04
    model are passed, so no upstream stage can run; an incomplete entry raises instead."""
    missing = missing_local_artifacts(id_code, cache_dir, racaf_cache_dir, image_size)
    if missing:
        raise UncachedEntryError(
            f"{id_code}: no local cache for {missing} under {cache_dir} / {racaf_cache_dir}. "
            "Training reads the extracted cache only -- run the notebook's [6] first.")
    return jtd._build_joint_sample(
        id_code, diagnosis, None, cache_dir, racaf_cache_dir, None, None, augment, rng,
        processed_dir=None, image_size=image_size,
    )


def make_epoch_dataset(entries, epoch, run_seed, cache_dir, racaf_cache_dir, batch_size, augment,
                       image_size=jtd.STAGE5_IMAGE_SIZE):
    """One epoch's `tf.data.Dataset`, built fresh every call -- never reused across epochs and
    never `.repeat()`-ed, so a batch can never span two epochs.

    `entries` must already be cached locally (`locally_cached_entries()`); every sample is read
    from that cache by `load_cached_sample()`.

    `augment=True` (training): reordered by `epoch_training_order(entries, run_seed, epoch)`, each
    sample built with `per_image_augmentation_rng(run_seed, epoch, id_code)`.
    `augment=False` (validation): `entries`' own order, no augmentation."""
    ordered_entries = epoch_training_order(entries, run_seed, epoch) if augment else list(entries)

    def gen():
        for id_code, diagnosis in ordered_entries:
            rng = per_image_augmentation_rng(run_seed, epoch, id_code) if augment else None
            sample = load_cached_sample(id_code, diagnosis, cache_dir, racaf_cache_dir, augment,
                                        rng, image_size=image_size)
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


def complete_local_cache(entries, cache_dir, racaf_cache_dir, persistent_cache_dir,
                         persistent_racaf_cache_dir, source_image_dir, local_image_dir,
                         known_empty_fov_ids=None, processed_dir=None,
                         image_size=jtd.STAGE5_IMAGE_SIZE):
    """Brings the local cache to completeness ONCE, before training; returns a report.

    For every entry not already fully local and not in `known_empty_fov_ids`:
      1. copy whatever exists in the loose persistent (Drive) cache
         (`joint_cache_staging.mirror_persistent_cache_to_local`, restricted to those entries);
      2. only for an entry still incomplete after that: stage its raw image
         (`stage_raw_images_for_uncached_entries`, restricted to those entries) and run the
         established Phase 1 generator (`precompute_joint_frozen_caches`) on just those entries;
         each newly cached entry's four files are copied to the persistent cache (never
         overwriting an existing file), and each empty-FOV result is reported.

    With every entry already local except the pinned empty-FOV ids, this does nothing: no Drive
    access, no raw image, no model. Raises if any entry ends incomplete without being empty-FOV.

    Report keys: `entries`, `already_local`, `mirrored_files`, `raw_images_staged`,
    `generated_ids`, `persisted_files`, `empty_fov_ids` (every entry still without a cache)."""
    entries = list(entries)
    known = set(known_empty_fov_ids or ())
    processed_dir = processed_dir if processed_dir is not None else jtd.DEFAULT_PROCESSED_DIR

    def incomplete(candidates):
        return [e for e in candidates
                if missing_local_artifacts(e[0], cache_dir, racaf_cache_dir, image_size)]

    report = {"entries": len(entries), "already_local": 0, "mirrored_files": 0,
              "raw_images_staged": 0, "generated_ids": [], "persisted_files": 0,
              "empty_fov_ids": []}
    not_local = incomplete(entries)
    report["already_local"] = len(entries) - len(not_local)
    candidates = [e for e in not_local if e[0] not in known]

    if candidates:
        mirror = jcs.mirror_persistent_cache_to_local(
            candidates, cache_dir, racaf_cache_dir, persistent_cache_dir,
            persistent_racaf_cache_dir, image_size=image_size)
        if mirror["drive_unreachable"] or mirror["corrupt"]:
            raise RuntimeError(f"Mirroring from the persistent cache did not complete "
                               f"(drive_unreachable={mirror['drive_unreachable']}, "
                               f"corrupt={mirror['corrupt']}). Nothing was generated or trained.")
        report["mirrored_files"] = mirror["copied"]

    to_generate = incomplete(candidates)
    newly_empty = set()
    if to_generate:
        staged = jcs.stage_raw_images_for_uncached_entries(
            to_generate, cache_dir, racaf_cache_dir, source_image_dir, local_image_dir,
            image_size=image_size)
        if staged["missing_at_source"] or staged["drive_unreachable"]:
            raise RuntimeError(f"Raw-image staging for cache generation did not complete "
                               f"(missing at source: {staged['missing_at_source']}, "
                               f"drive_unreachable={staged['drive_unreachable']}).")
        report["raw_images_staged"] = staged["copied"]

        stats = jtd.precompute_joint_frozen_caches(
            to_generate, image_dir=local_image_dir, cache_dir=cache_dir,
            racaf_cache_dir=racaf_cache_dir, processed_dir=processed_dir, image_size=image_size,
            progress_every=0)
        newly_empty = set(stats["skipped_empty_fov"])

        for id_code, _diagnosis in to_generate:
            if id_code in newly_empty or missing_local_artifacts(id_code, cache_dir,
                                                                 racaf_cache_dir, image_size):
                continue
            report["generated_ids"].append(id_code)
            local = jcd.artifact_paths(id_code, cache_dir, racaf_cache_dir, image_size)
            persistent = jcd.artifact_paths(id_code, persistent_cache_dir,
                                            persistent_racaf_cache_dir, image_size)
            for artifact, source in local.items():
                if not os.path.exists(persistent[artifact]):
                    jcs._copy_raw_image(source, persistent[artifact])  # atomic, size-checked
                    report["persisted_files"] += 1

    still_missing = [e[0] for e in incomplete(entries)]
    unexplained = sorted(set(still_missing) - known - newly_empty)
    if unexplained:
        raise RuntimeError(f"{len(unexplained)} entr(y/ies) have no complete cache and are not "
                           f"empty-field-of-view: {unexplained[:20]}. Nothing was trained.")
    report["empty_fov_ids"] = sorted(still_missing)
    return report
