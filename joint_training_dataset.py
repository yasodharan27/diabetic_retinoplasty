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

Two-phase workflow (`JOINT_TRAINING_ARCHITECTURE.md` §32) -- fixes the first real T4 run's RAM
exhaustion:
  - Root cause: `_make_joint_dataset()` previously sized its `tf.data` shuffle buffer to the
    ENTIRE dataset (`buffer_size=len(entries)`, up to 2929) -- but each already-materialized
    sample here is a `(512,512,8)` + `(256,256,3)` float32 pair (~8.75 MB), so that buffer alone
    demanded ~25 GB before an epoch could even start, regardless of Stage 3/4 inference cost.
    Fixed: the shuffle buffer is now capped at `DEFAULT_SHUFFLE_BUFFER_SIZE`, a small, FIXED
    constant that never scales with dataset size.
  - `precompute_joint_frozen_caches()` / `precompute_authoritative_joint_caches()` (Phase 1) let
    Stage 03/04/RACAF's expensive, Drive-I/O-bound per-image inference run as its own pass,
    BEFORE `load_joint_training_datasets()` (Phase 2) ever builds a `tf.data` pipeline or starts
    `Trainer.fit()`. Phase 1 processes one image at a time -- no per-image array is ever held
    past its own loop iteration, no list of samples is ever accumulated -- and writes straight to
    the SAME persistent, on-disk cache Phase 2 already reads (`_get_or_compute_joint_frozen_outputs`,
    unchanged). Already-cached entries are skipped (that function's own existing check), so Phase
    1 is always safe to interrupt and re-run: no valid cache entry is ever recomputed or deleted.
    Phase 2 still works correctly on its own even if Phase 1 was never run (any still-uncached
    entry is computed on the fly, exactly as before) -- Phase 1 is a recommended optimization to
    decouple slow cache-building from the training loop, not a new requirement for correctness.

Persistent-cache existence check, not bulk copy (`JOINT_TRAINING_ARCHITECTURE.md` §35) -- fixes a
real Colab crash (`OSError: [Errno 107] Transport endpoint is not connected`) that persisted after
the fixes above: `precompute_joint_frozen_caches()`'s `persistent_cache_dir`/
`persistent_racaf_cache_dir` parameters let a caller point at a Drive-mounted, already-populated
cache directory from a PRIOR run -- an entry found there (via a cheap `os.path.exists` stat, the
same check already used for `cache_dir`/`racaf_cache_dir`) is treated as a cache hit, WITHOUT ever
copying its content anywhere. This is what makes it safe for a caller to write newly computed
entries only to a fast local `cache_dir` (never recomputing anything already cached on Drive)
instead of first bulk-pulling the entire existing Drive cache down to local disk "just in case" --
a real run with ~8900 already-cached files hit exactly this crash when a prior version of this
workflow did that bulk pull, immediately after another already-heavy concurrent Drive copy (dataset
staging). Nothing about the cache CONTENT, format, or write path changes: a miss is still computed
and written to `cache_dir`/`racaf_cache_dir` exactly as before; only the caller decides how (and
whether) to sync the local cache_dir back up to `persistent_cache_dir` afterward
(`dataset_staging.sync_missing_files`, unchanged, bounded to newly written files only).

Diagnostic mode (`max_images`, `verbose_diagnostics`) -- lets a caller run Phase 1 against the REAL
code path (real checkpoints, real images, real cache I/O) on a small, controlled number of entries
before committing to a full run, with per-image RSS/GPU-memory/timing visibility instead of relying
on Colab's own resource graphs. Purely observational: it never changes which entries are computed
or what gets written to the cache.
"""

import logging
import os
import time

import numpy as np
import tensorflow as tf
from skimage.transform import resize as sk_resize

import config
import downstream_split
import local_feature_extraction_dataset as lfed
import racaf
from training import check_gpu
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

# A FIXED cap, never `len(entries)` -- each already-materialized sample here is ~8.75 MB
# (`(512,512,8)` + `(256,256,3)` float32), so a shuffle buffer sized to the whole dataset (up to
# 2929) previously demanded ~25 GB and was the actual cause of the first real T4 run's RAM
# exhaustion (see module docstring's "Two-phase workflow" section). 256 samples (~2.2 GB) gives
# meaningful shuffling while staying well within a T4 Colab runtime's memory budget regardless of
# how large the dataset itself is.
DEFAULT_SHUFFLE_BUFFER_SIZE = 256


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
                                          image_size=STAGE5_IMAGE_SIZE, known_not_all_cached=False):
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

    `known_not_all_cached=True` lets a caller that has ALREADY verified (via its own
    `os.path.exists` checks) that at least one of the three cache files is missing skip this
    function's own identical upfront check -- a pure I/O-count optimization for a caller
    (`precompute_joint_frozen_caches`) that iterates thousands of images against a Drive-mounted
    cache directory, where every avoided `os.path.exists` round trip matters. It changes no
    computed value, and no other existing caller passes it (defaults to the original, unchanged
    behavior). The three per-file existence guards below -- which protect a partially-populated
    cache left by a prior interrupted run from being needlessly overwritten -- are UNCHANGED and
    still run either way; this only skips the redundant "are all three already done" recheck.
    """
    all_cached = not known_not_all_cached and (
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


# --- Phase 1: cache precomputation, decoupled from Phase 2 (tf.data construction / training) ---

def _cache_entry_paths(id_code, cache_dir, racaf_cache_dir, image_size):
    vessel_cache = lfed._cache_path(cache_dir, id_code, "vessel", image_size)
    lesion_cache = lfed._cache_path(cache_dir, id_code, "lesion", image_size)
    reliability_cache = racaf.reliability_cache_path(racaf_cache_dir, id_code)
    return vessel_cache, lesion_cache, reliability_cache


def _cache_entry_exists(id_code, cache_dir, racaf_cache_dir, image_size):
    """True iff ALL THREE of `id_code`'s cache files already exist under `cache_dir`/
    `racaf_cache_dir` -- an existence-only check (`os.path.exists`, a stat call), never reading
    or copying any file's content. Shared by `precompute_joint_frozen_caches`'s own `cache_dir`
    check and its `persistent_cache_dir` check (see that parameter's docstring) so both use
    identically-defined "is this cached" logic."""
    vessel_cache, lesion_cache, reliability_cache = _cache_entry_paths(
        id_code, cache_dir, racaf_cache_dir, image_size,
    )
    return (
        os.path.exists(vessel_cache)
        and os.path.exists(lesion_cache)
        and os.path.exists(reliability_cache)
    )


def _current_process_rss_mb():
    """Current process RSS in MB via `psutil`, or `None` if `psutil` is unavailable -- never
    raises. Diagnostic-only: never called on the real (non-diagnostic) Phase 1 path."""
    try:
        import psutil
    except ImportError:
        return None
    try:
        return psutil.Process(os.getpid()).memory_info().rss / (1024.0 * 1024.0)
    except Exception:  # noqa: BLE001 -- diagnostic reporting must never crash a real run
        return None


def _current_gpu_memory_used_mb():
    """Current TensorFlow-reported GPU memory usage in MB (`tf.config.experimental.get_memory_info`),
    or `None` if no GPU is visible or the query is unsupported -- never raises. Reports only
    TensorFlow's own view (Stage 04); PyTorch's independent CUDA allocator (Stage 03) is not
    visible through this call, so a `None`/low value here does not by itself mean total GPU usage
    is low. Diagnostic-only: never called on the real (non-diagnostic) Phase 1 path."""
    try:
        if not tf.config.list_physical_devices("GPU"):
            return None
        info = tf.config.experimental.get_memory_info("GPU:0")
        return info["current"] / (1024.0 * 1024.0)
    except Exception:  # noqa: BLE001 -- diagnostic reporting must never crash a real run
        return None


def precompute_joint_frozen_caches(entries, image_dir=DEFAULT_TRAIN_IMAGE_DIR, cache_dir=DEFAULT_CACHE_DIR,
                                    racaf_cache_dir=DEFAULT_RACAF_CACHE_DIR,
                                    persistent_cache_dir=None, persistent_racaf_cache_dir=None,
                                    vessel_model=None,
                                    vessel_model_path=DEFAULT_VESSEL_MODEL_PATH, stage4_model=None,
                                    processed_dir=DEFAULT_PROCESSED_DIR, image_size=STAGE5_IMAGE_SIZE,
                                    progress_every=50, max_images=None, verbose_diagnostics=False):
    """Phase 1 of the two-phase joint training workflow (module docstring; `JOINT_TRAINING_
    ARCHITECTURE.md` §32/§33): populates every entry's Stage 03/04 + RACAF on-disk cache, ONE
    IMAGE AT A TIME, with no `tf.data` pipeline involved at all -- this is the expensive, CPU/
    Drive-I/O-bound half of joint training, deliberately separable from `load_joint_training_
    datasets()` (Phase 2) so it can be run, interrupted, and resumed independently of the actual
    `.fit()` call.

    Bounded memory: only ONE image's `(vessel_map, lesion_maps, kappa, r)` ever exists at a time
    -- each is discarded (via `_get_or_compute_joint_frozen_outputs`'s own cache-write and this
    function's loop moving on) before the next entry starts. Nothing per-image is accumulated
    into a list or returned; only small, fixed-size integer/string bookkeeping is.

    Resumable / cache-safe: an entry whose vessel/lesion/reliability cache files ALL already exist
    is skipped without touching `vessel_model`/`stage4_model` at all -- so interrupting this
    function (a Colab disconnect, a manual stop) and re-running it later never recomputes or
    deletes a valid cache entry; it only fills in whatever is still missing. That check happens
    exactly ONCE per image here (not twice): `_get_or_compute_joint_frozen_outputs` is called with
    `known_not_all_cached=True` in the "still needs computing" branch below, since this function
    has already established that at least one of the three cache files is missing -- see that
    parameter's docstring for why this specific redundancy (measured as a real contributor to
    Phase 1's Drive-round-trip count -- `JOINT_TRAINING_ARCHITECTURE.md` §33) was worth removing.

    `EmptyFieldOfViewError` (see that class's docstring, `vessel_segmentation_inference.py`) is
    caught per-image here too, logged, and skipped -- identical to `_make_joint_dataset`'s
    generator -- so one unprocessable image never blocks precomputing every other image's cache.

    Progress: every `progress_every` images (and once at the end), logs processed/cache-hit/
    skipped counts alongside elapsed wall-clock time and a rolling images-per-minute rate -- so a
    long Colab run's actual throughput is visible without guessing.

    `persistent_cache_dir`/`persistent_racaf_cache_dir` (both default `None`): an ADDITIONAL cache
    location checked for an existing entry -- via the SAME `os.path.exists`-only check used for
    `cache_dir`/`racaf_cache_dir`, never a content read or copy -- when the entry is not found
    under `cache_dir`/`racaf_cache_dir` itself. Intended use: `cache_dir`/`racaf_cache_dir` point
    at a fast local directory this run writes NEW entries to, while `persistent_cache_dir`/
    `persistent_racaf_cache_dir` point at the real, Drive-mounted, cross-session persistent cache
    -- so an image already cached from a PRIOR run is correctly skipped (never recomputed) WITHOUT
    first bulk-copying the entire persistent cache down to local disk (`JOINT_TRAINING_
    ARCHITECTURE.md` §35 -- that bulk-copy pattern is what caused a real Colab run to crash with
    `OSError: [Errno 107] Transport endpoint is not connected`, from thousands of concurrent Drive
    file opens). A miss (not found in either location) is always computed and written to
    `cache_dir`/`racaf_cache_dir` only -- `persistent_cache_dir`/`persistent_racaf_cache_dir` are
    never written to by this function. Leaving both at `None` (the default) reproduces the exact
    prior behavior: only `cache_dir`/`racaf_cache_dir` are ever checked or written.

    `max_images` (default `None`): if given, only the first `max_images` of `entries` are
    processed -- lets a caller run this function's REAL code path (real models, real images, real
    cache I/O) against a small, controlled number of images before committing to a full run,
    without altering which entries would be computed in a full run or what gets written to the
    cache for them.

    `verbose_diagnostics` (default `False`): if `True`, logs one line per image (in addition to
    the periodic `progress_every` summary) reporting its cache status, this image's own elapsed
    time, the running images/minute rate, current process RSS (via `psutil`, if installed), and
    current TensorFlow-reported GPU memory usage (if a GPU is visible) -- exactly the per-image
    visibility needed to tell a genuine memory-growth trend apart from normal one-time model-load
    overhead on a real run, without accumulating any of that data into `stats` (each line is
    logged and discarded; the returned `stats` dict's shape is unaffected either way).

    Returns `{"cached": int, "already_cached": int, "skipped_empty_fov": [image_id, ...],
    "elapsed_seconds": float}`.
    """
    # Must run before ANY TensorFlow op touches the GPU (JOINT_TRAINING_ARCHITECTURE.md §34):
    # without this, TF's default allocator claims ~all free VRAM the instant Stage 04's model is
    # first called, starving Stage 03's PyTorch model (also CUDA, via
    # vessel_segmentation_model.resolve_device()) of the memory it needs -- the two frameworks use
    # completely independent CUDA allocators that do not coordinate. `check_gpu()` (already used
    # by `train_image_quality.py`/`colab/common/environment.py`, reused here rather than
    # reimplemented) is a no-op on CPU-only machines and safe to call repeatedly.
    check_gpu()

    entries = list(entries)
    if max_images is not None:
        entries = entries[:max_images]
    resolved_vessel_model = vessel_model if vessel_model is not None else load_vessel_model(vessel_model_path)
    resolved_stage4_model = stage4_model if stage4_model is not None else racaf.load_frozen_stage4_model()

    stats = {"cached": 0, "already_cached": 0, "skipped_empty_fov": []}
    start_time = time.monotonic()

    def _log_progress(count):
        elapsed = time.monotonic() - start_time
        images_per_minute = (count / elapsed) * 60.0 if elapsed > 0 else 0.0
        logger.info(
            "Cache precomputation: %d/%d images processed (%d newly cached, %d already cached, "
            "%d skipped) -- %.1fs elapsed, %.1f images/min.",
            count, len(entries), stats["cached"], stats["already_cached"],
            len(stats["skipped_empty_fov"]), elapsed, images_per_minute,
        )

    def _log_diagnostic(i, id_code, status, image_elapsed):
        if not verbose_diagnostics:
            return
        elapsed = time.monotonic() - start_time
        images_per_minute = ((i + 1) / elapsed) * 60.0 if elapsed > 0 else 0.0
        rss_mb = _current_process_rss_mb()
        gpu_mb = _current_gpu_memory_used_mb()
        logger.info(
            "[diagnostic] %d/%d id=%s status=%s image_elapsed=%.2fs total_elapsed=%.1fs "
            "images/min=%.1f rss_mb=%s gpu_used_mb=%s",
            i + 1, len(entries), id_code, status, image_elapsed, elapsed, images_per_minute,
            f"{rss_mb:.1f}" if rss_mb is not None else "n/a",
            f"{gpu_mb:.1f}" if gpu_mb is not None else "n/a",
        )

    for i, (id_code, _diagnosis) in enumerate(entries):
        image_start = time.monotonic()
        already_cached = _cache_entry_exists(id_code, cache_dir, racaf_cache_dir, image_size)
        if not already_cached and persistent_cache_dir is not None:
            already_cached = _cache_entry_exists(
                id_code, persistent_cache_dir, persistent_racaf_cache_dir, image_size,
            )

        if already_cached:
            stats["already_cached"] += 1
            _log_diagnostic(i, id_code, "already_cached", time.monotonic() - image_start)
        else:
            vessel_cache, lesion_cache, reliability_cache = _cache_entry_paths(
                id_code, cache_dir, racaf_cache_dir, image_size,
            )
            raw_bgr = lfed._load_raw_bgr(image_dir, id_code)
            rgb_native = lfed._resolve_processed_rgb(raw_bgr, processed_dir, id_code)
            try:
                _get_or_compute_joint_frozen_outputs(
                    rgb_native, vessel_cache, lesion_cache, reliability_cache,
                    resolved_vessel_model, resolved_stage4_model, image_size=image_size,
                    known_not_all_cached=True,
                )
                stats["cached"] += 1
                _log_diagnostic(i, id_code, "cached", time.monotonic() - image_start)
            except EmptyFieldOfViewError:
                logger.warning(
                    "Skipping image_id=%s during cache precomputation: empty Stage 03 "
                    "field-of-view (no fundus disk detected).", id_code,
                )
                stats["skipped_empty_fov"].append(id_code)
                _log_diagnostic(i, id_code, "skipped_empty_fov", time.monotonic() - image_start)

        if progress_every and (i + 1) % progress_every == 0:
            _log_progress(i + 1)

    if progress_every and len(entries) % progress_every != 0:
        _log_progress(len(entries))
    stats["elapsed_seconds"] = time.monotonic() - start_time
    return stats


def precompute_authoritative_joint_caches(csv_path=DEFAULT_TRAIN_CSV, image_dir=DEFAULT_TRAIN_IMAGE_DIR,
                                           cache_dir=DEFAULT_CACHE_DIR, racaf_cache_dir=DEFAULT_RACAF_CACHE_DIR,
                                           persistent_cache_dir=None, persistent_racaf_cache_dir=None,
                                           vessel_model=None, vessel_model_path=DEFAULT_VESSEL_MODEL_PATH,
                                           stage4_model=None, val_split=DEFAULT_VAL_SPLIT, seed=DEFAULT_SEED,
                                           processed_dir=DEFAULT_PROCESSED_DIR, progress_every=50,
                                           max_images=None, verbose_diagnostics=False):
    """Phase 1 entry point for the real training workflow -- precomputes caches for BOTH halves of
    the SAME authoritative split (`split_train_val_ids`) `load_joint_training_datasets()` (Phase
    2) reads, never a second one. Loads `vessel_model`/`stage4_model` at most once each, regardless
    of how many of the combined 3662 entries still need computing. `persistent_cache_dir`/
    `persistent_racaf_cache_dir`/`max_images`/`verbose_diagnostics` are forwarded unchanged to
    `precompute_joint_frozen_caches` -- see that function's docstring."""
    check_gpu()  # before loading either model -- see precompute_joint_frozen_caches's identical call

    train_entries, val_entries = split_train_val_ids(csv_path, val_split=val_split, seed=seed)
    resolved_vessel_model = vessel_model if vessel_model is not None else load_vessel_model(vessel_model_path)
    resolved_stage4_model = stage4_model if stage4_model is not None else racaf.load_frozen_stage4_model()
    return precompute_joint_frozen_caches(
        train_entries + val_entries, image_dir=image_dir, cache_dir=cache_dir, racaf_cache_dir=racaf_cache_dir,
        persistent_cache_dir=persistent_cache_dir, persistent_racaf_cache_dir=persistent_racaf_cache_dir,
        vessel_model=resolved_vessel_model, stage4_model=resolved_stage4_model,
        processed_dir=processed_dir, progress_every=progress_every,
        max_images=max_images, verbose_diagnostics=verbose_diagnostics,
    )


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
        # A FIXED cap (never len(entries)) -- see DEFAULT_SHUFFLE_BUFFER_SIZE's comment. Sizing
        # this to the full dataset is what exhausted Colab's RAM on the first real T4 run: tf.data
        # holds `buffer_size` fully-materialized samples at once, not just their file paths.
        buffer_size = max(1, min(len(entries), DEFAULT_SHUFFLE_BUFFER_SIZE))
        ds = ds.shuffle(buffer_size=buffer_size, seed=seed, reshuffle_each_iteration=True)
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
