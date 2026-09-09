"""
One-time, controlled mirror of the PERSISTENT (Drive) joint-training cache onto local SSD.

    Google Drive persistent cache
              |  one-time controlled copy (this module)
              v
    /content/cache/local_feature_extraction
    /content/cache/racaf
              |
              v
    training / profiler -> GPU

Why this exists rather than reusing something already in the repo:

  * `dataset_staging.sync_missing_files()` is a directory-walking, 16-thread copy. Its own module
    docstring explicitly forbids using it to bulk-PULL a Drive cache: that is the exact load shape
    that produced `OSError: [Errno 107] Transport endpoint is not connected` in a real run
    (`JOINT_TRAINING_ARCHITECTURE.md` Sec 35). It is the right tool for the small, session-bounded
    PUSH of newly computed entries back to Drive, and the wrong one here.
  * Phase 1 (`precompute_joint_frozen_caches`) does already mirror persistent entries locally, one
    image at a time, and would work -- but it loads the frozen Stage 03 and Stage 04 models into
    GPU memory first, which a pure file copy never needs, and its mirror writes land via a plain
    `np.save` straight onto the final filename. An interrupted write therefore leaves a truncated
    `.npy` at the real path, which every later `os.path.exists` check reads as a valid cache hit
    (Sec 42). This module writes through a temp file and renames.

What it guarantees:

  * Never writes to, deletes from, renames or "repairs" anything under the persistent cache. The
    persistent side is opened read-only, and the only paths ever written are under the local dirs.
  * Never recomputes a frozen artifact. There is no model here at all -- an entry that is missing
    persistently is reported, never regenerated.
  * Distinguishes "absent" from "mount is down": persistent probes go through
    `jtd._persistent_exists`, so `ENOTCONN` raises `PersistentCacheUnavailableError` rather than
    reading as a miss. On a mount failure the mirror stops immediately rather than continuing to
    hammer a sick FUSE mount for thousands more files.
  * Copies atomically and validates before accepting: bytes to a temp file, size compared against
    the source, `np.load` of the temp file with a shape check via the project's own
    `jtd._validate_cached_array`, and only then `os.replace()` into the final name. A partially
    written file is never visible under a real cache filename.
  * Refuses to start when local free space is insufficient, using measured source sizes rather
    than an estimate.
  * Is resumable and idempotent: only files missing locally are copied, so re-running after an
    interruption picks up exactly where it stopped.

Copy strategy is SEQUENTIAL by default (`max_workers=1`). The goal here is correctness and mount
stability, not throughput: this runs once per runtime, and the one previous attempt to move this
many Drive files concurrently is what took the mount down. `max_workers` can be raised for a
healthy mount, and the trade is documented at that parameter.
"""

import os
import shutil
import time

import numpy as np

import joint_cache_diagnostics as jcd
import joint_training_dataset as jtd

# Local free space must exceed the copy by this much, so staging can never fill the disk the
# training run then needs for checkpoints and logs.
FREE_SPACE_MARGIN_BYTES = 2 * 1024 ** 3
FREE_SPACE_SAFETY_FACTOR = 1.05


class InsufficientLocalSpaceError(RuntimeError):
    """Refused to start: the measured source bytes do not fit in local free space with margin."""


def _free_bytes(path):
    resolved = os.path.abspath(path)
    while resolved and not os.path.exists(resolved):
        parent = os.path.dirname(resolved)
        if parent == resolved:
            break
        resolved = parent
    return shutil.disk_usage(resolved).free


def _persistent_size(path, artifact, id_code):
    """Source size in bytes, with the same absent-vs-unreachable distinction as every other
    persistent probe in this project."""
    try:
        return os.stat(path).st_size
    except FileNotFoundError:
        return None
    except OSError as error:
        raise jtd.PersistentCacheUnavailableError(artifact, id_code, path, error) from error


# =====================================================================
# 1. Plan -- measure before copying anything
# =====================================================================

def plan_mirror(entries, cache_dir, racaf_cache_dir, persistent_cache_dir,
                persistent_racaf_cache_dir, image_size=None, measure_sizes=True):
    """What would have to be copied, and whether it fits -- computed from real `os.stat` calls on
    the actual persistent files, never from a per-artifact size assumption.

    Classifies every (entry, artifact) into exactly one of:
      * already local          -- nothing to do
      * to copy                -- absent locally, present persistently
      * missing everywhere     -- absent in both; reported, NEVER regenerated here

    Stops at the first mount failure and returns what it has, with `drive_unreachable` set."""
    image_size = image_size if image_size is not None else jtd.STAGE5_IMAGE_SIZE
    entries = list(entries)
    # ONE directory listing per persistent cache dir replaces ~14,595 per-file Drive stats. A real
    # run measured ~1.0 s per Drive file operation, so per-file stat-ing the plan alone cost hours
    # before the first byte was copied. Sizes are derived from shape+dtype rather than stat'ed --
    # a `.npy` is exactly H*W*C*4 + 128 bytes, verified against real files (§43).
    import joint_cache_archive as _jca
    persistent_names = {
        "features": _jca.list_cache_dir(persistent_cache_dir),
        "racaf": _jca.list_cache_dir(persistent_racaf_cache_dir),
    }
    plan = {
        "entries": len(entries),
        "already_local_entries": 0,
        "to_copy": [],            # (id_code, artifact, source, destination, size_bytes)
        "missing_everywhere": [], # (id_code, [artifact, ...])
        "bytes_to_copy": 0,
        "bytes_by_artifact": {artifact: 0 for artifact in jcd.ARTIFACTS},
        "files_by_artifact": {artifact: 0 for artifact in jcd.ARTIFACTS},
        "drive_unreachable": False,
        "drive_error": None,
        "local_free_bytes": _free_bytes(cache_dir),
        "fits": None,
        "required_bytes_with_margin": None,
    }

    for id_code, _diagnosis in entries:
        local = jcd.artifact_paths(id_code, cache_dir, racaf_cache_dir, image_size)
        persistent = jcd.artifact_paths(id_code, persistent_cache_dir,
                                        persistent_racaf_cache_dir, image_size)
        missing_local = [a for a in jcd.ARTIFACTS if not os.path.exists(local[a])]
        if not missing_local:
            plan["already_local_entries"] += 1
            continue

        missing_both = []
        for artifact in missing_local:
            source = persistent[artifact]
            names = persistent_names["racaf"] if artifact == "reliability" else persistent_names["features"]
            if os.path.basename(source) not in names:
                missing_both.append(artifact)
                continue
            # Derived, never stat'ed -- see _jca.expected_artifact_bytes. The plan's byte total
            # is a space check, and the real size of every file is measured when it is copied.
            size = _jca.expected_artifact_bytes(artifact, image_size) or 0
            plan["to_copy"].append((id_code, artifact, source, local[artifact], size))
            plan["bytes_to_copy"] += size
            plan["bytes_by_artifact"][artifact] += size
            plan["files_by_artifact"][artifact] += 1
        if missing_both:
            plan["missing_everywhere"].append((id_code, missing_both))
    return _finalize_plan(plan)


def _finalize_plan(plan):
    required = int(plan["bytes_to_copy"] * FREE_SPACE_SAFETY_FACTOR) + FREE_SPACE_MARGIN_BYTES
    plan["required_bytes_with_margin"] = required
    plan["fits"] = plan["local_free_bytes"] >= required
    return plan


# =====================================================================
# 2. Copy -- atomically, with validation
# =====================================================================

def _copy_and_validate(source, destination, artifact, id_code, image_size):
    """One artifact, copied atomically and validated before it is allowed to take its real name.

    temp file -> size compared to source -> `np.load` + shape validation -> `os.replace`.

    A byte copy alone (what a size-checking `cp` gives) would not notice a file whose bytes
    arrived intact but whose array cannot be parsed; loading alone would not give a byte-identical
    local copy. Doing both gives a local file that is byte-for-byte the persistent one AND proven
    to load at the right shape -- the specific failure reported from a bad mount was
    `cannot reshape array of size 99296 into shape (512,512,4)`, which only the load catches."""
    os.makedirs(os.path.dirname(destination), exist_ok=True)
    temp = "%s.staging-%d.tmp" % (destination, os.getpid())
    try:
        try:
            source_size = os.stat(source).st_size
            shutil.copyfile(source, temp)
        except OSError as error:
            if getattr(error, "errno", None) in jtd._TRANSIENT_FUSE_ERRNOS:
                raise jtd.PersistentCacheUnavailableError(artifact, id_code, source, error) from error
            raise

        copied_size = os.stat(temp).st_size
        if copied_size != source_size:
            raise jtd.CorruptCacheFileError(
                artifact, id_code, source,
                "short read: copied %d of %d bytes" % (copied_size, source_size),
                persistent=True,
            )
        try:
            array = np.load(temp)
            try:
                jtd._validate_cached_array(array, artifact, id_code, source, image_size,
                                           persistent=True)
            finally:
                # `np.load` on an `.npz` returns a LAZY NpzFile that holds the file OPEN. Leaving
                # it open leaks a handle on POSIX and makes the `os.replace` below fail outright
                # on Windows, so the handle is closed before the rename either way.
                close = getattr(array, "close", None)
                if callable(close):
                    close()
        except jtd.CorruptCacheFileError:
            raise
        except Exception as error:  # noqa: BLE001 -- any parse failure means an unusable copy
            raise jtd.CorruptCacheFileError(artifact, id_code, source, repr(error),
                                            persistent=True) from error

        os.replace(temp, destination)  # atomic: the real filename only ever names a valid file
        return copied_size
    finally:
        if os.path.exists(temp):
            try:
                os.remove(temp)
            except OSError:
                pass


def mirror_persistent_cache_to_local(entries, cache_dir, racaf_cache_dir, persistent_cache_dir,
                                     persistent_racaf_cache_dir, image_size=None,
                                     plan=None, max_workers=1, progress_every=200,
                                     stop_on_corrupt=False):
    """Executes the plan. Returns a result dict; raises `InsufficientLocalSpaceError` before
    copying anything if the plan does not fit.

    `max_workers` (default 1 -- sequential): concurrency against the Drive FUSE mount. Left at 1
    deliberately. This is a once-per-runtime operation, and the previous attempt to move this many
    Drive files with a 16-thread pool is what produced `Errno 107`. Raise it only for a mount that
    has just been shown healthy, and understand the trade: throughput against the stability of the
    thing being read.

    `stop_on_corrupt` (default False): a single unreadable persistent file is recorded and skipped
    so one bad entry cannot block staging the other ~2900. Set True to stop at the first one.
    Either way the persistent file is never modified, and the entry simply stays a local miss.

    A mount failure (`PersistentCacheUnavailableError`) ALWAYS stops the run immediately --
    continuing would hammer a sick mount for thousands more files."""
    image_size = image_size if image_size is not None else jtd.STAGE5_IMAGE_SIZE
    if plan is None:
        plan = plan_mirror(entries, cache_dir, racaf_cache_dir, persistent_cache_dir,
                           persistent_racaf_cache_dir, image_size=image_size)
    result = {
        "planned_files": len(plan["to_copy"]),
        "copied": 0,
        "bytes_copied": 0,
        "skipped_already_present": 0,
        "corrupt": [],           # (id_code, artifact, path, detail)
        "drive_unreachable": False,
        "drive_error": None,
        "elapsed_seconds": 0.0,
        "plan": plan,
    }
    if plan["drive_unreachable"]:
        result["drive_unreachable"] = True
        result["drive_error"] = plan["drive_error"]
        return result
    if not plan["fits"]:
        raise InsufficientLocalSpaceError(
            "Refusing to stage: %.2f GiB needed (%.2f GiB of files + margin) but only %.2f GiB "
            "free at %s. Free space first -- nothing was copied."
            % (plan["required_bytes_with_margin"] / 1024 ** 3,
               plan["bytes_to_copy"] / 1024 ** 3,
               plan["local_free_bytes"] / 1024 ** 3, cache_dir))

    os.makedirs(cache_dir, exist_ok=True)
    os.makedirs(racaf_cache_dir, exist_ok=True)
    start = time.perf_counter()

    def stage_one(item):
        id_code, artifact, source, destination, _size = item
        if os.path.exists(destination):
            result["skipped_already_present"] += 1
            return 0
        return _copy_and_validate(source, destination, artifact, id_code, image_size)

    items = plan["to_copy"]
    if max_workers <= 1:
        for index, item in enumerate(items):
            try:
                copied = stage_one(item)
            except jtd.PersistentCacheUnavailableError as error:
                result["drive_unreachable"] = True
                result["drive_error"] = str(error)
                break
            except jtd.CorruptCacheFileError as error:
                result["corrupt"].append((item[0], item[1], item[2], str(error)))
                if stop_on_corrupt:
                    break
                continue
            if copied:
                result["copied"] += 1
                result["bytes_copied"] += copied
            if progress_every and (index + 1) % progress_every == 0:
                elapsed = time.perf_counter() - start
                print("  staged %d/%d files (%.2f GiB, %.1fs, %.1f files/s)"
                      % (index + 1, len(items), result["bytes_copied"] / 1024 ** 3,
                         elapsed, (index + 1) / elapsed if elapsed else 0.0))
    else:
        import concurrent.futures
        with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as pool:
            futures = {pool.submit(stage_one, item): item for item in items}
            for future in concurrent.futures.as_completed(futures):
                item = futures[future]
                try:
                    copied = future.result()
                except jtd.PersistentCacheUnavailableError as error:
                    result["drive_unreachable"] = True
                    result["drive_error"] = str(error)
                    for pending in futures:
                        pending.cancel()
                    break
                except jtd.CorruptCacheFileError as error:
                    result["corrupt"].append((item[0], item[1], item[2], str(error)))
                    continue
                if copied:
                    result["copied"] += 1
                    result["bytes_copied"] += copied

    result["elapsed_seconds"] = time.perf_counter() - start
    return result


# =====================================================================
# 3. Verify -- after mirroring
# =====================================================================

def verify_local_cache(entries, cache_dir, racaf_cache_dir, image_size=None,
                       persistent_cache_dir=None, persistent_racaf_cache_dir=None,
                       validate_contents=False, content_sample=0):
    """Post-mirror state. `validate_contents=False` (default) checks existence and file size only
    -- loading ~11,700 arrays would take minutes and buy nothing, since every file was already
    validated at copy time. `content_sample=N` additionally loads N entries' artifacts and checks
    their shapes, as a spot check."""
    image_size = image_size if image_size is not None else jtd.STAGE5_IMAGE_SIZE
    entries = list(entries)
    report = {
        "entries": len(entries),
        "fully_local": 0,
        "drive_fallback_required": 0,
        "missing_everywhere": 0,
        "corrupt_local": [],
        "empty_local_files": [],
        "artifact_counts": {artifact: 0 for artifact in jcd.ARTIFACTS},
        "local_bytes": 0,
        "content_checked": 0,
        "missing_everywhere_ids": [],
        "drive_fallback_ids": [],
    }
    have_persistent = persistent_cache_dir is not None and persistent_racaf_cache_dir is not None
    sampled = 0

    for id_code, _diagnosis in entries:
        local = jcd.artifact_paths(id_code, cache_dir, racaf_cache_dir, image_size)
        missing = []
        for artifact, path in local.items():
            if not os.path.exists(path):
                missing.append(artifact)
                continue
            report["artifact_counts"][artifact] += 1
            try:
                size = os.stat(path).st_size
            except OSError:
                missing.append(artifact)
                continue
            report["local_bytes"] += size
            if size == 0:
                report["empty_local_files"].append((id_code, artifact, path))

        if not missing:
            report["fully_local"] += 1
            if sampled < content_sample or validate_contents:
                sampled += 1
                report["content_checked"] += 1
                for artifact, path in local.items():
                    try:
                        array = np.load(path)
                        try:
                            jtd._validate_cached_array(array, artifact, id_code, path, image_size,
                                                       persistent=False)
                        finally:
                            close = getattr(array, "close", None)
                            if callable(close):
                                close()
                    except Exception as error:  # noqa: BLE001
                        report["corrupt_local"].append((id_code, artifact, path, repr(error)))
            continue

        if not have_persistent:
            report["missing_everywhere"] += 1
            report["missing_everywhere_ids"].append((id_code, missing))
            continue
        persistent = jcd.artifact_paths(id_code, persistent_cache_dir,
                                        persistent_racaf_cache_dir, image_size)
        try:
            recoverable = all(jtd._persistent_exists(persistent[a], a, id_code) for a in missing)
        except jtd.PersistentCacheUnavailableError:
            recoverable = False
        if recoverable:
            report["drive_fallback_required"] += 1
            if len(report["drive_fallback_ids"]) < 25:
                report["drive_fallback_ids"].append((id_code, missing))
        else:
            report["missing_everywhere"] += 1
            if len(report["missing_everywhere_ids"]) < 50:
                report["missing_everywhere_ids"].append((id_code, missing))
    return report


def sample_numerical_integrity(entries, cache_dir, racaf_cache_dir, persistent_cache_dir,
                               persistent_racaf_cache_dir, image_size=None, sample=5):
    """Loads a handful of staged artifacts alongside their persistent originals and compares them
    as arrays. Bounded on purpose -- never loads the whole cache into RAM."""
    image_size = image_size if image_size is not None else jtd.STAGE5_IMAGE_SIZE
    comparisons = []
    checked = 0
    for id_code, _diagnosis in list(entries):
        if checked >= sample:
            break
        local = jcd.artifact_paths(id_code, cache_dir, racaf_cache_dir, image_size)
        persistent = jcd.artifact_paths(id_code, persistent_cache_dir,
                                        persistent_racaf_cache_dir, image_size)
        if not all(os.path.exists(p) for p in local.values()):
            continue
        checked += 1
        for artifact in jcd.ARTIFACTS:
            entry = {"id_code": id_code, "artifact": artifact}
            try:
                if artifact == "reliability":
                    with np.load(local[artifact]) as local_npz:
                        local_array = np.array(local_npz["kappa"])
                    with np.load(persistent[artifact]) as persistent_npz:
                        persistent_array = np.array(persistent_npz["kappa"])
                else:
                    local_array = np.load(local[artifact])
                    persistent_array = np.load(persistent[artifact])
                difference = np.abs(local_array.astype(np.float64)
                                    - persistent_array.astype(np.float64))
                entry.update({
                    "shape": tuple(local_array.shape),
                    "dtype": str(local_array.dtype),
                    "shapes_match": local_array.shape == persistent_array.shape,
                    "dtypes_match": local_array.dtype == persistent_array.dtype,
                    "max_abs_diff": float(difference.max()) if difference.size else 0.0,
                    "mean_abs_diff": float(difference.mean()) if difference.size else 0.0,
                    "array_equal": bool(np.array_equal(local_array, persistent_array)),
                    "bytes_identical": (os.stat(local[artifact]).st_size
                                        == os.stat(persistent[artifact]).st_size),
                    "min": float(local_array.min()) if local_array.size else None,
                    "max": float(local_array.max()) if local_array.size else None,
                })
            except Exception as error:  # noqa: BLE001 -- report, never abort the check
                entry["error"] = repr(error)
            comparisons.append(entry)
    return comparisons


# =====================================================================
# 4. Raw APTOS images -- the MINIMUM the cache-backed path still needs
# =====================================================================
#
# With a complete local cache, `_build_joint_sample` never opens a raw image: its
# `if not (frozen_outputs_cached and rgb_cached)` guard short-circuits, so `lfed._load_raw_bgr`
# is not reached at all (JOINT_TRAINING_ARCHITECTURE.md Sec 44). Staging all 3,662 raw APTOS
# images is therefore ~9.5 GiB of local disk and ~3,663 Drive file opens spent on files nothing
# reads.
#
# The exception is an entry whose LOCAL cache is incomplete. For the known empty-FOV images that
# is permanent by design: Phase 1 caches their canonical RGB but cannot produce vessel/lesion/
# reliability, so every epoch re-enters the compute branch, reads the raw image, and Stage 03
# raises `EmptyFieldOfViewError`, which the generator catches and skips. Remove the raw image and
# that graceful skip becomes a hard `FileNotFoundError` -- the generator catches only
# `EmptyFieldOfViewError`, so the whole run dies on the first such entry (measured, Sec 44).
#
# So: stage the raw image for exactly the entries whose local cache is incomplete, and nothing
# else. That is a pure SCOPE reduction -- every entry still takes the identical code path it
# takes today, whether it ends in a cache hit, a recomputation, or an empty-FOV skip.

RAW_IMAGE_EXTENSION = ".png"


def entries_missing_local_cache(entries, cache_dir, racaf_cache_dir, image_size=None):
    """`[(id_code, [missing_artifact, ...]), ...]` for every entry that is NOT a complete local
    hit across all four artifacts -- i.e. exactly the entries whose sample build can still reach
    `lfed._load_raw_bgr`. Local `os.path.exists` only: no Drive path is touched, no model is
    loaded, nothing is written."""
    image_size = image_size if image_size is not None else jtd.STAGE5_IMAGE_SIZE
    incomplete = []
    for id_code, _diagnosis in entries:
        paths = jcd.artifact_paths(id_code, cache_dir, racaf_cache_dir, image_size)
        missing = [artifact for artifact, path in paths.items() if not os.path.exists(path)]
        if missing:
            incomplete.append((id_code, missing))
    return incomplete


def _copy_raw_image(source, destination, attempts=jtd._PERSISTENT_READ_ATTEMPTS,
                    base_delay=jtd._PERSISTENT_RETRY_BASE_DELAY_SECONDS):
    """Atomic, size-verified copy of one raw image, with the same bounded retry on transient FUSE
    errnos this module already uses for cache files. Same temp-then-`os.replace` convention as
    `_copy_and_validate` and `dataset_staging._copy_one`; no content validation, because a raw
    `.png` has no project-defined shape to check -- `cv2.imread` failing later is a real,
    surfaced error, not something to be pre-empted by decoding every file here."""
    os.makedirs(os.path.dirname(destination), exist_ok=True)
    temp = "%s.tmp-%d" % (destination, os.getpid())
    last_error = None
    for attempt in range(attempts):
        try:
            shutil.copyfile(source, temp)
            source_size = os.stat(source).st_size
            if os.path.getsize(temp) != source_size:
                raise OSError("copied size mismatch for %s" % source)
            os.replace(temp, destination)
            return source_size
        except OSError as error:
            last_error = error
            if os.path.exists(temp):
                try:
                    os.remove(temp)
                except OSError:
                    pass
            transient = getattr(error, "errno", None) in jtd._TRANSIENT_FUSE_ERRNOS
            if not transient or attempt == attempts - 1:
                raise
            time.sleep(base_delay * (2 ** attempt))
    raise last_error  # pragma: no cover -- the loop always returns or raises


def stage_raw_images_for_uncached_entries(entries, cache_dir, racaf_cache_dir, source_image_dir,
                                          local_image_dir, image_size=None,
                                          margin_bytes=FREE_SPACE_MARGIN_BYTES):
    """Copies `source_image_dir/<id>.png` -> `local_image_dir/<id>.png` for exactly the entries
    `entries_missing_local_cache` reports, and returns what it did.

    Read-only with respect to `source_image_dir`. Never deletes or overwrites an image already
    present locally. Never loads a model and never writes a cache file, so no frozen artifact can
    be regenerated here. An entry whose raw image is absent at the source is REPORTED, not
    fabricated and not silently dropped -- the training generator will raise for it exactly as it
    does today when a raw image is genuinely missing.

    Refuses to start (raising `InsufficientLocalSpaceError`) if the measured source bytes plus
    `margin_bytes` do not fit in local free space. The per-file `os.stat` on the source is
    affordable here precisely because this is a handful of files, not 14,595."""
    image_size = image_size if image_size is not None else jtd.STAGE5_IMAGE_SIZE
    entries = list(entries)
    incomplete = entries_missing_local_cache(entries, cache_dir, racaf_cache_dir, image_size)
    result = {
        "entries": len(entries),
        "entries_needing_raw": len(incomplete),
        "needing_raw_ids": [id_code for id_code, _ in incomplete],
        "copied": 0,
        "already_local": 0,
        "missing_at_source": [],
        "bytes_copied": 0,
        "source_bytes": 0,
        "elapsed_seconds": 0.0,
        "drive_unreachable": False,
        "drive_error": None,
        "local_image_dir": local_image_dir,
    }
    if not incomplete:
        return result

    os.makedirs(local_image_dir, exist_ok=True)
    pending = []
    for id_code, _missing in incomplete:
        destination = os.path.join(local_image_dir, id_code + RAW_IMAGE_EXTENSION)
        if os.path.exists(destination):
            result["already_local"] += 1
            continue
        source = os.path.join(source_image_dir, id_code + RAW_IMAGE_EXTENSION)
        try:
            size = os.stat(source).st_size
        except FileNotFoundError:
            result["missing_at_source"].append(id_code)
            continue
        except OSError as error:
            if getattr(error, "errno", None) in jtd._TRANSIENT_FUSE_ERRNOS:
                result["drive_unreachable"] = True
                result["drive_error"] = repr(error)
                return result
            raise
        result["source_bytes"] += size
        pending.append((id_code, source, destination))

    if pending:
        free = _free_bytes(local_image_dir)
        required = int(result["source_bytes"] * FREE_SPACE_SAFETY_FACTOR) + margin_bytes
        if free < required:
            raise InsufficientLocalSpaceError(
                "Refusing to stage %d raw image(s): %.2f GiB free, %.2f GiB required "
                "(%.2f GiB of images + %.2f GiB margin)."
                % (len(pending), free / 1024.0 ** 3, required / 1024.0 ** 3,
                   result["source_bytes"] / 1024.0 ** 3, margin_bytes / 1024.0 ** 3))

    start = time.perf_counter()
    for id_code, source, destination in pending:
        try:
            result["bytes_copied"] += _copy_raw_image(source, destination)
            result["copied"] += 1
        except OSError as error:
            if getattr(error, "errno", None) in jtd._TRANSIENT_FUSE_ERRNOS:
                result["drive_unreachable"] = True
                result["drive_error"] = repr(error)
                break
            raise
    result["elapsed_seconds"] = time.perf_counter() - start
    return result


# =====================================================================
# 5. Rendering
# =====================================================================

def _gib(num_bytes):
    return "n/a" if num_bytes is None else "%.2f GiB" % (num_bytes / float(1024 ** 3))


def print_plan(plan):
    print("=" * 78)
    print("LOCAL CACHE MIRROR PLAN (measured from the real persistent files)")
    print("=" * 78)
    print("  training entries              : %d" % plan["entries"])
    print("  already fully local           : %d" % plan["already_local_entries"])
    print("  files to copy                 : %d" % len(plan["to_copy"]))
    print("  entries missing persistently  : %d  (reported, never regenerated)"
          % len(plan["missing_everywhere"]))
    print("")
    print("  %-14s %10s %14s" % ("artifact", "files", "bytes"))
    for artifact in jcd.ARTIFACTS:
        print("  %-14s %10d %14s" % (artifact, plan["files_by_artifact"][artifact],
                                     _gib(plan["bytes_by_artifact"][artifact])))
    print("  %-14s %10d %14s" % ("TOTAL", len(plan["to_copy"]), _gib(plan["bytes_to_copy"])))
    print("")
    print("  local free space              : %s" % _gib(plan["local_free_bytes"]))
    print("  required (files + 5%% + 2 GiB) : %s" % _gib(plan["required_bytes_with_margin"]))
    print("  fits                          : %s" % ("YES" if plan["fits"] else "NO -- will refuse"))
    if plan["drive_unreachable"]:
        print("")
        print("  DRIVE UNREACHABLE while planning -- stopped early, nothing copied:")
        print("    %s" % plan["drive_error"])
    if plan["missing_everywhere"]:
        print("")
        print("  entries with no persistent artifact (first 20):")
        for id_code, artifacts in plan["missing_everywhere"][:20]:
            print("    %s -> %s" % (id_code, artifacts))


def print_verification(report):
    print("=" * 78)
    print("LOCAL CACHE VERIFICATION")
    print("=" * 78)
    print("  total training entries   : %d" % report["entries"])
    print("  fully local              : %d" % report["fully_local"])
    print("  Drive fallback required  : %d   <-- must be 0 before profiling"
          % report["drive_fallback_required"])
    print("  missing everywhere       : %d   <-- expected: the known empty-FOV entries"
          % report["missing_everywhere"])
    print("  corrupt local            : %d" % len(report["corrupt_local"]))
    print("  zero-byte local files    : %d" % len(report["empty_local_files"]))
    print("  local cache size         : %s" % _gib(report["local_bytes"]))
    print("  artifact counts          : %s" % report["artifact_counts"])
    if report["content_checked"]:
        print("  entries content-checked  : %d" % report["content_checked"])
    if report["missing_everywhere_ids"]:
        print("")
        print("  entries missing everywhere (these are NOT regenerated):")
        for id_code, artifacts in report["missing_everywhere_ids"]:
            print("    %s -> missing %s" % (id_code, artifacts))
    if report["drive_fallback_ids"]:
        print("")
        print("  entries that would still read Drive (first 25):")
        for id_code, artifacts in report["drive_fallback_ids"]:
            print("    %s -> %s" % (id_code, artifacts))
    for id_code, artifact, path, detail in report["corrupt_local"]:
        print("    CORRUPT LOCAL: %s %s %s -- %s" % (id_code, artifact, path, detail))


def print_integrity(comparisons):
    print("=" * 78)
    print("NUMERICAL INTEGRITY -- staged local copy vs persistent original")
    print("=" * 78)
    if not comparisons:
        print("  no fully-staged entry was available to compare")
        return
    print("  %-12s %-12s %-16s %-8s %12s %10s %8s"
          % ("image", "artifact", "shape", "dtype", "max|diff|", "equal", "bytes"))
    for entry in comparisons:
        if "error" in entry:
            print("  %-12s %-12s ERROR: %s" % (entry["id_code"], entry["artifact"], entry["error"]))
            continue
        print("  %-12s %-12s %-16s %-8s %12.3g %10s %8s"
              % (entry["id_code"][:12], entry["artifact"], str(entry["shape"]), entry["dtype"],
                 entry["max_abs_diff"], entry["array_equal"], entry["bytes_identical"]))
    mismatches = [e for e in comparisons if not e.get("array_equal", False) and "error" not in e]
    print("")
    print("  VERDICT: %s" % ("every sampled artifact is numerically identical to its persistent "
                             "original" if not mismatches
                             else "%d sampled artifact(s) DIFFER -- investigate before training"
                                  % len(mismatches)))


def print_result(result):
    print("=" * 78)
    print("MIRROR RESULT")
    print("=" * 78)
    print("  files planned            : %d" % result["planned_files"])
    print("  files copied             : %d" % result["copied"])
    print("  bytes copied             : %s" % _gib(result["bytes_copied"]))
    print("  already present, skipped : %d" % result["skipped_already_present"])
    print("  corrupt / unreadable     : %d" % len(result["corrupt"]))
    print("  elapsed                  : %.1fs (%.1f files/s)"
          % (result["elapsed_seconds"],
             result["copied"] / result["elapsed_seconds"] if result["elapsed_seconds"] else 0.0))
    if result["drive_unreachable"]:
        print("")
        print("  STOPPED -- Drive became unreachable mid-copy. Nothing was recomputed and the")
        print("  persistent cache was not modified. Re-mount Drive and re-run; already-staged")
        print("  files are kept, so this resumes where it stopped.")
        print("    %s" % result["drive_error"])
    for id_code, artifact, path, detail in result["corrupt"][:20]:
        print("    CORRUPT PERSISTENT (left untouched): %s %s %s" % (id_code, artifact, path))
        print("      %s" % detail)


def print_raw_image_staging(result):
    print("=" * 78)
    print("RAW APTOS IMAGES -- staged only for entries without a complete local cache")
    print("=" * 78)
    print("  split entries                 : %d" % result["entries"])
    print("  entries needing a raw image   : %d   <-- expected: the known empty-FOV entries"
          % result["entries_needing_raw"])
    print("  images copied this run        : %d (%.1f MiB in %.1fs)"
          % (result["copied"], result["bytes_copied"] / 1024.0 ** 2, result["elapsed_seconds"]))
    print("  already present locally       : %d" % result["already_local"])
    print("  local image dir               : %s" % result["local_image_dir"])
    if result["needing_raw_ids"]:
        print("  ids: %s" % ", ".join(result["needing_raw_ids"][:25]))
        if len(result["needing_raw_ids"]) > 25:
            print("       ... and %d more" % (len(result["needing_raw_ids"]) - 25))
    if result["missing_at_source"]:
        print("")
        print("  NOT FOUND at the source (left missing, never fabricated): %s"
              % ", ".join(result["missing_at_source"][:25]))
    if result["drive_unreachable"]:
        print("")
        print("  DRIVE UNREACHABLE mid-copy: %s" % result["drive_error"])
        print("  Nothing was recomputed and nothing on Drive was modified. Re-mount and re-run.")
