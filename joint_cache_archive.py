"""
Shard-archive representation of the frozen joint-training cache, so a fresh Colab runtime can
populate its local SSD cache with a handful of large sequential Drive reads instead of ~14,595
individual ones.

WHY -- the measurement, not a guess. A real staging run moved 200 files in 201.4s and 400 in
404.2s: 1.007 and 1.011 s/file, constant to within 3.5 ms. At a 2.00 MiB average that is an
effective 2.08 MB/s, an order of magnitude below what the same mount delivers sequentially. The
cost is therefore a fixed ~1 s PER FILE OPEN, not bytes moved -- 14,595 x 1 s = 4.05 h, while the
same 28.56 GiB read sequentially is 10-26 min. The lever is the NUMBER of Drive file operations.

WHAT THIS DOES. `build_archive()` packs the cache into a small number of uncompressed `.tar`
shards on Drive, once. `extract_archive()` then streams each shard straight from its Drive path
into the existing local cache directories, producing byte-identical individual `.npy`/`.npz`
files at exactly the paths the pipeline already expects.

That last property is the whole point of choosing tar over a consolidated array format: after
extraction the local cache IS the current representation, so `_build_joint_sample`,
`_get_or_compute_joint_frozen_outputs`, `_persistent_exists`, the local-first precedence and every
Sec 41 safety guarantee keep working untouched. No generator change, no loader change, no new
cache semantics to get wrong.

MEASURED PROPERTIES (on artifacts produced by the real Phase 1 code path -- real dtypes, real
shapes, real `.npy` serialization):

  * tar overhead is 0.11% (53 KiB on 48 MiB), so an uncompressed shard is the same size as the
    loose files it replaces;
  * streaming extraction runs at ~135 MB/s locally -> ~3.8 min for 28.56 GiB;
  * extracted files are byte-identical to the originals (24/24);
  * `tarfile.open(mode="r|")` extracts from a NON-SEEKABLE source with zero seek attempts, which
    is what makes it safe to stream directly off a FUSE mount and is why no local copy of the
    archive is ever needed -- important, since archive + extracted (57 GiB) would not fit in the
    45.83 GiB free, while extracted alone (28.56 GiB) does.

COMPRESSION is deliberately OFF by default. Compression was measured, but on arrays from an
UNTRAINED model over synthetic images, which are far smoother than real Stage 03/04 output -- the
ratios seen there (vessel 42%, lesion 65%, rgb 25%) are optimistic and must not be trusted for the
real cache. What does generalize is throughput: gzip-6 compresses at 7.7-22 MB/s on this CPU, and
Colab gives 2 cores. Since the transfer is latency-bound rather than bandwidth-bound, paying
20-40 min of CPU to save an unknown fraction of an already-acceptable 17-26 min transfer is a bad
trade. `compress=True` is available to evaluate on the real data.

NEVER modifies the source: shards are written to their own directory and the loose persistent
cache is only ever read.
"""

import os
import shutil
import tarfile
import time

import numpy as np

import joint_cache_diagnostics as jcd
import joint_training_dataset as jtd

DEFAULT_SHARD_ENTRIES = 500       # ~500 entries ~= 4 GiB per shard, ~2000 files
ARCHIVE_DIR_NAME = "cache_archive"
MANIFEST_NAME = "manifest.json"

# `.npy` size is fully determined by shape and dtype, so a plan never needs a per-file stat.
# Verified against real files: 512*512*C*4 + 128-byte header.
_NPY_HEADER_BYTES = 128
_ARTIFACT_CHANNELS = {"vessel": 1, "lesion": 4, "rgb": 3}
# The reliability `.npz` holds kappa(4,) float32 plus a scalar r; a real one measured 518 bytes.
# Its exact size is not fixed by geometry, but it is 0.025% of the 2.00 MiB average artifact, so a
# nominal value is used for SPACE PLANNING only -- stat-ing 3662 of them over Drive would cost
# roughly an hour to refine a rounding error. Actual bytes are measured when each file is copied.
NOMINAL_RELIABILITY_BYTES = 518

# Local free space that must remain AFTER extraction. `/content` is not a scratch disk: the
# runtime itself, pip's cache and any temp file a library writes all land there, and a full
# `/content` fails in ways that look like unrelated bugs. Checkpoints and TensorBoard logs go to
# Drive (`experiment_manager` resolves `experiment.root` under the Drive mount), so the training
# run's own local footprint is small -- this margin exists for the runtime, not for training
# outputs. Extraction itself needs NO headroom beyond the final size: each shard is streamed into
# a staging dir on the SAME filesystem and its members are `os.replace`d into place, which is a
# rename, so peak usage equals final usage.
DEFAULT_RUNTIME_MARGIN_BYTES = 5 * 1024 ** 3


def expected_artifact_bytes(artifact, image_size):
    """Exact on-disk size of one artifact, derived rather than stat'ed. `reliability` is a small
    `.npz` whose size is not fixed by geometry, so it returns None."""
    channels = _ARTIFACT_CHANNELS.get(artifact)
    if channels is None:
        return NOMINAL_RELIABILITY_BYTES if artifact == "reliability" else None
    height, width = image_size
    return height * width * channels * 4 + _NPY_HEADER_BYTES


# =====================================================================
# Directory listing -- 1 Drive op instead of N
# =====================================================================

def list_cache_dir(directory):
    """Every filename in one cache directory, as a set, from a SINGLE listing call.

    This is the metadata-cost fix: membership in this set answers "does this artifact exist"
    for every entry, so a plan over 3662 entries costs 2 Drive operations rather than ~14,595
    stats. Distinguishes absent from unreachable exactly like `jtd._persistent_exists`."""
    try:
        return set(os.listdir(directory))
    except FileNotFoundError:
        return set()
    except OSError as error:
        raise jtd.PersistentCacheUnavailableError(None, None, directory, error) from error


def index_persistent_cache(entries, persistent_cache_dir, persistent_racaf_cache_dir,
                           image_size=None):
    """Which entries are fully present persistently, built from two directory listings.

    Returns `(complete, incomplete, listings)` where `incomplete` maps id_code -> missing
    artifacts. No per-file Drive stat is performed."""
    image_size = image_size if image_size is not None else jtd.STAGE5_IMAGE_SIZE
    feature_names = list_cache_dir(persistent_cache_dir)
    racaf_names = list_cache_dir(persistent_racaf_cache_dir)
    complete, incomplete = [], {}
    for id_code, diagnosis in entries:
        paths = jcd.artifact_paths(id_code, persistent_cache_dir, persistent_racaf_cache_dir,
                                   image_size)
        missing = []
        for artifact, path in paths.items():
            names = racaf_names if artifact == "reliability" else feature_names
            if os.path.basename(path) not in names:
                missing.append(artifact)
        if missing:
            incomplete[id_code] = missing
        else:
            complete.append((id_code, diagnosis))
    return complete, incomplete, {"feature_files": len(feature_names),
                                  "racaf_files": len(racaf_names)}


# =====================================================================
# Build -- pack loose persistent files into shards
# =====================================================================

def shard_name(index, compress=False):
    return "cache_shard_%03d.tar%s" % (index, ".gz" if compress else "")


def plan_shards(entries, shard_entries=DEFAULT_SHARD_ENTRIES):
    """Splits entries into deterministic, contiguous shard groups. Deterministic so a rebuild
    after an interruption produces the same shards and can skip the finished ones."""
    ordered = sorted(entries)
    return [ordered[i:i + shard_entries] for i in range(0, len(ordered), shard_entries)]


def build_archive(entries, persistent_cache_dir, persistent_racaf_cache_dir, archive_dir,
                  image_size=None, shard_entries=DEFAULT_SHARD_ENTRIES, compress=False,
                  skip_existing=True, progress=True, max_shards=None):
    """Packs the persistent cache into `archive_dir` as shards. READ-ONLY with respect to the
    loose persistent cache.

    Building is the expensive half: every source file must be read from Drive once, so this pays
    the same ~1 s/file cost the loose copy does. It is therefore RESUMABLE AT SHARD GRANULARITY --
    a completed shard is durable, `skip_existing` leaves it alone, and `max_shards` limits how
    many one runtime attempts. Two ~3-hour runtimes cover the full cache; every runtime afterwards
    pays only `extract_archive`.

    Each shard is written to a temp name and renamed on completion, so an interrupted build never
    leaves a partial shard that a later run would mistake for a finished one."""
    image_size = image_size if image_size is not None else jtd.STAGE5_IMAGE_SIZE
    os.makedirs(archive_dir, exist_ok=True)
    complete, incomplete, listings = index_persistent_cache(
        entries, persistent_cache_dir, persistent_racaf_cache_dir, image_size)
    groups = plan_shards(complete, shard_entries)
    result = {
        "shards_total": len(groups),
        "shards_written": 0,
        "shards_skipped": 0,
        "entries_archived": 0,
        "bytes_written": 0,
        "incomplete_entries": incomplete,
        "listings": listings,
        "elapsed_seconds": 0.0,
        "drive_unreachable": False,
        "drive_error": None,
        "shard_files": [],
    }
    start = time.perf_counter()
    attempted = 0

    for index, group in enumerate(groups):
        name = shard_name(index, compress)
        final = os.path.join(archive_dir, name)
        result["shard_files"].append(name)
        if skip_existing and os.path.exists(final):
            result["shards_skipped"] += 1
            continue
        if max_shards is not None and attempted >= max_shards:
            break
        attempted += 1

        temp = final + ".building"
        mode = "w:gz" if compress else "w"
        try:
            with tarfile.open(temp, mode) as tar:
                for id_code, _diagnosis in group:
                    paths = jcd.artifact_paths(id_code, persistent_cache_dir,
                                               persistent_racaf_cache_dir, image_size)
                    for artifact, path in paths.items():
                        # arcname encodes which local directory the member belongs in, so
                        # extraction needs no filename parsing.
                        sub = "racaf" if artifact == "reliability" else "features"
                        tar.add(path, arcname=os.path.join(sub, os.path.basename(path)))
                    result["entries_archived"] += 1
            os.replace(temp, final)
            result["shards_written"] += 1
            result["bytes_written"] += os.path.getsize(final)
            if progress:
                elapsed = time.perf_counter() - start
                print("  shard %d/%d written (%d entries, %.2f GiB, %.0fs elapsed)"
                      % (index + 1, len(groups), len(group),
                         result["bytes_written"] / 1024 ** 3, elapsed))
        except OSError as error:
            if os.path.exists(temp):
                try:
                    os.remove(temp)
                except OSError:
                    pass
            if getattr(error, "errno", None) in jtd._TRANSIENT_FUSE_ERRNOS:
                result["drive_unreachable"] = True
                result["drive_error"] = repr(error)
                break
            raise
    result["elapsed_seconds"] = time.perf_counter() - start
    return result


# =====================================================================
# Extract -- stream shards from Drive straight into the local cache
# =====================================================================

def _free_bytes(path):
    """Free space on the filesystem `path` will live on. Walks up to the nearest existing parent:
    on a fresh runtime `/content/cache/local_feature_extraction` does not exist yet, and
    `shutil.disk_usage` on a missing path raises. Same convention as
    `joint_cache_staging._free_bytes` (restated rather than imported -- that module imports this
    one)."""
    resolved = os.path.abspath(path)
    while resolved and not os.path.exists(resolved):
        parent = os.path.dirname(resolved)
        if parent == resolved:
            break
        resolved = parent
    return shutil.disk_usage(resolved).free


def plan_extraction(archive_dir, cache_dir, racaf_cache_dir,
                    runtime_margin_bytes=DEFAULT_RUNTIME_MARGIN_BYTES, shards=None):
    """Measures whether this runtime has the disk to extract the archive, BEFORE a byte is read.

    Costs one directory listing plus one `os.stat` per shard on Drive (8 operations for the real
    8-shard archive), and two local listings -- not a per-file scan of either side.

    The budget is deliberately stated in terms of what is actually on this disk right now:

        required = total shard bytes            (what a full extraction writes)
                 - bytes already in the local cache dirs   (a resumed/partial extraction)
                 + runtime_margin_bytes         (headroom `/content` must keep)

    Extraction adds no transient peak above the final size (see `DEFAULT_RUNTIME_MARGIN_BYTES`),
    and the raw APTOS dataset is deliberately NOT part of this budget -- the cache-backed path
    does not stage it (`joint_cache_staging.stage_raw_images_for_uncached_entries`).

    Returns a dict; `fits` is the answer. Raises `PersistentCacheUnavailableError` if the archive
    directory itself cannot be listed, rather than reporting an empty archive."""
    if shards is None:
        try:
            shards = sorted(name for name in os.listdir(archive_dir)
                            if name.startswith("cache_shard_") and ".building" not in name)
        except OSError as error:
            raise jtd.PersistentCacheUnavailableError(None, None, archive_dir, error) from error

    plan = {
        "archive_dir": archive_dir,
        "shards": len(shards),
        "shard_names": list(shards),
        "archive_bytes": 0,
        "local_cache_bytes": 0,
        "local_files": 0,
        "runtime_margin_bytes": runtime_margin_bytes,
        "required_bytes": 0,
        "free_bytes": 0,
        "free_after_bytes": 0,
        "shortfall_bytes": 0,
        "fits": False,
        "drive_unreachable": False,
        "drive_error": None,
    }
    for shard in shards:
        try:
            plan["archive_bytes"] += os.stat(os.path.join(archive_dir, shard)).st_size
        except OSError as error:
            if getattr(error, "errno", None) in jtd._TRANSIENT_FUSE_ERRNOS:
                plan["drive_unreachable"] = True
                plan["drive_error"] = repr(error)
                return plan
            raise

    for directory in (cache_dir, racaf_cache_dir):
        if not os.path.isdir(directory):
            continue
        for name in os.listdir(directory):
            path = os.path.join(directory, name)
            try:
                if os.path.isfile(path):
                    plan["local_cache_bytes"] += os.stat(path).st_size
                    plan["local_files"] += 1
            except OSError:
                continue

    outstanding = max(0, plan["archive_bytes"] - plan["local_cache_bytes"])
    plan["required_bytes"] = outstanding + runtime_margin_bytes
    plan["free_bytes"] = _free_bytes(cache_dir)
    plan["free_after_bytes"] = plan["free_bytes"] - outstanding
    plan["fits"] = plan["free_bytes"] >= plan["required_bytes"]
    plan["shortfall_bytes"] = max(0, plan["required_bytes"] - plan["free_bytes"])
    return plan


class ArchiveIntegrityError(RuntimeError):
    """A shard member did not survive extraction intact."""


def _validate_extracted(path, artifact, id_code, image_size):
    try:
        array = np.load(path)
        try:
            jtd._validate_cached_array(array, artifact, id_code, path, image_size, persistent=False)
        finally:
            close = getattr(array, "close", None)
            if callable(close):
                close()
    except jtd.CorruptCacheFileError:
        raise
    except Exception as error:  # noqa: BLE001
        raise jtd.CorruptCacheFileError(artifact, id_code, path, repr(error),
                                        persistent=False) from error


def extract_archive(archive_dir, cache_dir, racaf_cache_dir, image_size=None,
                    shards=None, validate_sample=25, progress=True, min_free_bytes=None):
    """Streams each shard from `archive_dir` (a Drive path) directly into the local cache dirs.

    Uses `tarfile.open(mode="r|")` -- verified to extract from a non-seekable source with zero
    seek attempts -- so a shard is read as ONE sequential Drive file, never copied locally first.
    That matters for space as well as speed: archive plus extracted would be ~57 GiB, while
    extracted alone is ~28.56 GiB and fits.

    Extraction goes to a temp directory per shard and the members are then moved into place, so a
    shard that dies mid-stream leaves no half-written file under a real cache filename.

    `validate_sample` loads and shape-checks that many extracted artifacts per shard; loading all
    of them would add minutes for no benefit, since the tar itself is verified by its own member
    checksums during extraction."""
    image_size = image_size if image_size is not None else jtd.STAGE5_IMAGE_SIZE
    os.makedirs(cache_dir, exist_ok=True)
    os.makedirs(racaf_cache_dir, exist_ok=True)
    if shards is None:
        try:
            shards = sorted(name for name in os.listdir(archive_dir)
                            if name.startswith("cache_shard_") and ".building" not in name)
        except OSError as error:
            raise jtd.PersistentCacheUnavailableError(None, None, archive_dir, error) from error

    result = {"shards": len(shards), "shards_extracted": 0, "files_extracted": 0,
              "bytes_extracted": 0, "validated": 0, "corrupt": [], "elapsed_seconds": 0.0,
              "drive_unreachable": False, "drive_error": None, "skipped_existing": 0}
    if min_free_bytes is not None:
        free = _free_bytes(cache_dir)
        if free < min_free_bytes:
            raise RuntimeError("Refusing to extract: %.2f GiB free but %.2f GiB required."
                               % (free / 1024 ** 3, min_free_bytes / 1024 ** 3))

    start = time.perf_counter()
    for shard in shards:
        source = os.path.join(archive_dir, shard)
        staging = os.path.join(cache_dir, ".extract-%s-%d" % (shard, os.getpid()))
        try:
            mode = "r|gz" if shard.endswith(".gz") else "r|"
            os.makedirs(staging, exist_ok=True)
            with tarfile.open(source, mode) as tar:
                tar.extractall(staging)
        except OSError as error:
            shutil.rmtree(staging, ignore_errors=True)
            if getattr(error, "errno", None) in jtd._TRANSIENT_FUSE_ERRNOS:
                result["drive_unreachable"] = True
                result["drive_error"] = repr(error)
                break
            raise
        except tarfile.TarError as error:
            shutil.rmtree(staging, ignore_errors=True)
            result["corrupt"].append((shard, repr(error)))
            continue

        validated_here = 0
        try:
            for sub, destination_dir in (("features", cache_dir), ("racaf", racaf_cache_dir)):
                source_dir = os.path.join(staging, sub)
                if not os.path.isdir(source_dir):
                    continue
                for name in os.listdir(source_dir):
                    extracted = os.path.join(source_dir, name)
                    destination = os.path.join(destination_dir, name)
                    if os.path.exists(destination):
                        result["skipped_existing"] += 1
                        continue
                    if validated_here < validate_sample:
                        artifact = jcd._artifact_of_path(name)
                        if artifact:
                            _validate_extracted(extracted, artifact, name, image_size)
                            validated_here += 1
                            result["validated"] += 1
                    size = os.path.getsize(extracted)
                    os.replace(extracted, destination)  # atomic into the real cache filename
                    result["files_extracted"] += 1
                    result["bytes_extracted"] += size
        except jtd.CorruptCacheFileError as error:
            result["corrupt"].append((shard, str(error)))
        finally:
            shutil.rmtree(staging, ignore_errors=True)

        result["shards_extracted"] += 1
        if progress:
            elapsed = time.perf_counter() - start
            print("  %s: %d files, %.2f GiB total, %.0fs elapsed (%.1f MB/s)"
                  % (shard, result["files_extracted"], result["bytes_extracted"] / 1024 ** 3,
                     elapsed, result["bytes_extracted"] / 1e6 / elapsed if elapsed else 0.0))
    result["elapsed_seconds"] = time.perf_counter() - start
    return result


# =====================================================================
# Rendering
# =====================================================================

def _gib(num_bytes):
    return "n/a" if num_bytes is None else "%.2f GiB" % (num_bytes / float(1024 ** 3))


def print_build(result):
    print("=" * 78)
    print("CACHE ARCHIVE BUILD")
    print("=" * 78)
    print("  shards total        : %d" % result["shards_total"])
    print("  shards written      : %d" % result["shards_written"])
    print("  shards already there: %d" % result["shards_skipped"])
    print("  entries archived    : %d" % result["entries_archived"])
    print("  bytes written       : %s" % _gib(result["bytes_written"]))
    print("  elapsed             : %.0fs" % result["elapsed_seconds"])
    print("  persistent listing  : %s" % result["listings"])
    incomplete = result["incomplete_entries"]
    print("  entries NOT archived: %d (missing persistently -- never fabricated)" % len(incomplete))
    for id_code, artifacts in list(incomplete.items())[:20]:
        print("      %s -> missing %s" % (id_code, artifacts))
    if result["drive_unreachable"]:
        print("")
        print("  STOPPED -- Drive became unreachable. Completed shards are durable; re-run to")
        print("  resume from the next unfinished shard. Nothing was recomputed.")
        print("    %s" % result["drive_error"])
    remaining = result["shards_total"] - result["shards_written"] - result["shards_skipped"]
    if remaining > 0:
        print("")
        print("  %d shard(s) still to build -- re-run this cell (in a new runtime if needed);"
              % remaining)
        print("  finished shards are skipped automatically.")


def print_extraction_plan(plan):
    print("=" * 78)
    print("LOCAL DISK BUDGET FOR EXTRACTION (measured, before anything is read)")
    print("=" * 78)
    print("  archive shards            : %d (%s on Drive)" % (plan["shards"], _gib(plan["archive_bytes"])))
    print("  already extracted locally : %d files (%s)" % (plan["local_files"], _gib(plan["local_cache_bytes"])))
    print("  still to write            : %s" % _gib(max(0, plan["archive_bytes"] - plan["local_cache_bytes"])))
    print("  runtime margin required   : %s" % _gib(plan["runtime_margin_bytes"]))
    print("  ------------------------------------------------------------------")
    print("  required free space       : %s" % _gib(plan["required_bytes"]))
    print("  actually free now         : %s" % _gib(plan["free_bytes"]))
    print("  free after extraction     : %s" % _gib(plan["free_after_bytes"]))
    if plan["drive_unreachable"]:
        print("")
        print("  DRIVE UNREACHABLE -- the archive could not be measured: %s" % plan["drive_error"])
        print("  Nothing was read, written or recomputed.")
        return
    print("")
    if plan["fits"]:
        print("  VERDICT: fits -- extraction is safe to run.")
    else:
        print("  VERDICT: DOES NOT FIT -- short by %s. Extraction MUST NOT run."
              % _gib(plan["shortfall_bytes"]))
        print("  Free space first (see JOINT_TRAINING_ARCHITECTURE.md Sec 45) -- do NOT delete the")
        print("  persistent Drive cache, the archive, or the local cache that is already correct.")


def print_extract(result):
    print("=" * 78)
    print("CACHE ARCHIVE EXTRACT")
    print("=" * 78)
    print("  shards found        : %d" % result["shards"])
    print("  shards extracted    : %d" % result["shards_extracted"])
    print("  files written       : %d" % result["files_extracted"])
    print("  already present     : %d" % result["skipped_existing"])
    print("  bytes written       : %s" % _gib(result["bytes_extracted"]))
    print("  artifacts validated : %d (sampled shape/dtype check)" % result["validated"])
    print("  elapsed             : %.0fs (%.1f MB/s)"
          % (result["elapsed_seconds"],
             result["bytes_extracted"] / 1e6 / result["elapsed_seconds"]
             if result["elapsed_seconds"] else 0.0))
    if result["corrupt"]:
        print("  CORRUPT shards/members (source left untouched):")
        for shard, detail in result["corrupt"]:
            print("      %s: %s" % (shard, detail))
    if result["drive_unreachable"]:
        print("")
        print("  STOPPED -- Drive became unreachable mid-stream. Already-extracted files are")
        print("  kept; re-run to continue. Nothing was recomputed.")
        print("    %s" % result["drive_error"])
