"""
MEASUREMENT-ONLY instrumentation for the Phase 1 joint-training cache path
(`joint_training_dataset.py`). Adds NO behavior to the pipeline and is imported by nothing the
pipeline itself runs: `joint_training_dataset.py`, `joint_training_model.py`, `racaf.py`,
`local_feature_extraction_dataset.py` and every notebook training cell are unchanged and never
import this module. It exists to answer, from the REAL Colab runtime against the REAL cache
directories, a question a code read alone cannot settle: when Phase 1 is slow, WHICH artifact is
being recomputed, and WHERE was it (or was it not) found.

Method -- deliberately not a reimplementation:

  * The per-image work is driven by calling the REAL, UNMODIFIED
    `joint_training_dataset.precompute_joint_frozen_caches()` with a one-entry `entries` list,
    with exactly the arguments the notebook's Phase 1 cell passes. Its loop body is per-image and
    independent (only `stats` bookkeeping and the periodic progress log accumulate across
    entries), so N one-entry calls execute the identical per-image code path as one N-entry call
    -- while making every measurement attributable to a single image. `vessel_model`/
    `stage4_model` are loaded ONCE by the caller and passed in through that function's own
    existing parameters, exactly as `precompute_authoritative_joint_caches()` already does.

  * Nothing is simulated. No synthetic cache is created, no Drive path is mocked, no cache
    function is replaced by a simplified equivalent, and no timing is fabricated. The only
    patching done is COUNTING/TIMING instrumentation (`_instrument`) which calls straight through
    to the real implementation and returns its real result -- see that class's docstring for the
    exact list and for why no patch can change a computed value.

  * Existence is never inferred from configuration. Every reported path is resolved to an
    absolute path and probed with a real `os.path.exists` against the real filesystem.

What this module may write: local-cache files, and ONLY as a side effect of the real Phase 1 code
path doing what it always does (populating `cache_dir`/`racaf_cache_dir`, whose intended value is
a per-session local SSD directory). It never writes to, deletes from, or modifies the persistent
(Drive) cache -- and does not merely assume so: `run_diagnostic()` records every filesystem write
the instrumented code performs, classifies it by root, and reports a tripwire list
(`report["drive_write_violations"]`) that must be empty.

Reading the numbers -- two counter families are kept strictly separate:
  * `fs` / `compute` counters: operations performed BY THE INSTRUMENTED PIPELINE CODE. These are
    the measurement.
  * `probe_stats`: `os.path.exists` calls made by THIS MODULE to record each artifact's
    before/after state. They are the diagnostic's own overhead, are never folded into the
    pipeline counters, and would not occur during a real run.
"""

import os
import shutil
import time

import numpy as np

import joint_training_dataset as jtd
import local_feature_extraction_dataset as lfed
import racaf

DEFAULT_MAX_IMAGES = 5

# The four independently-cached per-image artifacts, in the order they are reported.
ARTIFACTS = ("vessel", "lesion", "reliability", "rgb")

# Cache states, reported per artifact. Derived ONLY from recorded operations plus the measured
# before-state -- never from what the configuration says should have happened.
LOCAL_HIT = "LOCAL_HIT"
PERSISTENT_HIT_MIRRORED = "PERSISTENT_HIT_MIRRORED"
PERSISTENT_HIT_NOT_MIRRORED = "PERSISTENT_HIT_NOT_MIRRORED"
COMPUTED_FROM_MISS = "COMPUTED_FROM_MISS"
MIXED_LOCAL_PERSISTENT = "MIXED_LOCAL_PERSISTENT"
MISSING = "MISSING"
ERROR = "ERROR"

# Compute entry points, attributed to the artifact each one produces. Any of these running is, by
# definition, expensive regeneration rather than a cache hit.
_COMPUTE_ATTRIBUTION = {
    "predict_vessel_mask": "vessel",
    "racaf.prepare_stage4_input": "lesion",
    "racaf.tta_views": "lesion",
    "racaf.compute_reliability": "reliability",
    "_resize_rgb_01": "rgb",
}
# Raw-image access is its own line item: it is what a full cache hit is supposed to avoid.
_RAW_CALLS = ("lfed._load_raw_bgr", "lfed._resolve_processed_rgb")


# =====================================================================
# Path classification
# =====================================================================

def _norm(path):
    return os.path.normcase(os.path.abspath(path))


def _artifact_of_path(path):
    """Which of `ARTIFACTS` an observed cache path belongs to, or `None`. Matches the filename
    conventions the real path builders produce (`APTOS_{id}_{kind}_{h}x{w}.npy` from
    `lfed._cache_path`, `APTOS_{id}_racaf_reliability.npz` from `racaf.reliability_cache_path`)."""
    name = os.path.basename(path)
    if "_racaf_reliability" in name:
        return "reliability"
    for kind in ("vessel", "lesion", "rgb"):
        if "_" + kind + "_" in name:
            return kind
    return None


class _Roots:
    """The registered cache roots, each flagged persistent (Drive) or local, used to classify
    every observed filesystem path by longest-prefix match. A path under no registered root is
    reported as `other` rather than silently attributed to either side."""

    def __init__(self, roots):
        # roots: iterable of (label, path, is_persistent)
        self.entries = [(label, _norm(path), bool(persistent))
                        for label, path, persistent in roots if path is not None]
        self.entries.sort(key=lambda item: len(item[1]), reverse=True)

    def classify(self, path):
        normalized = _norm(path)
        for label, root, is_persistent in self.entries:
            if normalized == root or normalized.startswith(root + os.sep):
                return label, is_persistent
        return "other", False


# =====================================================================
# Recorder + instrumentation
# =====================================================================

class _Op:
    __slots__ = ("op", "path", "root", "is_persistent", "artifact", "seconds", "nbytes")

    def __init__(self, op, path, root, is_persistent, artifact, seconds, nbytes):
        self.op = op
        self.path = path
        self.root = root
        self.is_persistent = is_persistent
        self.artifact = artifact
        self.seconds = seconds
        self.nbytes = nbytes


class _Recorder:
    """Accumulates every instrumented filesystem operation and compute call. Pure bookkeeping --
    it never decides anything the pipeline does."""

    def __init__(self, roots):
        self.roots = roots
        self.ops = []
        self.calls = []  # (name, seconds)

    def record(self, op, path, seconds, nbytes=0):
        root, is_persistent = self.roots.classify(path)
        self.ops.append(_Op(op, path, root, is_persistent, _artifact_of_path(path), seconds, nbytes))

    def record_call(self, name, seconds):
        self.calls.append((name, seconds))

    # --- queries -------------------------------------------------------
    def matching(self, op=None, artifact=None, is_persistent=None):
        return [o for o in self.ops
                if (op is None or o.op == op)
                and (artifact is None or o.artifact == artifact)
                and (is_persistent is None or o.is_persistent == is_persistent)]

    def count(self, **kwargs):
        return len(self.matching(**kwargs))

    def seconds(self, **kwargs):
        return sum(o.seconds for o in self.matching(**kwargs))

    def call_count(self, name):
        return sum(1 for recorded, _ in self.calls if recorded == name)

    def call_seconds(self, name):
        return sum(seconds for recorded, seconds in self.calls if recorded == name)

    def compute_calls_for(self, artifact):
        return [(name, seconds) for name, seconds in self.calls
                if _COMPUTE_ATTRIBUTION.get(name) == artifact]

    def raw_calls(self):
        return [(name, seconds) for name, seconds in self.calls if name in _RAW_CALLS]


class _OsPathProxy:
    def __init__(self, real, recorder):
        self._real = real
        self._recorder = recorder

    def exists(self, path):
        start = time.perf_counter()
        result = self._real.exists(path)
        self._recorder.record("stat", path, time.perf_counter() - start)
        return result

    def __getattr__(self, name):
        return getattr(self._real, name)


class _OsProxy:
    def __init__(self, real, recorder):
        self._real = real
        self._recorder = recorder
        self.path = _OsPathProxy(real.path, recorder)

    def makedirs(self, name, *args, **kwargs):
        start = time.perf_counter()
        result = self._real.makedirs(name, *args, **kwargs)
        self._recorder.record("mkdir", name, time.perf_counter() - start)
        return result

    def __getattr__(self, name):
        return getattr(self._real, name)


def _nbytes_of(value):
    return int(getattr(value, "nbytes", 0) or 0)


class _NpProxy:
    """Forwards every numpy attribute unchanged; times and counts only `load`/`save`/`savez`, and
    returns their real return values untouched.

    Honest limitation, stated rather than hidden: `np.load` on a `.npz` returns a LAZY `NpzFile`,
    whose member arrays are read on first access -- which happens in the pipeline, after this
    wrapper has returned, and is therefore outside the recorded interval. Only the reliability
    cache is `.npz`, and it holds `kappa` (4 floats) plus a scalar `r`, so the unrecorded
    remainder is a few dozen bytes; negligible next to the `.npy` artifacts, but reported because
    the figure is not exactly complete."""

    def __init__(self, real, recorder):
        self._real = real
        self._recorder = recorder

    def load(self, file, *args, **kwargs):
        start = time.perf_counter()
        result = self._real.load(file, *args, **kwargs)
        self._recorder.record("read", file, time.perf_counter() - start, _nbytes_of(result))
        return result

    def save(self, file, arr, *args, **kwargs):
        start = time.perf_counter()
        result = self._real.save(file, arr, *args, **kwargs)
        self._recorder.record("write", file, time.perf_counter() - start, _nbytes_of(arr))
        return result

    def savez(self, file, *args, **kwargs):
        start = time.perf_counter()
        result = self._real.savez(file, *args, **kwargs)
        nbytes = sum(_nbytes_of(value) for value in list(args) + list(kwargs.values()))
        self._recorder.record("write", file, time.perf_counter() - start, nbytes)
        return result

    def __getattr__(self, name):
        return getattr(self._real, name)


def _timed(name, func, recorder):
    def wrapper(*args, **kwargs):
        start = time.perf_counter()
        try:
            return func(*args, **kwargs)
        finally:
            recorder.record_call(name, time.perf_counter() - start)
    wrapper.__name__ = getattr(func, "__name__", name)
    wrapper.__doc__ = getattr(func, "__doc__", None)
    return wrapper


class _instrument:
    """Context manager installing counting/timing shims, then restoring the originals in a
    `finally`-equivalent `__exit__` -- so a raised exception (e.g. `EmptyFieldOfViewError`) can
    never leave the pipeline patched.

    Every shim calls straight through to the real implementation and returns its real result
    unaltered, so no computed value, no cache decision and no on-disk byte can differ from an
    uninstrumented run:

      * `jtd.os` / `jtd.np`: attribute-forwarding proxies. Only `os.path.exists`, `os.makedirs`,
        `np.load`, `np.save`, `np.savez` are wrapped; everything else (`np.concatenate`,
        `np.float32`, `os.path.join`, ...) forwards untouched via `__getattr__`. These are the
        module-global names `joint_training_dataset.py` looks up at call time, so the patch
        confines instrumentation to that module -- numpy's and TensorFlow's own internal I/O is
        never intercepted.
      * `lfed._load_raw_bgr` / `lfed._resolve_processed_rgb`: raw-image access, wrapped for
        count/time only.
      * `jtd._resize_rgb_01`, `jtd.predict_vessel_mask`, `racaf.prepare_stage4_input`,
        `racaf.tta_views`, `racaf.compute_reliability`: the expensive compute entry points, so
        "was this recomputed?" is answered by observation rather than inference.
    """

    def __init__(self, recorder):
        self._recorder = recorder
        self._saved = []

    def __enter__(self):
        recorder = self._recorder
        self._patch(jtd, "os", _OsProxy(jtd.os, recorder))
        self._patch(jtd, "np", _NpProxy(jtd.np, recorder))
        self._patch(lfed, "_load_raw_bgr",
                    _timed("lfed._load_raw_bgr", lfed._load_raw_bgr, recorder))
        self._patch(lfed, "_resolve_processed_rgb",
                    _timed("lfed._resolve_processed_rgb", lfed._resolve_processed_rgb, recorder))
        self._patch(jtd, "_resize_rgb_01", _timed("_resize_rgb_01", jtd._resize_rgb_01, recorder))
        self._patch(jtd, "predict_vessel_mask",
                    _timed("predict_vessel_mask", jtd.predict_vessel_mask, recorder))
        self._patch(racaf, "prepare_stage4_input",
                    _timed("racaf.prepare_stage4_input", racaf.prepare_stage4_input, recorder))
        self._patch(racaf, "tta_views", _timed("racaf.tta_views", racaf.tta_views, recorder))
        self._patch(racaf, "compute_reliability",
                    _timed("racaf.compute_reliability", racaf.compute_reliability, recorder))
        return recorder

    def _patch(self, module, name, replacement):
        self._saved.append((module, name, getattr(module, name)))
        setattr(module, name, replacement)

    def __exit__(self, exc_type, exc, traceback):
        for module, name, original in reversed(self._saved):
            setattr(module, name, original)
        self._saved = []
        return False


# =====================================================================
# Before/after state probing (this module's own stats, never mixed into pipeline counters)
# =====================================================================

def artifact_paths(id_code, cache_dir, racaf_cache_dir, image_size):
    """The four artifact paths for one image, built by the REAL path builders the pipeline uses
    -- never by a filename template duplicated here."""
    return {
        "vessel": lfed._cache_path(cache_dir, id_code, "vessel", image_size),
        "lesion": lfed._cache_path(cache_dir, id_code, "lesion", image_size),
        "reliability": racaf.reliability_cache_path(racaf_cache_dir, id_code),
        "rgb": jtd._canonical_rgb_cache_path(id_code, cache_dir, image_size),
    }


class _ProbeCounter:
    """This module's OWN `os.path.exists` calls -- reported separately so they can never be
    mistaken for operations the pipeline performed."""

    def __init__(self):
        self.local = 0
        self.persistent = 0

    @property
    def total(self):
        return self.local + self.persistent


def probe_states(id_code, cache_dir, racaf_cache_dir, persistent_cache_dir,
                 persistent_racaf_cache_dir, image_size, probe_counter=None):
    """Real `os.path.exists` probes of every artifact's local and persistent path."""
    local = artifact_paths(id_code, cache_dir, racaf_cache_dir, image_size)
    persistent = (
        artifact_paths(id_code, persistent_cache_dir, persistent_racaf_cache_dir, image_size)
        if persistent_cache_dir is not None and persistent_racaf_cache_dir is not None else None
    )
    states = {}
    for artifact in ARTIFACTS:
        local_path = local[artifact]
        local_exists = os.path.exists(local_path)
        if probe_counter is not None:
            probe_counter.local += 1
        persistent_path = persistent[artifact] if persistent is not None else None
        if persistent_path is None:
            persistent_exists = None
        else:
            persistent_exists = os.path.exists(persistent_path)
            if probe_counter is not None:
                probe_counter.persistent += 1
        states[artifact] = {
            "local_path": local_path,
            "local_exists": local_exists,
            "persistent_path": persistent_path,
            "persistent_exists": persistent_exists,
        }
    return states


# =====================================================================
# Per-artifact classification -- from measurements only
# =====================================================================

def classify_artifact(before, after, recorder, artifact, errored=False):
    """The observed state for one artifact, decided in this order:

      1. `ERROR`      -- the real code path raised for this image.
      2. `COMPUTED_FROM_MISS` -- a compute entry point attributed to this artifact actually ran.
      3. `PERSISTENT_HIT_MIRRORED` / `PERSISTENT_HIT_NOT_MIRRORED` -- content was read from a
         persistent root; mirrored iff a local write for the same artifact was also recorded.
      4. `MIXED_LOCAL_PERSISTENT` -- read locally, yet the pipeline also stat'ed a persistent path
         for this artifact (the local cache did not by itself satisfy the lookup), or the file
         appeared locally without an observed compute or persistent read to explain it.
      5. `LOCAL_HIT`  -- present locally BEFORE the call, and no persistent access occurred.
      6. `MISSING`    -- still absent afterward and nothing computed it (e.g. an entry the real
         code path deliberately skipped).

    `LOCAL_HIT` is asserted only when the artifact was already local BEFORE the call -- an
    artifact this call created is always reported by whatever created it, never as a hit.
    """
    if errored:
        return ERROR
    if recorder.compute_calls_for(artifact):
        return COMPUTED_FROM_MISS
    persistent_reads = recorder.count(op="read", artifact=artifact, is_persistent=True)
    local_writes = recorder.count(op="write", artifact=artifact, is_persistent=False)
    if persistent_reads:
        return PERSISTENT_HIT_MIRRORED if local_writes else PERSISTENT_HIT_NOT_MIRRORED
    local_reads = recorder.count(op="read", artifact=artifact, is_persistent=False)
    persistent_stats = recorder.count(op="stat", artifact=artifact, is_persistent=True)
    if local_reads and persistent_stats:
        return MIXED_LOCAL_PERSISTENT
    if before["local_exists"]:
        return LOCAL_HIT
    if after["local_exists"]:
        return MIXED_LOCAL_PERSISTENT
    return MISSING


def artifact_measurements(before, after, recorder, artifact, errored=False):
    compute = recorder.compute_calls_for(artifact)
    return {
        "artifact": artifact,
        "action": classify_artifact(before, after, recorder, artifact, errored=errored),
        "local_before": before["local_exists"],
        "persistent_before": before["persistent_exists"],
        "local_after": after["local_exists"],
        "local_path": before["local_path"],
        "persistent_path": before["persistent_path"],
        "local_stats": recorder.count(op="stat", artifact=artifact, is_persistent=False),
        "drive_stats": recorder.count(op="stat", artifact=artifact, is_persistent=True),
        "local_stat_seconds": recorder.seconds(op="stat", artifact=artifact, is_persistent=False),
        "drive_stat_seconds": recorder.seconds(op="stat", artifact=artifact, is_persistent=True),
        "local_reads": recorder.count(op="read", artifact=artifact, is_persistent=False),
        "drive_reads": recorder.count(op="read", artifact=artifact, is_persistent=True),
        "local_read_seconds": recorder.seconds(op="read", artifact=artifact, is_persistent=False),
        "drive_read_seconds": recorder.seconds(op="read", artifact=artifact, is_persistent=True),
        "local_writes": recorder.count(op="write", artifact=artifact, is_persistent=False),
        "drive_writes": recorder.count(op="write", artifact=artifact, is_persistent=True),
        "local_write_seconds": recorder.seconds(op="write", artifact=artifact, is_persistent=False),
        "bytes_read": sum(op.nbytes for op in recorder.matching(op="read", artifact=artifact)),
        "bytes_written": sum(op.nbytes for op in recorder.matching(op="write", artifact=artifact)),
        "computed": bool(compute),
        "compute_calls": [name for name, _ in compute],
        "compute_seconds": sum(seconds for _, seconds in compute),
        # Time this artifact can be held responsible for: its own file operations plus the compute
        # attributed to it. Vessel/lesion/reliability additionally SHARE one
        # `_get_or_compute_joint_frozen_outputs` call, so the per-image wall clock below is the
        # authoritative total; these attributable figures explain where that total went.
        "attributable_seconds": (
            recorder.seconds(op="stat", artifact=artifact)
            + recorder.seconds(op="read", artifact=artifact)
            + recorder.seconds(op="write", artifact=artifact)
            + sum(seconds for _, seconds in compute)
        ),
    }


# =====================================================================
# Cache-root inspection
# =====================================================================

def _filesystem_of(path):
    """The mount point and filesystem type backing `path`, read from `/proc/mounts` by longest
    matching mount point. Returns `None` where `/proc/mounts` is unavailable (e.g. Windows) --
    never raises, and never guesses."""
    try:
        with open("/proc/mounts", encoding="utf-8", errors="replace") as handle:
            mounts = [line.split() for line in handle]
    except OSError:
        return None
    target = _norm(path)
    best = None
    for fields in mounts:
        if len(fields) < 3:
            continue
        mount_point, fstype = fields[1], fields[2]
        normalized = _norm(mount_point)
        if target == normalized or target.startswith(normalized.rstrip(os.sep) + os.sep):
            if best is None or len(normalized) > len(best[0]):
                best = (normalized, mount_point, fstype)
    if best is None:
        return None
    return {"mount_point": best[1], "fstype": best[2]}


def inspect_cache_root(path, count_files=True, max_entries=250000, sample_size=32):
    """Facts about one cache root, measured rather than assumed. Uses ONE non-recursive
    `os.scandir` pass (a directory listing, not a per-file stat storm) and stops at `max_entries`,
    so it stays bounded even against a large Drive-mounted directory. Total size is ESTIMATED from
    at most `sample_size` per-kind file stats and is labelled as an estimate -- a byte-exact total
    would need one stat per file, which over Drive FUSE is exactly the kind of cost this
    diagnostic exists to avoid provoking."""
    result = {
        "path": path,
        "abspath": os.path.abspath(path) if path is not None else None,
        "exists": False,
        "filesystem": None,
        "counts": {},
        "total_files": 0,
        "truncated": False,
        "listing_seconds": None,
        "estimated_bytes": None,
        "estimate_basis": None,
        "error": None,
    }
    if path is None:
        return result
    result["exists"] = os.path.isdir(path)
    result["filesystem"] = _filesystem_of(path)
    if not result["exists"] or not count_files:
        return result

    counts = {kind: 0 for kind in ARTIFACTS}
    counts["other"] = 0
    samples = {kind: [] for kind in ARTIFACTS}
    start = time.perf_counter()
    try:
        with os.scandir(path) as entries:
            for entry in entries:
                if result["total_files"] >= max_entries:
                    result["truncated"] = True
                    break
                if not entry.is_file():
                    continue
                result["total_files"] += 1
                kind = _artifact_of_path(entry.name)
                if kind is None:
                    counts["other"] += 1
                    continue
                counts[kind] += 1
                if len(samples[kind]) < sample_size:
                    samples[kind].append(entry.path)
    except OSError as error:
        result["error"] = repr(error)
    result["listing_seconds"] = time.perf_counter() - start
    result["counts"] = counts

    estimated = 0
    basis = {}
    for kind, paths in samples.items():
        if not paths:
            continue
        sizes = []
        for sampled in paths:
            try:
                sizes.append(os.path.getsize(sampled))
            except OSError:
                continue
        if not sizes:
            continue
        mean = sum(sizes) / float(len(sizes))
        basis[kind] = {"sampled_files": len(sizes), "mean_bytes": mean}
        estimated += mean * counts[kind]
    if basis:
        result["estimated_bytes"] = estimated
        result["estimate_basis"] = basis
    return result


def disk_usage(path):
    """Free/used/total for the filesystem holding `path`. Walks up to the nearest EXISTING
    ancestor first, since on a fresh runtime the cache directory itself does not exist yet and the
    interesting number is the local SSD's capacity, not whether that particular directory has been
    created."""
    resolved = os.path.abspath(path)
    while resolved and not os.path.exists(resolved):
        parent = os.path.dirname(resolved)
        if parent == resolved:
            break
        resolved = parent
    try:
        usage = shutil.disk_usage(resolved)
    except OSError as error:
        return {"path": resolved, "error": repr(error)}
    return {
        "path": resolved,
        "total_bytes": usage.total,
        "used_bytes": usage.used,
        "free_bytes": usage.free,
    }


# =====================================================================
# Numerical integrity (read-only)
# =====================================================================

def compare_cached_rgb_to_fresh(id_code, image_dir, cached_rgb_path, processed_dir,
                                image_size=jtd.STAGE5_IMAGE_SIZE):
    """Loads an EXISTING canonical-RGB cache file and compares it against a freshly generated
    value for the same image, using the real, unmodified `lfed._load_raw_bgr` /
    `lfed._resolve_processed_rgb` / `jtd._resize_rgb_01` chain -- the exact chain that produced
    the cached bytes. Strictly read-only: nothing is written, and the fresh array is discarded.
    Never regenerates anything beyond this one image."""
    result = {"id_code": id_code, "cached_rgb_path": cached_rgb_path, "compared": False,
              "error": None}
    if not os.path.exists(cached_rgb_path):
        result["error"] = "cached RGB file does not exist -- nothing to compare"
        return result
    try:
        cached = np.load(cached_rgb_path)
        raw_bgr = lfed._load_raw_bgr(image_dir, id_code)
        rgb_native = lfed._resolve_processed_rgb(raw_bgr, processed_dir, id_code)
        fresh = jtd._resize_rgb_01(rgb_native, image_size)
    except Exception as error:  # noqa: BLE001 -- a diagnostic must report, never crash the cell
        result["error"] = repr(error)
        return result
    result.update({
        "compared": True,
        "cached_shape": tuple(cached.shape),
        "fresh_shape": tuple(fresh.shape),
        "cached_dtype": str(cached.dtype),
        "fresh_dtype": str(fresh.dtype),
        "shapes_match": tuple(cached.shape) == tuple(fresh.shape),
        "dtypes_match": cached.dtype == fresh.dtype,
    })
    if result["shapes_match"]:
        difference = np.abs(cached.astype(np.float64) - fresh.astype(np.float64))
        result["max_abs_diff"] = float(difference.max())
        result["mean_abs_diff"] = float(difference.mean())
        result["exactly_equal"] = bool(np.array_equal(cached, fresh))
        result["bitwise_identical"] = bool(cached.tobytes() == fresh.tobytes())
    return result


# =====================================================================
# The diagnostic run
# =====================================================================

def _run_one_image(id_code, diagnosis, roots, image_dir, cache_dir, racaf_cache_dir,
                   persistent_cache_dir, persistent_racaf_cache_dir, processed_dir,
                   vessel_model, stage4_model, image_size, probe_counter):
    """One image through the REAL `precompute_joint_frozen_caches`, instrumented. `progress_every=0`
    only silences that function's periodic progress log (`if progress_every and ...`); it changes
    no cache decision."""
    before = probe_states(id_code, cache_dir, racaf_cache_dir, persistent_cache_dir,
                          persistent_racaf_cache_dir, image_size, probe_counter)
    recorder = _Recorder(roots)
    errored = None
    stats = None
    start = time.perf_counter()
    try:
        with _instrument(recorder):
            stats = jtd.precompute_joint_frozen_caches(
                [(id_code, diagnosis)],
                image_dir=image_dir,
                cache_dir=cache_dir,
                racaf_cache_dir=racaf_cache_dir,
                persistent_cache_dir=persistent_cache_dir,
                persistent_racaf_cache_dir=persistent_racaf_cache_dir,
                vessel_model=vessel_model,
                stage4_model=stage4_model,
                processed_dir=processed_dir,
                image_size=image_size,
                progress_every=0,
            )
    except Exception as error:  # noqa: BLE001 -- one bad image must not abort the diagnostic
        errored = repr(error)
    elapsed = time.perf_counter() - start
    after = probe_states(id_code, cache_dir, racaf_cache_dir, persistent_cache_dir,
                         persistent_racaf_cache_dir, image_size, probe_counter)

    raw_calls = recorder.raw_calls()
    # An empty-FOV image is not an exception HERE -- the real Phase 1 catches
    # `EmptyFieldOfViewError` itself, logs it and continues -- but its Stage 03/04/RACAF artifacts
    # genuinely failed to be produced, so they are reported as `ERROR` rather than as an
    # attempted-and-therefore-"computed" state. Canonical RGB is deliberately excluded: the real
    # code path caches it for such an image anyway, since RGB does not depend on FOV detection.
    skipped_empty_fov = bool(stats) and id_code in stats.get("skipped_empty_fov", [])
    return {
        "id_code": id_code,
        "diagnosis": diagnosis,
        "elapsed_seconds": elapsed,
        "error": errored,
        "skipped_empty_fov": skipped_empty_fov,
        # The real function's OWN branch counters -- ground truth for which Phase 1 branch ran,
        # independent of this module's path-based classification, so the two can be cross-checked.
        "phase1_stats": stats,
        "artifacts": {
            artifact: artifact_measurements(
                before[artifact], after[artifact], recorder, artifact,
                errored=bool(errored) or (skipped_empty_fov and artifact != "rgb"),
            )
            for artifact in ARTIFACTS
        },
        "raw_image": {
            "path": os.path.join(image_dir, id_code + ".png"),
            "exists": os.path.exists(os.path.join(image_dir, id_code + ".png")),
            "loaded": recorder.call_count("lfed._load_raw_bgr") > 0,
            "load_count": recorder.call_count("lfed._load_raw_bgr"),
            "load_seconds": recorder.call_seconds("lfed._load_raw_bgr"),
            "stage02_ran": recorder.call_count("lfed._resolve_processed_rgb") > 0,
            "stage02_seconds": recorder.call_seconds("lfed._resolve_processed_rgb"),
            "calls": [name for name, _ in raw_calls],
        },
        "totals": {
            "local_stats": recorder.count(op="stat", is_persistent=False),
            "drive_stats": recorder.count(op="stat", is_persistent=True),
            "local_reads": recorder.count(op="read", is_persistent=False),
            "drive_reads": recorder.count(op="read", is_persistent=True),
            "local_writes": recorder.count(op="write", is_persistent=False),
            "drive_writes": recorder.count(op="write", is_persistent=True),
            "mkdirs": recorder.count(op="mkdir"),
            "drive_seconds": recorder.seconds(is_persistent=True),
            "local_seconds": recorder.seconds(is_persistent=False),
            "compute_seconds": sum(seconds for name, seconds in recorder.calls
                                   if name in _COMPUTE_ATTRIBUTION),
            "raw_seconds": sum(seconds for _, seconds in raw_calls),
        },
        "drive_write_paths": [op.path for op in recorder.matching(op="write", is_persistent=True)]
                             + [op.path for op in recorder.matching(op="mkdir", is_persistent=True)],
        "_recorder": recorder,
    }


def _measure_build_joint_sample(id_code, diagnosis, roots, image_dir, cache_dir, racaf_cache_dir,
                                persistent_cache_dir, persistent_racaf_cache_dir, processed_dir,
                                vessel_model, stage4_model, image_size):
    """The TRAINING-TIME path (Phase 2's per-sample function), measured on one image with
    `augment=False`. This is what every epoch actually runs, so it -- not Phase 1 -- answers
    whether steady-state training has a Drive dependency."""
    recorder = _Recorder(roots)
    errored = None
    start = time.perf_counter()
    try:
        with _instrument(recorder):
            jtd._build_joint_sample(
                id_code, diagnosis, image_dir, cache_dir, racaf_cache_dir,
                vessel_model, stage4_model, False, None,
                processed_dir=processed_dir, image_size=image_size,
                persistent_cache_dir=persistent_cache_dir,
                persistent_racaf_cache_dir=persistent_racaf_cache_dir,
            )
    except Exception as error:  # noqa: BLE001
        errored = repr(error)
    return {
        "id_code": id_code,
        "elapsed_seconds": time.perf_counter() - start,
        "error": errored,
        "local_stats": recorder.count(op="stat", is_persistent=False),
        "drive_stats": recorder.count(op="stat", is_persistent=True),
        "local_reads": recorder.count(op="read", is_persistent=False),
        "drive_reads": recorder.count(op="read", is_persistent=True),
        "local_writes": recorder.count(op="write", is_persistent=False),
        "drive_writes": recorder.count(op="write", is_persistent=True),
        "raw_loaded": recorder.call_count("lfed._load_raw_bgr") > 0,
        "compute_calls": [name for name, _ in recorder.calls if name in _COMPUTE_ATTRIBUTION],
        "drive_paths_touched": sorted({op.path for op in recorder.matching(is_persistent=True)}),
    }


def select_diagnostic_entries(entries, max_images=DEFAULT_MAX_IMAGES, explicit_ids=None):
    """The images to measure. `explicit_ids` (e.g. a known empty-FOV id observed in a real Phase 1
    log) selects exactly those, in order, and is not padded; otherwise the first `max_images`
    entries of the authoritative split are taken -- the same entries a real Phase 1 run reaches
    first, so the measurement reflects work that run actually did."""
    entries = list(entries)
    if explicit_ids:
        by_id = {id_code: diagnosis for id_code, diagnosis in entries}
        selected = []
        for id_code in explicit_ids:
            selected.append((id_code, by_id.get(id_code, 0)))
        return selected
    return entries[:max_images]


def run_diagnostic(entries, image_dir, cache_dir, racaf_cache_dir,
                   persistent_cache_dir=None, persistent_racaf_cache_dir=None,
                   processed_dir=jtd.DEFAULT_PROCESSED_DIR,
                   vessel_model=None, vessel_model_path=jtd.DEFAULT_VESSEL_MODEL_PATH,
                   stage4_model=None, image_size=jtd.STAGE5_IMAGE_SIZE,
                   max_images=DEFAULT_MAX_IMAGES, explicit_ids=None,
                   run_second_pass=True, measure_build_sample=True,
                   compare_rgb_numerically=True, count_root_files=True,
                   experiment_dir=None):
    """Runs the two-pass measurement and returns a plain-dict report (`print_report` renders it).

    Pass 1 measures each image as the runtime finds it. Pass 2 re-measures the SAME images
    immediately afterward: with pass 1 having completed, every artifact should now be local, so
    pass 2 is the direct test of whether a warmed local working set removes the Drive dependency.

    Models are loaded once here and passed into every per-image call through
    `precompute_joint_frozen_caches`'s own `vessel_model`/`stage4_model` parameters.

    `experiment_dir` (optional): the training run's checkpoint directory, reported for its path,
    existence and backing filesystem only. It is never created, written to or resolved through
    `experiment_manager` -- this diagnostic must not touch experiment semantics."""
    selected = select_diagnostic_entries(entries, max_images=max_images, explicit_ids=explicit_ids)
    roots = _Roots([
        ("local_cache", cache_dir, False),
        ("local_racaf_cache", racaf_cache_dir, False),
        ("drive_cache", persistent_cache_dir, True),
        ("drive_racaf_cache", persistent_racaf_cache_dir, True),
        ("staged_images", image_dir, False),
        ("processed_dir", processed_dir, False),
    ])
    probe_counter = _ProbeCounter()

    model_load_start = time.perf_counter()
    resolved_vessel = vessel_model if vessel_model is not None else jtd.load_vessel_model(vessel_model_path)
    resolved_stage4 = stage4_model if stage4_model is not None else racaf.load_frozen_stage4_model()
    model_load_seconds = time.perf_counter() - model_load_start

    report = {
        "selected_ids": [id_code for id_code, _ in selected],
        "image_size": tuple(image_size),
        "model_load_seconds": model_load_seconds,
        "roots": {
            "staged_image_dir": inspect_cache_root(image_dir, count_files=count_root_files),
            "local_cache_dir": inspect_cache_root(cache_dir, count_files=count_root_files),
            "local_racaf_cache_dir": inspect_cache_root(racaf_cache_dir, count_files=count_root_files),
            "persistent_cache_dir": inspect_cache_root(persistent_cache_dir, count_files=count_root_files),
            "persistent_racaf_cache_dir": inspect_cache_root(persistent_racaf_cache_dir,
                                                             count_files=count_root_files),
            "processed_dir": inspect_cache_root(processed_dir, count_files=count_root_files),
            # Reported read-only: never created, never resolved through `experiment_manager`.
            "experiment_dir": inspect_cache_root(experiment_dir, count_files=count_root_files),
        },
        "disk_usage": {"local": disk_usage(cache_dir)},
        "expected_paths": {},
        "pass1": [],
        "pass2": [],
        "build_sample": [],
        "rgb_numerical_check": None,
        "drive_write_violations": [],
    }

    # Exact expected filenames for the first two ids, each probed individually -- the decisive
    # answer to "is the notebook checking the same location the artifacts were persisted to".
    for id_code, _ in selected[:2]:
        report["expected_paths"][id_code] = probe_states(
            id_code, cache_dir, racaf_cache_dir, persistent_cache_dir,
            persistent_racaf_cache_dir, image_size, probe_counter,
        )

    for id_code, diagnosis in selected:
        report["pass1"].append(_run_one_image(
            id_code, diagnosis, roots, image_dir, cache_dir, racaf_cache_dir,
            persistent_cache_dir, persistent_racaf_cache_dir, processed_dir,
            resolved_vessel, resolved_stage4, image_size, probe_counter,
        ))

    if run_second_pass:
        for id_code, diagnosis in selected:
            report["pass2"].append(_run_one_image(
                id_code, diagnosis, roots, image_dir, cache_dir, racaf_cache_dir,
                persistent_cache_dir, persistent_racaf_cache_dir, processed_dir,
                resolved_vessel, resolved_stage4, image_size, probe_counter,
            ))

    if measure_build_sample:
        for id_code, diagnosis in selected:
            report["build_sample"].append(_measure_build_joint_sample(
                id_code, diagnosis, roots, image_dir, cache_dir, racaf_cache_dir,
                persistent_cache_dir, persistent_racaf_cache_dir, processed_dir,
                resolved_vessel, resolved_stage4, image_size,
            ))

    if compare_rgb_numerically and selected:
        # At most ONE image, and only against a cache file that already exists -- read-only.
        id_code = selected[0][0]
        paths = artifact_paths(id_code, cache_dir, racaf_cache_dir, image_size)
        report["rgb_numerical_check"] = compare_cached_rgb_to_fresh(
            id_code, image_dir, paths["rgb"], processed_dir, image_size,
        )

    for pass_name in ("pass1", "pass2"):
        for record in report[pass_name]:
            for path in record["drive_write_paths"]:
                report["drive_write_violations"].append((pass_name, record["id_code"], path))
    report["probe_stats"] = {"local": probe_counter.local, "persistent": probe_counter.persistent,
                             "total": probe_counter.total}
    report["summary"] = summarize(report)
    return report


def summarize(report):
    """Per-artifact totals across pass 1, plus the pass-1/pass-2 aggregates."""
    summary = {"artifacts": {}, "pass_totals": {}}
    for artifact in ARTIFACTS:
        records = [record["artifacts"][artifact] for record in report["pass1"]]
        times = [record["attributable_seconds"] for record in records]
        summary["artifacts"][artifact] = {
            "local_hits": sum(1 for r in records if r["action"] == LOCAL_HIT),
            "persistent_hits": sum(1 for r in records
                                   if r["action"] in (PERSISTENT_HIT_MIRRORED,
                                                      PERSISTENT_HIT_NOT_MIRRORED)),
            "computed": sum(1 for r in records if r["action"] == COMPUTED_FROM_MISS),
            "missing": sum(1 for r in records if r["action"] == MISSING),
            "errors": sum(1 for r in records if r["action"] == ERROR),
            "drive_stats": sum(r["drive_stats"] for r in records),
            "drive_reads": sum(r["drive_reads"] for r in records),
            "local_writes": sum(r["local_writes"] for r in records),
            "avg_seconds": (sum(times) / len(times)) if times else 0.0,
            "max_seconds": max(times) if times else 0.0,
        }
    for pass_name in ("pass1", "pass2"):
        records = report[pass_name]
        summary["pass_totals"][pass_name] = {
            "images": len(records),
            "elapsed_seconds": sum(r["elapsed_seconds"] for r in records),
            "local_stats": sum(r["totals"]["local_stats"] for r in records),
            "drive_stats": sum(r["totals"]["drive_stats"] for r in records),
            "local_reads": sum(r["totals"]["local_reads"] for r in records),
            "drive_reads": sum(r["totals"]["drive_reads"] for r in records),
            "local_writes": sum(r["totals"]["local_writes"] for r in records),
            "drive_writes": sum(r["totals"]["drive_writes"] for r in records),
            "compute_seconds": sum(r["totals"]["compute_seconds"] for r in records),
            "raw_seconds": sum(r["totals"]["raw_seconds"] for r in records),
            "drive_seconds": sum(r["totals"]["drive_seconds"] for r in records),
            "raw_loads": sum(r["raw_image"]["load_count"] for r in records),
        }
    return summary


# =====================================================================
# Rendering
# =====================================================================

def _gib(num_bytes):
    if num_bytes is None:
        return "n/a"
    return "%.2f GiB" % (num_bytes / float(1024 ** 3))


def _print_root(label, info):
    if info["abspath"] is None:
        print("  %-26s (not configured)" % (label + ":"))
        return
    print("  %-26s %s" % (label + ":", info["abspath"]))
    if not info["exists"]:
        print("      exists: NO" + ("  (error: %s)" % info["error"] if info["error"] else ""))
        return
    filesystem = info["filesystem"]
    fs_text = ("%s on %s" % (filesystem["fstype"], filesystem["mount_point"])) if filesystem else "unknown"
    print("      exists: YES   filesystem: %s" % fs_text)
    if info["counts"]:
        counts = info["counts"]
        print("      files: total=%d  vessel=%d lesion=%d reliability=%d rgb=%d other=%d%s" % (
            info["total_files"], counts.get("vessel", 0), counts.get("lesion", 0),
            counts.get("reliability", 0), counts.get("rgb", 0), counts.get("other", 0),
            "  (TRUNCATED)" if info["truncated"] else ""))
        print("      listing took %.2fs; estimated size %s%s" % (
            info["listing_seconds"] or 0.0, _gib(info["estimated_bytes"]),
            " (sampled estimate, not exact)" if info["estimated_bytes"] is not None else ""))
    if info["error"]:
        print("      error: %s" % info["error"])


def _print_image(record, pass_label):
    print("")
    print("image_id=%s  [%s]" % (record["id_code"], pass_label))
    if record["error"]:
        print("  ERROR: %s" % record["error"])
    if record.get("skipped_empty_fov"):
        print("  NOTE: the real Phase 1 skipped this image -- empty Stage 03 field of view. Its "
              "Stage 03/04/RACAF artifacts are reported as ERROR; canonical RGB is still cached.")
    for artifact in ARTIFACTS:
        measurement = record["artifacts"][artifact]
        persistent_before = measurement["persistent_before"]
        print("  %s:" % artifact.capitalize())
        print("    local: %s" % ("HIT" if measurement["local_before"] else "MISS"))
        print("    persistent: %s" % ("n/a (no persistent dir configured)" if persistent_before is None
                                      else ("HIT" if persistent_before else "MISS")))
        print("    action: %s" % measurement["action"])
        print("    drive_stats: %d  drive_reads: %d  local_stats: %d  local_reads: %d  local_writes: %d"
              % (measurement["drive_stats"], measurement["drive_reads"], measurement["local_stats"],
                 measurement["local_reads"], measurement["local_writes"]))
        print("    drive_time: %.3fs  local_time: %.3fs  compute_time: %.3fs  attributable: %.3fs"
              % (measurement["drive_stat_seconds"] + measurement["drive_read_seconds"],
                 measurement["local_stat_seconds"] + measurement["local_read_seconds"]
                 + measurement["local_write_seconds"],
                 measurement["compute_seconds"], measurement["attributable_seconds"]))
        print("    recomputed: %s%s" % (
            "YES" if measurement["computed"] else "NO",
            ("  (" + ", ".join(measurement["compute_calls"]) + ")") if measurement["computed"] else ""))
    raw = record["raw_image"]
    print("  Raw image:")
    print("    path: %s" % raw["path"])
    print("    exists: %s  loaded: %s (%dx)  load_time: %.3fs" % (
        "YES" if raw["exists"] else "NO", "YES" if raw["loaded"] else "NO",
        raw["load_count"], raw["load_seconds"]))
    print("    Stage02 (_resolve_processed_rgb) ran: %s  time: %.3fs" % (
        "YES" if raw["stage02_ran"] else "NO", raw["stage02_seconds"]))
    totals = record["totals"]
    print("  Phase 1 branch counters (the real function's own): %s" % (record["phase1_stats"],))
    print("  TOTAL: elapsed=%.3fs  drive_ops=%d (stats=%d reads=%d writes=%d)  local_ops=%d  compute=%.3fs"
          % (record["elapsed_seconds"],
             totals["drive_stats"] + totals["drive_reads"] + totals["drive_writes"],
             totals["drive_stats"], totals["drive_reads"], totals["drive_writes"],
             totals["local_stats"] + totals["local_reads"] + totals["local_writes"],
             totals["compute_seconds"]))


def print_report(report):
    """Renders the whole report. Prints only measured values -- every 'PROVEN' line below is
    generated from a recorded number, and anything the run did not measure is listed as UNKNOWN
    rather than inferred."""
    print("=" * 78)
    print("JOINT CACHE DIAGNOSTIC -- MEASUREMENT ONLY (no full Phase 1, no training)")
    print("=" * 78)
    print("Images measured : %s" % ", ".join(report["selected_ids"]))
    print("Canonical size  : %s" % (report["image_size"],))
    print("Model load time : %.2fs (once, reused for every image below)" % report["model_load_seconds"])

    print("")
    print("--- RESOLVED PATHS (probed, not assumed) ---")
    for label in ("staged_image_dir", "local_cache_dir", "local_racaf_cache_dir",
                  "persistent_cache_dir", "persistent_racaf_cache_dir", "processed_dir",
                  "experiment_dir"):
        _print_root(label, report["roots"][label])
    # Stated explicitly rather than left implicit: canonical RGB has no directory of its own, so
    # "the canonical RGB cache" IS the frozen-cache directory above. The per-file existence block
    # below is what actually settles whether its files are there.
    rgb_local = report["roots"]["local_cache_dir"]["counts"].get("rgb")
    rgb_persistent = report["roots"]["persistent_cache_dir"]["counts"].get("rgb")
    print("  %-26s shares local_cache_dir (%s rgb file(s)) and persistent_cache_dir (%s rgb file(s))"
          % ("canonical RGB cache:",
             "n/a" if rgb_local is None else rgb_local,
             "n/a" if rgb_persistent is None else rgb_persistent))
    usage = report["disk_usage"]["local"]
    if "error" in usage:
        print("  local disk usage: error %s" % usage["error"])
    else:
        print("  local disk (%s): total=%s used=%s free=%s" % (
            usage["path"], _gib(usage["total_bytes"]), _gib(usage["used_bytes"]),
            _gib(usage["free_bytes"])))

    print("")
    print("--- EXACT EXPECTED ARTIFACT PATHS (per-file existence, decisive for RGB) ---")
    for id_code, states in report["expected_paths"].items():
        print("  %s" % id_code)
        for artifact in ARTIFACTS:
            state = states[artifact]
            print("    %-12s local       %-5s %s" % (
                artifact, "EXISTS" if state["local_exists"] else "ABSENT", state["local_path"]))
            if state["persistent_path"] is None:
                print("    %-12s persistent  n/a (no persistent dir configured)" % "")
            else:
                print("    %-12s persistent  %-5s %s" % (
                    "", "EXISTS" if state["persistent_exists"] else "ABSENT",
                    state["persistent_path"]))

    print("")
    print("=" * 78)
    print("PASS 1 -- as the runtime found it")
    print("=" * 78)
    for record in report["pass1"]:
        _print_image(record, "pass 1")

    if report["pass2"]:
        print("")
        print("=" * 78)
        print("PASS 2 -- same images again (local working set should now be complete)")
        print("=" * 78)
        for record in report["pass2"]:
            _print_image(record, "pass 2")

    print("")
    print("=" * 78)
    print("SUMMARY (pass 1)")
    print("=" * 78)
    header = ("Artifact", "LocalHit", "PersHit", "Computed", "DriveRd", "LocWr", "AvgTime", "MaxTime")
    print("%-12s %8s %8s %9s %8s %6s %9s %9s" % header)
    for artifact in ARTIFACTS:
        stats = report["summary"]["artifacts"][artifact]
        print("%-12s %8d %8d %9d %8d %6d %8.3fs %8.3fs" % (
            artifact, stats["local_hits"], stats["persistent_hits"], stats["computed"],
            stats["drive_reads"], stats["local_writes"], stats["avg_seconds"], stats["max_seconds"]))

    for pass_name in ("pass1", "pass2"):
        if not report[pass_name]:
            continue
        totals = report["summary"]["pass_totals"][pass_name]
        print("")
        print("%s TOTALS (%d images)" % (pass_name.upper(), totals["images"]))
        print("  TOTAL DRIVE OPERATIONS : %d (stats=%d reads=%d writes=%d)" % (
            totals["drive_stats"] + totals["drive_reads"] + totals["drive_writes"],
            totals["drive_stats"], totals["drive_reads"], totals["drive_writes"]))
        print("  TOTAL DRIVE READS      : %d  (%.3fs in Drive operations)" % (
            totals["drive_reads"], totals["drive_seconds"]))
        print("  TOTAL LOCAL OPERATIONS : %d (stats=%d reads=%d writes=%d)" % (
            totals["local_stats"] + totals["local_reads"] + totals["local_writes"],
            totals["local_stats"], totals["local_reads"], totals["local_writes"]))
        print("  TOTAL COMPUTATION      : %.3fs (raw image loads: %d, %.3fs)" % (
            totals["compute_seconds"], totals["raw_loads"], totals["raw_seconds"]))
        print("  TOTAL ELAPSED TIME     : %.3fs" % totals["elapsed_seconds"])

    if report["build_sample"]:
        print("")
        print("=" * 78)
        print("TRAINING-TIME PATH (_build_joint_sample, augment=False) -- what every epoch runs")
        print("=" * 78)
        for record in report["build_sample"]:
            print("  %s: elapsed=%.3fs drive_stats=%d drive_reads=%d drive_writes=%d "
                  "local_reads=%d raw_loaded=%s compute=%s%s" % (
                      record["id_code"], record["elapsed_seconds"], record["drive_stats"],
                      record["drive_reads"], record["drive_writes"], record["local_reads"],
                      "YES" if record["raw_loaded"] else "NO",
                      record["compute_calls"] or "none",
                      ("  ERROR: " + record["error"]) if record["error"] else ""))
            if record["drive_paths_touched"]:
                for path in record["drive_paths_touched"]:
                    print("      drive path touched: %s" % path)

    check = report["rgb_numerical_check"]
    print("")
    print("=" * 78)
    print("CANONICAL RGB NUMERICAL INTEGRITY (one image, read-only)")
    print("=" * 78)
    if check is None:
        print("  not run")
    elif not check["compared"]:
        print("  not compared: %s" % check["error"])
    else:
        print("  id_code            : %s" % check["id_code"])
        print("  cached file        : %s" % check["cached_rgb_path"])
        print("  shape  cached/fresh: %s / %s  (match: %s)" % (
            check["cached_shape"], check["fresh_shape"], check["shapes_match"]))
        print("  dtype  cached/fresh: %s / %s  (match: %s)" % (
            check["cached_dtype"], check["fresh_dtype"], check["dtypes_match"]))
        if "max_abs_diff" in check:
            print("  max abs difference : %.10g" % check["max_abs_diff"])
            print("  mean abs difference: %.10g" % check["mean_abs_diff"])
            print("  exactly equal      : %s" % check["exactly_equal"])
            print("  bitwise identical  : %s" % check["bitwise_identical"])

    print("")
    print("=" * 78)
    print("INTEGRITY TRIPWIRE")
    print("=" * 78)
    if report["drive_write_violations"]:
        print("  FAILED -- the instrumented code wrote to a persistent (Drive) root:")
        for pass_name, id_code, path in report["drive_write_violations"]:
            print("    [%s] %s -> %s" % (pass_name, id_code, path))
    else:
        print("  PASS -- zero writes and zero mkdirs on any persistent (Drive) root across all passes.")
    probes = report["probe_stats"]
    print("  Diagnostic's OWN probes (excluded from every pipeline counter above): "
          "%d total (%d local, %d persistent)" % (probes["total"], probes["local"], probes["persistent"]))

    _print_classification(report)


def _print_classification(report):
    """PROVEN / LIKELY / UNKNOWN, each line generated from a recorded number. Nothing is asserted
    here that the run did not measure."""
    proven, likely, unknown = [], [], []
    summary = report["summary"]
    pass1 = summary["pass_totals"]["pass1"]

    for artifact in ARTIFACTS:
        stats = summary["artifacts"][artifact]
        proven.append("%s over %d image(s): %d local hit, %d persistent hit, %d recomputed, "
                      "%d missing (avg attributable %.3fs, max %.3fs)" % (
                          artifact, pass1["images"], stats["local_hits"], stats["persistent_hits"],
                          stats["computed"], stats["missing"], stats["avg_seconds"],
                          stats["max_seconds"]))

    roots = report["roots"]
    persistent = roots["persistent_cache_dir"]
    if persistent["exists"] and persistent["counts"]:
        counts = persistent["counts"]
        proven.append("persistent cache dir holds vessel=%d lesion=%d rgb=%d file(s)%s" % (
            counts.get("vessel", 0), counts.get("lesion", 0), counts.get("rgb", 0),
            " (listing TRUNCATED -- counts are lower bounds)" if persistent["truncated"] else ""))
        if counts.get("rgb", 0) == 0 and counts.get("vessel", 0) > 0:
            proven.append("the persistent cache contains vessel entries but ZERO canonical-RGB "
                          "entries -- RGB cannot be a persistent hit for any image there yet")
    elif persistent["abspath"] is None:
        proven.append("no persistent cache dir was configured for this run -- a persistent hit was "
                      "therefore impossible by construction, not by absence of files")
    elif not persistent["exists"]:
        proven.append("the configured persistent cache dir does NOT exist: " + persistent["abspath"])

    proven.append("pass 1 totals: %d Drive ops, %d Drive reads, %.3fs in Drive ops, %.3fs compute, "
                  "%d raw image load(s), %.3fs elapsed over %d image(s)" % (
                      pass1["drive_stats"] + pass1["drive_reads"] + pass1["drive_writes"],
                      pass1["drive_reads"], pass1["drive_seconds"], pass1["compute_seconds"],
                      pass1["raw_loads"], pass1["elapsed_seconds"], pass1["images"]))

    if report["pass2"]:
        pass2 = summary["pass_totals"]["pass2"]
        proven.append("pass 2 totals: %d Drive ops, %d Drive reads, %.3fs compute, %d raw load(s), "
                      "%.3fs elapsed" % (
                          pass2["drive_stats"] + pass2["drive_reads"] + pass2["drive_writes"],
                          pass2["drive_reads"], pass2["compute_seconds"], pass2["raw_loads"],
                          pass2["elapsed_seconds"]))
        if pass2["drive_reads"] == 0 and pass2["compute_seconds"] == 0 and pass2["raw_loads"] == 0:
            proven.append("pass 2 achieved a complete local working set: zero Drive content reads, "
                          "zero recomputation, zero raw image loads")
        else:
            offenders = sorted({artifact for record in report["pass2"] for artifact in ARTIFACTS
                                if record["artifacts"][artifact]["action"] != LOCAL_HIT})
            proven.append("pass 2 did NOT reach a pure local working set -- artifact(s) still not a "
                          "local hit: %s" % (", ".join(offenders) or "none identified"))

    if report["build_sample"]:
        drive_reads = sum(r["drive_reads"] for r in report["build_sample"])
        drive_stats = sum(r["drive_stats"] for r in report["build_sample"])
        computed = sum(1 for r in report["build_sample"] if r["compute_calls"])
        proven.append("training-time path (_build_joint_sample) over %d image(s): %d Drive stat(s), "
                      "%d Drive read(s), %d image(s) triggering recomputation" % (
                          len(report["build_sample"]), drive_stats, drive_reads, computed))

    if pass1["images"]:
        per_image = pass1["elapsed_seconds"] / float(pass1["images"])
        likely.append("at the measured pass-1 rate of %.2fs/image, a full 3662-image Phase 1 would "
                      "take about %.0f minutes -- an extrapolation from %d image(s), not a "
                      "measurement of the full run" % (per_image, per_image * 3662 / 60.0,
                                                       pass1["images"]))
    if pass1["compute_seconds"] > 0 and pass1["elapsed_seconds"] > 0:
        share = 100.0 * pass1["compute_seconds"] / pass1["elapsed_seconds"]
        likely.append("compute accounted for %.1f%% of pass-1 wall clock and Drive operations for "
                      "%.1f%% -- so the dominant cost of these images was %s" % (
                          share, 100.0 * pass1["drive_seconds"] / pass1["elapsed_seconds"],
                          "regeneration" if share >= 50.0 else "I/O or model/framework overhead"))

    unknown.append("whether these images are representative of all 3662 -- only %d were measured"
                   % pass1["images"])
    unknown.append("total real Phase 1 wall clock, which only a full Phase 1 run can establish")
    if not report["build_sample"]:
        unknown.append("training-time (_build_joint_sample) behavior -- not measured this run")
    if report["rgb_numerical_check"] is None or not report["rgb_numerical_check"]["compared"]:
        unknown.append("canonical RGB numerical equality -- no comparable cached file was available")

    print("")
    print("=" * 78)
    print("CLASSIFICATION (measurements only)")
    print("=" * 78)
    for label, items in (("PROVEN", proven), ("LIKELY", likely), ("UNKNOWN", unknown)):
        print("")
        print("%s:" % label)
        for item in items:
            print("  - %s" % item)
    print("")
    print("No root cause is declared here. The lines above are the observations; the diagnosis is "
          "made from them, not by this script.")
