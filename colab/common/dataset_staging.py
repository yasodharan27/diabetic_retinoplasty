"""
Stages a dataset from its Google Drive master copy onto the Colab VM's
local SSD at the start of a session, so per-epoch training I/O never
crosses the slow Google Drive FUSE mount after the initial copy.

Google Drive's FUSE mount (`/content/drive`) is latency-bound per file
open, not bandwidth-bound -- reading thousands of small files from it once
per training epoch (this project's `tf.data` pipelines do not cache
decoded images, so every epoch re-reads every training image from its
source directory) is the dominant cost of a slow Colab training run, not
compute. Copying the dataset to local disk once, up front, turns every
subsequent epoch's reads into fast local I/O.

Generic and dataset-agnostic by design: callers (a stage notebook) pass in
which Drive directory to stage and what to call it locally -- nothing here
is hardcoded to EyeQ or any other dataset, so the same functions stage
EyeQ today and APTOS2019 / IDRiD once those stages are implemented. Which
dataset a given notebook stages is entirely up to that notebook (e.g.
`colab_config.EYEQ_RAW_DIR` for Stage 1) -- this module has no notion of
"the" dataset.

The Drive master copy is read-only from this module's perspective:
`stage_dataset()` only ever copies *out of* Drive, never writes back into
it, and nothing here modifies a single byte of the source dataset.

`sync_missing_files(source_dir, dest_dir)` is the incremental, direction-
agnostic sibling used for a *derived* directory that grows over time
rather than a fixed dataset staged once -- e.g. a per-image inference
cache written locally during a long Colab run (so every write is fast
local I/O, not Drive-latency-bound) and periodically flushed back up to
Drive. Unlike `stage_dataset()`, it copies only whatever is missing at the
destination and can run in either direction (Drive -> local to resume an
interrupted cache-precomputation run, or local -> Drive to persist newly
written entries), reusing the same latency-tolerant thread-pool copy.

IMPORTANT -- do not use `sync_missing_files()` to bulk-pull an entire persistent Drive cache
down to local disk "just in case" before deciding what to compute (`JOINT_TRAINING_ARCHITECTURE.md`
Sec 35): a Drive-persisted cache can grow to thousands of small files across repeated runs, and
copying all of them at once -- especially stacked immediately after another already-heavy
concurrent copy (e.g. dataset staging) -- is exactly the kind of sustained, high-concurrency burst
against Drive's FUSE mount that has caused a real Colab run to fail with
`OSError: [Errno 107] Transport endpoint is not connected` (the FUSE daemon disconnecting under
load). Determining whether a given cache entry already exists needs only a cheap `os.path.exists`
stat against its known path -- never its content -- so a caller that only needs to know "is this
already cached" should check directly against the persistent Drive directory instead of pulling
it down first; use `sync_missing_files()` for the (much smaller, session-bounded) push of newly
computed entries back up to Drive afterward.

Every individual copy (`_copy_one`, shared by `stage_dataset()` and `sync_missing_files()`) is
atomic -- written to a temp file, size-verified, then renamed into place -- and retries transient
Drive/FUSE errors (ENOTCONN and similar) with backoff, so a mid-copy Drive hiccup can never leave
a corrupt/partial file mistaken for a valid cache entry, and does not need to abort an entire
staging/sync run on its own.

Verifying the result of a copy is deliberately split in two:
  - `verify_staged_copy()` here is a generic, dataset-structure-agnostic
    check (file count + total bytes match between Drive source and local
    destination) that works for any dataset.
  - A dataset's own structural verifier (e.g.
    `verify_dataset.verify_eyeq_dataset()`) can be run again against the
    staged local directory for a deeper, dataset-aware check (labels.csv,
    per-split image counts, corruption spot-check) -- reusing that
    existing code completely unmodified, since it already accepts any
    `raw_dir` path and has no idea whether that path is on Drive or local
    disk.
"""

import concurrent.futures
import errno
import os
import posixpath
import random
import shutil
import time
from dataclasses import dataclass

LOCAL_DATASETS_ROOT = "/content/datasets"
DEFAULT_MAX_WORKERS = 16

# Errno values characteristic of a Google Drive FUSE hiccup under load (mount temporarily
# disconnected/stale), as opposed to a real, non-retryable failure (permission denied, disk
# full, file not found) -- retrying those would never help and would only hide a real problem.
# ENOTCONN ("Transport endpoint is not connected") is the specific error this project has hit in
# a real Colab run (JOINT_TRAINING_ARCHITECTURE.md Sec 35), caused by too many concurrent Drive
# file opens; ESTALE/ETIMEDOUT/EIO are the other well-documented FUSE-instability symptoms.
_TRANSIENT_ERRNOS = {errno.ENOTCONN, errno.ESTALE, errno.ETIMEDOUT, errno.EIO}
DEFAULT_RETRY_ATTEMPTS = 4
DEFAULT_RETRY_BASE_DELAY_SECONDS = 1.0


def _is_transient_os_error(exc):
    return isinstance(exc, OSError) and exc.errno in _TRANSIENT_ERRNOS


@dataclass(frozen=True)
class StagedDataset:
    """Result of staging one dataset onto the local SSD."""
    name: str
    drive_source_dir: str
    local_dir: str
    file_count: int
    total_bytes: int
    seconds_elapsed: float
    was_already_staged: bool


def _dir_stats(root):
    """(file_count, total_bytes) for every file under `root`, recursively."""
    file_count = 0
    total_bytes = 0
    for dirpath, _, filenames in os.walk(root):
        for name in filenames:
            file_count += 1
            total_bytes += os.path.getsize(os.path.join(dirpath, name))
    return file_count, total_bytes


def _copy_one(src, dst, retry_attempts=DEFAULT_RETRY_ATTEMPTS,
              retry_base_delay=DEFAULT_RETRY_BASE_DELAY_SECONDS):
    """Copies `src` to `dst` atomically: written to a same-directory temp file first, then
    `os.replace()`d into place, so a mid-copy failure (a Drive FUSE disconnect, a killed
    runtime) never leaves a partial/corrupt file at `dst` -- a half-written cache entry would
    otherwise be silently treated as a valid cache hit by every `os.path.exists` check in this
    project. The copied size is verified against the source before the rename as a cheap extra
    guard against a truncated read/write going undetected.

    Retries up to `retry_attempts` times, with exponential backoff, but ONLY for the errno
    values in `_TRANSIENT_ERRNOS` -- a Drive FUSE hiccup, not a real failure (wrong permissions,
    disk full, source missing), which is left to raise immediately since retrying it cannot
    help. This is what makes both `stage_dataset()` and `sync_missing_files()` tolerant of the
    same transient Drive instability that previously crashed a real Colab run with
    `OSError: [Errno 107] Transport endpoint is not connected` mid-copy."""
    os.makedirs(os.path.dirname(dst), exist_ok=True)
    tmp_dst = f"{dst}.tmp-{os.getpid()}-{random.randint(0, 1_000_000)}"
    last_error = None
    for attempt in range(retry_attempts):
        try:
            shutil.copy2(src, tmp_dst)
            if os.path.getsize(tmp_dst) != os.path.getsize(src):
                raise OSError(errno.EIO, f"Copied size mismatch for {src!r} -> {tmp_dst!r}")
            os.replace(tmp_dst, dst)
            return
        except OSError as exc:
            last_error = exc
            try:
                if os.path.exists(tmp_dst):
                    os.remove(tmp_dst)
            except OSError:
                pass
            if not _is_transient_os_error(exc) or attempt == retry_attempts - 1:
                raise
            time.sleep(retry_base_delay * (2 ** attempt))
    raise last_error  # pragma: no cover -- loop always returns or raises above


def stage_dataset(drive_source_dir, dataset_name, local_root=LOCAL_DATASETS_ROOT,
                   force=False, max_workers=DEFAULT_MAX_WORKERS):
    """
    Copy `drive_source_dir` (e.g. `colab_config.EYEQ_RAW_DIR`) to
    `<local_root>/<dataset_name>` on the Colab VM's local disk, and return
    a `StagedDataset` describing the result.

    Files are copied concurrently with a thread pool rather than one at a
    time: Drive's FUSE mount is latency-bound per file open, so overlapping
    many small-file copies cuts wall-clock staging time substantially
    versus a naive sequential copy, where each blocking file open is
    serialized behind the last. `max_workers` is tunable -- raise it for a
    (usually) faster copy, lower it if Drive starts throttling/erroring
    under high concurrency.

    Idempotent: if the local copy already exists and `force=False`, the
    copy is skipped entirely (a Colab session reconnect, or re-running a
    notebook's Dataset Loading cell, should not re-copy tens of thousands
    of files it already has). Pass `force=True` to re-stage from scratch
    (e.g. if the Drive-side dataset changed).

    Read-only with respect to Drive -- only ever reads `drive_source_dir`,
    never writes to it. Raises `RuntimeError` if `drive_source_dir` doesn't
    exist (nothing to stage), or propagates any individual copy failure
    (e.g. a mid-copy Drive disconnect) rather than silently producing a
    partial dataset.
    """
    if not os.path.isdir(drive_source_dir):
        raise RuntimeError(
            f"Cannot stage dataset {dataset_name!r}: {drive_source_dir} does not exist on Drive."
        )

    local_dir = posixpath.join(local_root, dataset_name)
    was_already_staged = os.path.isdir(local_dir) and not force
    elapsed = 0.0

    if was_already_staged:
        print(f"[{dataset_name}] already staged at {local_dir} -- skipping copy (force=True to re-copy).")
    else:
        if os.path.isdir(local_dir):
            shutil.rmtree(local_dir)
        os.makedirs(local_root, exist_ok=True)

        file_pairs = []
        for dirpath, _, filenames in os.walk(drive_source_dir):
            rel_dir = os.path.relpath(dirpath, drive_source_dir)
            dest_dir = local_dir if rel_dir == "." else os.path.join(local_dir, rel_dir)
            for name in filenames:
                file_pairs.append((os.path.join(dirpath, name), os.path.join(dest_dir, name)))

        print(f"[{dataset_name}] staging {len(file_pairs)} files: {drive_source_dir} -> {local_dir} "
              f"({max_workers} parallel workers) ...")
        start = time.time()
        with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as pool:
            futures = [pool.submit(_copy_one, src, dst) for src, dst in file_pairs]
            for future in concurrent.futures.as_completed(futures):
                future.result()  # re-raise immediately on any individual copy failure
        elapsed = time.time() - start
        print(f"[{dataset_name}] staged {len(file_pairs)} files in {elapsed:.1f}s "
              f"({len(file_pairs) / elapsed if elapsed > 0 else 0:.0f} files/s)")

    file_count, total_bytes = _dir_stats(local_dir)
    return StagedDataset(
        name=dataset_name,
        drive_source_dir=drive_source_dir,
        local_dir=local_dir,
        file_count=file_count,
        total_bytes=total_bytes,
        seconds_elapsed=elapsed,
        was_already_staged=was_already_staged,
    )


def sync_missing_files(source_dir, dest_dir, max_workers=DEFAULT_MAX_WORKERS):
    """
    Copies every file under `source_dir` that does not already exist at the corresponding
    relative path under `dest_dir` -- using the same latency-tolerant thread-pool-concurrency
    copy as `stage_dataset()` (`_copy_one`, reused unchanged), but INCREMENTAL and DIRECTION-
    AGNOSTIC rather than `stage_dataset()`'s all-or-nothing "already staged, skip everything"
    check. That all-or-nothing check fits a dataset staged once per session; it does not fit a
    cache directory that grows incrementally, one file at a time, across many runs and both
    directions (pulling an existing Drive cache down to local disk before a run, and pushing
    newly-written local cache entries back up to Drive during/after one).

    A file already present at its destination path is never re-copied, re-verified, or
    overwritten -- exactly this project's own "an existing cache entry is never recomputed"
    resumability convention, applied to the copy step itself rather than to what generates the
    file. Read-only with respect to `source_dir`; only ever creates new files under `dest_dir`,
    never deletes or modifies an existing one there. A `source_dir` that does not exist (e.g. no
    Drive cache has been written yet on a first-ever run) is treated as "nothing to copy", not an
    error.

    Each copy (`_copy_one`) is atomic (temp file + rename) and retries transient Drive/FUSE
    errors (ENOTCONN/"Transport endpoint is not connected" and similar) with backoff on its own
    -- see `_copy_one`'s docstring. If a file's copy still fails after those retries (a real,
    non-transient error), it is recorded in the returned `failures` list rather than aborting the
    whole call: every other independent file still gets copied, and a failed file is simply still
    missing from `dest_dir` afterward, so a later re-run of this same function will retry exactly
    that file again (the existing resumability guarantee already covers this -- nothing special
    has to be done to "resume" a partial sync).

    Returns `(copied_count, already_present_count, failures)`, where `failures` is a list of
    `(source_path, dest_path, error_message)` tuples (empty on full success).
    """
    if not os.path.isdir(source_dir):
        return 0, 0, []

    to_copy = []
    already_present = 0
    for dirpath, _, filenames in os.walk(source_dir):
        rel_dir = os.path.relpath(dirpath, source_dir)
        dest_subdir = dest_dir if rel_dir == "." else os.path.join(dest_dir, rel_dir)
        for name in filenames:
            dest_path = os.path.join(dest_subdir, name)
            if os.path.exists(dest_path):
                already_present += 1
            else:
                to_copy.append((os.path.join(dirpath, name), dest_path))

    failures = []
    if to_copy:
        with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as pool:
            future_to_pair = {pool.submit(_copy_one, src, dst): (src, dst) for src, dst in to_copy}
            for future in concurrent.futures.as_completed(future_to_pair):
                src, dst = future_to_pair[future]
                try:
                    future.result()
                except OSError as exc:
                    failures.append((src, dst, str(exc)))

    if failures:
        print(f"sync_missing_files: {len(failures)} of {len(to_copy)} file(s) failed to copy "
              f"(non-transient, or exhausted retries) -- they remain missing at the destination "
              f"and will be retried on the next call:")
        for src, dst, error in failures:
            print(f"  FAILED: {src} -> {dst}: {error}")

    return len(to_copy) - len(failures), already_present, failures


def verify_staged_copy(staged: StagedDataset):
    """
    Generic, dataset-structure-agnostic copy verification: re-counts files
    and total bytes under the Drive source and compares against the local
    copy. Raises `RuntimeError` with the specific mismatch on failure;
    returns `True` on success.

    This does not know about `labels.csv`, an `images/` subfolder, or any
    other dataset-specific convention -- for that, run the dataset's own
    verifier against the staged local directory as well, e.g.:

        staged = stage_dataset(colab_config.EYEQ_RAW_DIR, "EyeQ")
        verify_staged_copy(staged)
        verify_dataset.verify_eyeq_dataset(staged.local_dir)
    """
    source_count, source_bytes = _dir_stats(staged.drive_source_dir)
    if source_count != staged.file_count:
        raise RuntimeError(
            f"Staged copy verification failed for {staged.name!r}: Drive source has "
            f"{source_count} files, local copy has {staged.file_count}."
        )
    if source_bytes != staged.total_bytes:
        raise RuntimeError(
            f"Staged copy verification failed for {staged.name!r}: Drive source is "
            f"{source_bytes} bytes, local copy is {staged.total_bytes} bytes."
        )
    print(
        f"[{staged.name}] copy verified: {staged.file_count} files, "
        f"{staged.total_bytes / 1e6:.1f} MB match between Drive and local SSD."
    )
    return True
