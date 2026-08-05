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
staging only ever copies *out of* Drive, never writes back into it, and
nothing here modifies a single byte of the source dataset.

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
import os
import posixpath
import shutil
import time
from dataclasses import dataclass

LOCAL_DATASETS_ROOT = "/content/datasets"
DEFAULT_MAX_WORKERS = 16


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


def _copy_one(src, dst):
    os.makedirs(os.path.dirname(dst), exist_ok=True)
    shutil.copy2(src, dst)


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
