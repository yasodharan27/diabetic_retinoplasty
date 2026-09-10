"""
MEASUREMENT-ONLY profiler for the joint Stage 05-08 + RACAF TRAINING step
(`joint_training_dataset.py` + `joint_training_model.py`). A sibling to
`joint_cache_diagnostics.py`, which profiles Phase 1; this one profiles Phase 2 -- the actual
per-batch training loop -- to answer one question from measurements rather than plausibility:
when a step takes ~4s instead of ~2s, WHERE does the time go.

Nothing here is imported by the pipeline: `joint_training_dataset.py`,
`joint_training_model.py`, `training/`, `racaf.py` and every notebook training cell are unchanged
and never import this module.

Design -- the decisive experiment is separation, not a single number:

  * PHASE C (input only) iterates the REAL `tf.data` pipeline with NO model attached. Its
    throughput is the ceiling the input pipeline can sustain.
  * PHASE D (compute only) runs the REAL model, loss and optimizer on ONE already-materialized
    batch, reused, so no dataset work happens at all. Its throughput is the ceiling the GPU can
    sustain.
  * PHASE E (combined) runs both together, timing `next(iterator)` separately from the train
    step, so input starvation is observed directly rather than inferred.

  With `prefetch`, a healthy pipeline gives combined ~= max(input, compute). If combined is close
  to input-only and much slower than compute-only, the input pipeline is the bottleneck; the
  reverse means the GPU is. If combined is much worse than BOTH, the two are contending (CPU, RAM
  or disk), which no single-phase measurement can reveal.

Safety -- this must not perturb the run it is diagnosing:

  * Model weights are snapshotted before and restored after, so the diagnostic's real gradient
    steps leave the model numerically where they found it. The optimizer's own slot/iteration
    state IS advanced, so rebuild + recompile the model before the real run (re-run the model
    construction cell); the report says so explicitly.
  * No `Trainer`, no callbacks, no `ModelCheckpoint`, no `TensorBoard`. Nothing is written to the
    experiment directory, so the diagnostic never writes to Drive on its own account.
  * Every filesystem operation the pipeline performs is recorded and classified local vs
    persistent, and a tripwire reports any write to a persistent root.
  * Cache files are read, never modified. A full local cache means the production path only
    reads; the tripwire proves it rather than assuming it.
"""

import os
import shutil
import subprocess
import posixpath
import tempfile
import threading
import time
import uuid

import numpy as np
import tensorflow as tf

import joint_cache_diagnostics as jcd
import joint_training_dataset as jtd

DEFAULT_BATCHES = 30


# =====================================================================
# 1. Storage / mount survey
# =====================================================================

def _run(command, timeout=120):
    """A shell command's stdout, or None. Never raises -- a survey must not abort the cell."""
    try:
        completed = subprocess.run(command, shell=True, capture_output=True, text=True,
                                   timeout=timeout)
    except Exception:  # noqa: BLE001
        return None
    return completed.stdout.strip() or None


def _statvfs_report(path):
    try:
        stats = os.statvfs(path)
    except (OSError, AttributeError):
        return None
    return {
        "total_bytes": stats.f_blocks * stats.f_frsize,
        "free_bytes": stats.f_bavail * stats.f_frsize,
        "used_bytes": (stats.f_blocks - stats.f_bfree) * stats.f_frsize,
        "inodes_total": stats.f_files,
        "inodes_free": stats.f_favail,
        "inodes_used": (stats.f_files - stats.f_ffree) if stats.f_files else None,
    }


def survey_storage(paths, du_root="/content", du_depth=2):
    """Free/used space, filesystem type and inode pressure for every interesting path, plus the
    largest directories under `du_root`. `du` is bounded by `du_depth` and never descends into
    `/content/drive` -- walking a Drive FUSE mount to size it would itself take minutes."""
    report = {"paths": {}, "largest_dirs": None, "df": _run("df -h"), "du_root": du_root}
    for label, path in paths.items():
        entry = {
            "path": path,
            "exists": os.path.exists(path) if path else False,
            "filesystem": jcd._filesystem_of(path) if path else None,
            "statvfs": _statvfs_report(path) if path and os.path.exists(path) else None,
            "size_bytes": None,
            "file_count": None,
        }
        # Directory size via `du` (fast, one pass) -- skipped for the Drive mount.
        if entry["exists"] and path and not path.rstrip("/").startswith("/content/drive"):
            out = _run(f"du -sb {path!r} 2>/dev/null")
            if out:
                try:
                    entry["size_bytes"] = int(out.split()[0])
                except (ValueError, IndexError):
                    pass
            count = _run(f"find {path!r} -type f 2>/dev/null | wc -l")
            if count:
                try:
                    entry["file_count"] = int(count.split()[0])
                except (ValueError, IndexError):
                    pass
        report["paths"][label] = entry

    if os.path.isdir(du_root):
        report["largest_dirs"] = _run(
            f"du -h --max-depth={du_depth} --exclude=/content/drive {du_root!r} 2>/dev/null "
            "| sort -rh | head -25", timeout=300,
        )
    return report


# =====================================================================
# 2. GPU / CPU telemetry sampled DURING training (same process, no second cell)
# =====================================================================

class _TelemetrySampler:
    """Samples nvidia-smi and psutil on a background thread while the measured loop runs in the
    foreground. This is a subprocess of the training process, not a competing notebook cell, so it
    observes the GPU while it is actually under load -- the gap a post-run nvidia-smi cannot fill."""

    def __init__(self, interval=0.25):
        self.interval = interval
        self.gpu_util = []
        self.gpu_mem_used_mb = []
        self.gpu_mem_total_mb = None
        self.cpu_percent = []
        self.ram_used_gb = []
        self._stop = threading.Event()
        self._thread = None
        self._psutil = None
        try:
            import psutil
            self._psutil = psutil
        except ImportError:
            pass

    def _sample_gpu(self):
        out = _run("nvidia-smi --query-gpu=utilization.gpu,memory.used,memory.total "
                   "--format=csv,noheader,nounits", timeout=10)
        if not out:
            return
        try:
            util, used, total = (part.strip() for part in out.splitlines()[0].split(","))
            self.gpu_util.append(float(util))
            self.gpu_mem_used_mb.append(float(used))
            self.gpu_mem_total_mb = float(total)
        except (ValueError, IndexError):
            pass

    def _loop(self):
        if self._psutil is not None:
            self._psutil.cpu_percent(interval=None)  # prime the counter
        while not self._stop.is_set():
            self._sample_gpu()
            if self._psutil is not None:
                try:
                    self.cpu_percent.append(self._psutil.cpu_percent(interval=None))
                    self.ram_used_gb.append(self._psutil.virtual_memory().used / float(1024 ** 3))
                except Exception:  # noqa: BLE001
                    pass
            self._stop.wait(self.interval)

    def start(self):
        self._stop.clear()
        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()
        return self

    def stop(self):
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=5)
        return self.summary()

    def summary(self):
        def stats(values):
            if not values:
                return None
            return {"mean": sum(values) / len(values), "max": max(values), "min": min(values),
                    "samples": len(values)}
        return {
            "gpu_util_percent": stats(self.gpu_util),
            "gpu_mem_used_mb": stats(self.gpu_mem_used_mb),
            "gpu_mem_total_mb": self.gpu_mem_total_mb,
            "cpu_percent": stats(self.cpu_percent),
            "ram_used_gb": stats(self.ram_used_gb),
            "psutil_available": self._psutil is not None,
        }


def tf_gpu_memory():
    try:
        if not tf.config.list_physical_devices("GPU"):
            return None
        info = tf.config.experimental.get_memory_info("GPU:0")
        return {"current_mb": info["current"] / (1024.0 ** 2),
                "peak_mb": info.get("peak", 0) / (1024.0 ** 2)}
    except Exception:  # noqa: BLE001
        return None


# =====================================================================
# 3. Runtime configuration snapshot (for the previous-vs-current comparison)
# =====================================================================

def snapshot_configuration(model=None):
    """Every runtime setting that could plausibly differ between two runs, read from the live
    process rather than from what the notebook says it configured."""
    snapshot = {
        "tensorflow_version": tf.__version__,
        "keras_version": getattr(tf.keras, "__version__", None),
        "gpu_devices": [device.name for device in tf.config.list_physical_devices("GPU")],
        "gpu_details": None,
        "mixed_precision_policy": None,
        "eager_execution": tf.executing_eagerly(),
        "intra_op_threads": tf.config.threading.get_intra_op_parallelism_threads(),
        "inter_op_threads": tf.config.threading.get_inter_op_parallelism_threads(),
        "cpu_count": os.cpu_count(),
        "memory_growth": None,
        "tf_gpu_memory": tf_gpu_memory(),
        "shuffle_buffer_size": jtd.DEFAULT_SHUFFLE_BUFFER_SIZE,
        "stage5_image_size": jtd.STAGE5_IMAGE_SIZE,
        "stage6_image_size": jtd.STAGE6_IMAGE_SIZE,
        "env": {name: os.environ.get(name) for name in
                ("TF_GPU_ALLOCATOR", "TF_FORCE_GPU_ALLOW_GROWTH", "XLA_FLAGS",
                 "TF_XLA_FLAGS", "TF_ENABLE_ONEDNN_OPTS", "CUDA_VISIBLE_DEVICES")},
    }
    try:
        snapshot["mixed_precision_policy"] = tf.keras.mixed_precision.global_policy().name
    except Exception:  # noqa: BLE001
        pass
    try:
        devices = tf.config.list_physical_devices("GPU")
        if devices:
            snapshot["gpu_details"] = tf.config.experimental.get_device_details(devices[0])
            snapshot["memory_growth"] = tf.config.experimental.get_memory_growth(devices[0])
    except Exception:  # noqa: BLE001
        pass
    snapshot["nvidia_smi"] = _run("nvidia-smi --query-gpu=name,driver_version,memory.total,"
                                  "utilization.gpu --format=csv,noheader")

    if model is not None:
        try:
            trainable = int(sum(np.prod(v.shape) for v in model.trainable_variables))
            non_trainable = int(sum(np.prod(v.shape) for v in model.non_trainable_variables))
            optimizer = getattr(model, "optimizer", None)
            snapshot["model"] = {
                "total_params": int(model.count_params()),
                "trainable_params": trainable,
                "non_trainable_params": non_trainable,
                "num_trainable_tensors": len(model.trainable_variables),
                "loss": getattr(getattr(model, "loss", None), "__name__", str(getattr(model, "loss", None))),
                "optimizer": type(optimizer).__name__ if optimizer is not None else None,
                "learning_rate": (float(tf.keras.backend.get_value(optimizer.learning_rate))
                                  if optimizer is not None and hasattr(optimizer, "learning_rate") else None),
                "jit_compile": getattr(model, "jit_compile", None),
                "steps_per_execution": (int(tf.keras.backend.get_value(model.steps_per_execution))
                                        if getattr(model, "steps_per_execution", None) is not None else None),
                "dtype_policy": getattr(getattr(model, "dtype_policy", None), "name", None),
            }
        except Exception as error:  # noqa: BLE001
            snapshot["model"] = {"error": repr(error)}
    return snapshot


# =====================================================================
# 4. The three measurement phases
# =====================================================================

def describe_pipeline_failure(error):
    """Turns whatever surfaced out of `next(iterator)` into a classified diagnosis.

    A cache failure inside the `from_generator` callback reaches the consumer wrapped as
    `tf.errors.UnknownError: ... IteratorGetNext ...`, whose message concatenates the original
    Python traceback as text. That hides the two facts that matter -- WHICH artifact and WHICH
    image -- behind a TensorFlow frame, which is exactly why the first Phase C attempt was hard to
    read. This walks the `__cause__`/`__context__` chain for a typed cache error and, failing
    that, pattern-matches the wrapped message.

    Returns a dict whose `category` is one of the classes the profiler must distinguish:
    DRIVE_FUSE (B), CORRUPT_CACHE, LOCAL_CACHE (A), or UNKNOWN."""
    chain, seen, current = [], set(), error
    while current is not None and id(current) not in seen:
        seen.add(id(current))
        chain.append(current)
        current = current.__cause__ or current.__context__

    for item in chain:
        if isinstance(item, jtd.PersistentCacheUnavailableError):
            return {
                "category": "DRIVE_FUSE",
                "artifact": item.artifact, "id_code": item.id_code, "path": item.path,
                "errno": item.errno, "local_path": item.local_path,
                "local_exists": item.local_exists,
                "message": str(item),
                "meaning": "The persistent Drive mount failed while the LOCAL entry for this "
                           "image was missing. This is a Drive/FUSE problem (B), not a local "
                           "cache problem -- and nothing was recomputed.",
            }
        if isinstance(item, jtd.CorruptCacheFileError):
            return {
                "category": "CORRUPT_CACHE",
                "artifact": item.artifact, "id_code": item.id_code, "path": item.path,
                "persistent": item.persistent, "detail": item.detail,
                "message": str(item),
                "meaning": "A cache file was reachable but unusable. On a persistent path this is "
                           "usually a truncated FUSE read rather than genuine corruption; the "
                           "file was left untouched either way.",
            }

    text = str(error)
    if "Transport endpoint is not connected" in text or "Errno 107" in text:
        return {"category": "DRIVE_FUSE", "message": text,
                "meaning": "Drive FUSE dropped (errno 107). Raised from below the typed-error "
                           "layer -- re-mount Drive before re-running."}
    if "cannot reshape array of size" in text:
        return {"category": "CORRUPT_CACHE", "message": text,
                "meaning": "A truncated read (NumPy could not reshape a short buffer). Almost "
                           "always a partial FUSE read of a persistent file."}
    return {"category": "UNKNOWN", "message": text,
            "meaning": "Not a recognized cache failure -- see the message."}


def preflight_cache_audit(entries, cache_dir, racaf_cache_dir, image_size,
                          persistent_cache_dir=None, persistent_racaf_cache_dir=None,
                          sample_limit=None):
    """PER-IMAGE, PER-ARTIFACT existence audit run BEFORE any measurement -- the question a
    directory-level count cannot answer.

    Phase C failed with `Errno 107` reading a PERSISTENT lesion path. Under the documented
    precedence that can only happen when the LOCAL entry for that specific image is missing, so
    the pipeline correctly fell through to the Drive fallback -- and doing that for thousands of
    images in a row is what takes a FUSE mount down. Whether the local cache is complete for the
    exact entries the profiler will iterate is therefore the first thing to establish, and it is
    established here by stat-ing every artifact of every entry rather than by counting files in a
    directory.

    Uses the REAL path builders (`lfed._cache_path`, `racaf.reliability_cache_path`,
    `jtd._canonical_rgb_cache_path`) so a filename-convention mismatch would show up as a miss
    here exactly as it would in training. Local paths are stat'ed with plain `os.path.exists`;
    persistent paths go through `jtd._persistent_exists`, so a dead mount is reported as
    UNREACHABLE rather than silently as "absent".

    These are the PROFILER's own probes and are never mixed into the pipeline operation counts."""
    entries = list(entries)
    if sample_limit is not None:
        entries = entries[:sample_limit]
    result = {
        "entries_checked": len(entries),
        "fully_local": 0,
        "would_read_drive": 0,
        "missing_everywhere": 0,
        "drive_unreachable": False,
        "drive_error": None,
        "per_artifact_local_missing": {artifact: 0 for artifact in jcd.ARTIFACTS},
        "per_artifact_persistent_missing": {artifact: 0 for artifact in jcd.ARTIFACTS},
        "examples_needing_drive": [],
        "examples_missing_everywhere": [],
        "probe_count": 0,
    }
    have_persistent = persistent_cache_dir is not None and persistent_racaf_cache_dir is not None

    for id_code, _diagnosis in entries:
        local = jcd.artifact_paths(id_code, cache_dir, racaf_cache_dir, image_size)
        missing_local = []
        for artifact, path in local.items():
            result["probe_count"] += 1
            if not os.path.exists(path):
                missing_local.append(artifact)
                result["per_artifact_local_missing"][artifact] += 1
        if not missing_local:
            result["fully_local"] += 1
            continue

        if not have_persistent:
            result["missing_everywhere"] += 1
            if len(result["examples_missing_everywhere"]) < 10:
                result["examples_missing_everywhere"].append((id_code, missing_local))
            continue

        persistent = jcd.artifact_paths(id_code, persistent_cache_dir,
                                        persistent_racaf_cache_dir, image_size)
        missing_persistent = []
        for artifact in missing_local:
            result["probe_count"] += 1
            try:
                present = jtd._persistent_exists(persistent[artifact], artifact, id_code)
            except jtd.PersistentCacheUnavailableError as error:
                # The mount is down right now. Stop probing -- hammering it makes it worse.
                result["drive_unreachable"] = True
                result["drive_error"] = str(error)
                return result
            if not present:
                missing_persistent.append(artifact)
                result["per_artifact_persistent_missing"][artifact] += 1
        if missing_persistent:
            result["missing_everywhere"] += 1
            if len(result["examples_missing_everywhere"]) < 10:
                result["examples_missing_everywhere"].append((id_code, missing_persistent))
        else:
            result["would_read_drive"] += 1
            if len(result["examples_needing_drive"]) < 10:
                result["examples_needing_drive"].append((id_code, missing_local))
    return result


def _roots_for(cache_dir, racaf_cache_dir, persistent_cache_dir, persistent_racaf_cache_dir,
               image_dir, drive_mount="/content/drive"):
    return jcd._Roots([
        ("local_cache", cache_dir, False),
        ("local_racaf_cache", racaf_cache_dir, False),
        ("staged_images", image_dir, False),
        ("drive_cache", persistent_cache_dir, True),
        ("drive_racaf_cache", persistent_racaf_cache_dir, True),
        ("drive_mount", drive_mount, True),
    ])


def profile_input_only(dataset, roots, batches=DEFAULT_BATCHES):
    """PHASE C -- the real tf.data pipeline with NO model attached. Every filesystem operation the
    generator performs is recorded and attributed per artifact, so cache I/O is separated from the
    rest of per-sample work rather than lumped into one 'input' figure."""
    recorder = jcd._Recorder(roots)
    telemetry = _TelemetrySampler().start()
    per_batch = []
    failure = None
    iterator = iter(dataset)
    start_all = time.perf_counter()
    with jcd._instrument(recorder):
        # The first batch pays one-time costs (generator start, shuffle-buffer fill), so it is
        # timed but reported separately rather than averaged into steady state.
        for _ in range(batches):
            start = time.perf_counter()
            try:
                next(iterator)
            except StopIteration:
                break
            except Exception as error:  # noqa: BLE001 -- classified, then re-reported, below
                # A cache failure inside the generator reaches us wrapped in tf.errors.UnknownError
                # ("IteratorGetNext..."), which hides which artifact and which image failed.
                # `describe_pipeline_failure` digs the real cause back out so Phase C ends with a
                # usable diagnosis instead of an opaque TensorFlow traceback.
                failure = describe_pipeline_failure(error)
                break
            per_batch.append(time.perf_counter() - start)
    total = time.perf_counter() - start_all
    return {
        "batches": len(per_batch),
        "total_seconds": total,
        "first_batch_seconds": per_batch[0] if per_batch else None,
        "per_batch_seconds": per_batch,
        "steady_mean_seconds": (sum(per_batch[1:]) / len(per_batch[1:])) if len(per_batch) > 1 else None,
        "telemetry": telemetry.stop(),
        "failure": failure,
        "recorder": recorder,
    }


def profile_compute_only(model, batch, batches=DEFAULT_BATCHES):
    """PHASE D -- the real model, loss and optimizer on ONE already-materialized batch, reused, so
    zero dataset work happens. Weights are restored afterward.

    Forward and backward are timed separately by running the forward pass alone first, then the
    full taped step: backward is (taped step - forward), which is the only decomposition available
    without a full TF profiler trace. On GPU this needs an explicit synchronization per timing
    boundary, done by forcing a host read of a scalar derived from the result."""
    (inputs, labels) = batch
    saved_weights = model.get_weights()
    optimizer = model.optimizer
    loss_fn = model.loss

    def sync(tensor):
        # Forces the async GPU queue to drain, so the timing boundary is real.
        return float(tf.reduce_sum(tf.cast(tensor, tf.float32)).numpy())

    forward_times, step_times, apply_times = [], [], []
    telemetry = _TelemetrySampler().start()
    # Warm-up: first call builds/compiles the graph and autotunes cuDNN. Never averaged in.
    warm_start = time.perf_counter()
    with tf.GradientTape() as tape:
        outputs = model(inputs, training=True)
        loss = loss_fn(labels, outputs)
    grads = tape.gradient(loss, model.trainable_variables)
    optimizer.apply_gradients(zip(grads, model.trainable_variables))
    sync(outputs)
    warmup_seconds = time.perf_counter() - warm_start

    start_all = time.perf_counter()
    for _ in range(batches):
        start = time.perf_counter()
        outputs = model(inputs, training=True)
        sync(outputs)
        forward_times.append(time.perf_counter() - start)

        start = time.perf_counter()
        with tf.GradientTape() as tape:
            outputs = model(inputs, training=True)
            loss = loss_fn(labels, outputs)
        grads = tape.gradient(loss, model.trainable_variables)
        sync(grads[0])
        step_times.append(time.perf_counter() - start)

        start = time.perf_counter()
        optimizer.apply_gradients(zip(grads, model.trainable_variables))
        sync(model.trainable_variables[0])
        apply_times.append(time.perf_counter() - start)
    total = time.perf_counter() - start_all
    summary = telemetry.stop()
    model.set_weights(saved_weights)  # leave the model numerically where we found it

    def mean(values):
        return sum(values) / len(values) if values else None
    forward = mean(forward_times)
    taped = mean(step_times)
    return {
        "batches": len(step_times),
        "total_seconds": total,
        "warmup_seconds": warmup_seconds,
        "forward_seconds": forward,
        "forward_and_backward_seconds": taped,
        "backward_seconds": (taped - forward) if (taped is not None and forward is not None) else None,
        "optimizer_seconds": mean(apply_times),
        "per_step_seconds": (total / len(step_times)) if step_times else None,
        "telemetry": summary,
        "tf_gpu_memory": tf_gpu_memory(),
    }


def profile_combined(model, dataset, roots, batches=DEFAULT_BATCHES):
    """PHASE E -- dataset and model together, the real training loop shape. `next(iterator)` is
    timed separately from the train step, so input starvation is observed directly. Weights are
    restored afterward."""
    saved_weights = model.get_weights()
    optimizer = model.optimizer
    loss_fn = model.loss
    recorder = jcd._Recorder(roots)
    telemetry = _TelemetrySampler().start()

    def sync(tensor):
        return float(tf.reduce_sum(tf.cast(tensor, tf.float32)).numpy())

    wait_times, compute_times, batch_times = [], [], []
    failure = None
    iterator = iter(dataset)
    start_all = time.perf_counter()
    with jcd._instrument(recorder):
        for _ in range(batches):
            batch_start = time.perf_counter()
            start = time.perf_counter()
            try:
                inputs, labels = next(iterator)
            except StopIteration:
                break
            except Exception as error:  # noqa: BLE001 -- classified by describe_pipeline_failure
                failure = describe_pipeline_failure(error)
                break
            wait_times.append(time.perf_counter() - start)

            start = time.perf_counter()
            with tf.GradientTape() as tape:
                outputs = model(inputs, training=True)
                loss = loss_fn(labels, outputs)
            grads = tape.gradient(loss, model.trainable_variables)
            optimizer.apply_gradients(zip(grads, model.trainable_variables))
            sync(outputs)
            compute_times.append(time.perf_counter() - start)
            batch_times.append(time.perf_counter() - batch_start)
    total = time.perf_counter() - start_all
    summary = telemetry.stop()
    model.set_weights(saved_weights)

    def mean(values, skip_first=True):
        values = values[1:] if (skip_first and len(values) > 1) else values
        return sum(values) / len(values) if values else None
    return {
        "batches": len(batch_times),
        "total_seconds": total,
        "first_batch_seconds": batch_times[0] if batch_times else None,
        "input_wait_seconds": mean(wait_times),
        "compute_seconds": mean(compute_times),
        "batch_seconds": mean(batch_times),
        "per_batch_seconds": batch_times,
        "telemetry": summary,
        "tf_gpu_memory": tf_gpu_memory(),
        "failure": failure,
        "recorder": recorder,
    }


# =====================================================================
# 4b. PHASE F -- step breakdown with correct asynchronous attribution
# =====================================================================
#
# Why this exists alongside Phase D, rather than replacing it.
#
# Phase D answers "roughly where does a step go" and its per-section syncs drain ONE tensor
# (`grads[0]`, `trainable_variables[0]`). TensorFlow's eager execution is asynchronous: an op
# returns a handle as soon as it is enqueued, and reading any tensor only forces the ops that
# tensor depends on. `grads[0]` is the gradient of the FIRST trainable variable, which backprop
# produces LAST, so draining it happens to cover most of the backward pass -- but that is a
# property of variable ordering, not a guarantee, and `trainable_variables[0]` is updated near
# the START of `apply_gradients`, so draining it does NOT cover the other 392 updates. Work that
# is still in flight when a timing boundary is taken lands in whichever section drains next.
# That is exactly the misattribution to rule out before believing "the optimizer takes 2.1 s".
#
# Phase F therefore:
#   * drains EVERY tensor a section produces, not one of them, and measures the drain's own cost
#     separately so it can be subtracted rather than silently included;
#   * times `apply_gradients` BOTH with and without a drain. Enqueue-only time is CPU work
#     (Python, op dispatch); drain time is GPU work. That single pair answers "is the optimizer
#     CPU-bound or GPU-bound" directly instead of by inference;
#   * records `time.process_time()` (CPU time actually burned by this process) beside wall time
#     for every section. CPU/wall near 1.0 means the host is busy; near 0.0 means it is blocked
#     waiting on the device;
#   * cross-checks the whole decomposition against a single-drain run of the same work. If the
#     per-section sum matches the one-drain total, the per-section boundaries are real;
#   * measures the COMPILED path (`model.train_on_batch`, which Keras runs inside a `tf.function`)
#     next to the eager tape loop. `Trainer.fit()` -> `model.fit()` uses the compiled path, so an
#     eager-only measurement is not measuring what training does.
#
# Measurement only: no architecture, optimizer, batch size, loss, precision policy or JIT setting
# is changed here. Weights are snapshotted and restored; the optimizer's slot state IS advanced,
# so rebuild and recompile before a real run.


def _grad_tensor(gradient):
    """Dense tensor for a gradient that may be a `tf.IndexedSlices` (sparse) update."""
    return gradient.values if isinstance(gradient, tf.IndexedSlices) else gradient


def _dtype_name(tensor):
    """Keras 3 `Variable.dtype` is a plain string while `tf.Tensor.dtype` is a `tf.DType`, and
    this inventory mixes both. Normalize to the name."""
    dtype = getattr(tensor, "dtype", None)
    return getattr(dtype, "name", None) or str(dtype)


def _dtype_bytes(tensor):
    try:
        return tf.as_dtype(_dtype_name(tensor)).size
    except Exception:  # noqa: BLE001
        return 4


def _element_count(tensor):
    shape = getattr(tensor, "shape", None)
    if shape is None:
        return 0
    try:
        dimensions = [int(d) for d in shape]
    except TypeError:
        return 0
    return int(np.prod(dimensions)) if dimensions else 1


def _drain(tensors):
    """Forces every op producing `tensors` to complete, with ONE host read.

    `tf.add_n` over per-tensor reductions makes the single scalar depend on all of them, so the
    `.numpy()` cannot return until the whole set is materialized -- unlike reading one tensor,
    which leaves the rest in flight. Cast to float32 first so a mixed-precision (float16) or
    integer tensor is handled identically."""
    parts = [tf.reduce_sum(tf.cast(_grad_tensor(t), tf.float32))
             for t in tensors if t is not None]
    if not parts:
        return 0.0
    return float(tf.add_n(parts).numpy())


class _Section:
    """Wall time, process CPU time and (optionally) GPU telemetry for one measured region."""

    def __init__(self, name):
        self.name = name
        self.wall = []
        self.cpu = []
        self._wall0 = None
        self._cpu0 = None

    def __enter__(self):
        self._wall0 = time.perf_counter()
        self._cpu0 = time.process_time()
        return self

    def __exit__(self, *exc):
        self.wall.append(time.perf_counter() - self._wall0)
        self.cpu.append(time.process_time() - self._cpu0)
        return False

    def summary(self, skip_first=True):
        wall = self.wall[1:] if (skip_first and len(self.wall) > 1) else self.wall
        cpu = self.cpu[1:] if (skip_first and len(self.cpu) > 1) else self.cpu
        if not wall:
            return None
        mean_wall = sum(wall) / len(wall)
        mean_cpu = sum(cpu) / len(cpu)
        return {
            "samples": len(wall),
            "wall_seconds": mean_wall,
            "cpu_seconds": mean_cpu,
            "cpu_over_wall": (mean_cpu / mean_wall) if mean_wall else None,
            "min_wall_seconds": min(wall),
            "max_wall_seconds": max(wall),
            # Every individual iteration, so a one-off outlier is visible rather than averaged
            # into a mean that then looks like steady state.
            "all_wall_seconds": list(wall),
        }


def describe_training_stack(model, gradients=None):
    """The static inventory the step breakdown has to be read against: how many tensors the
    optimizer actually updates, what dtypes they are, how many slot variables back them, whether
    any gradient processing is configured, and whether anything is compiled or XLA'd.

    Read from the live objects, never from what the notebook says it configured."""
    optimizer = getattr(model, "optimizer", None)
    trainable = list(model.trainable_variables)

    def inventory(tensors):
        by_dtype, total_elements, total_bytes = {}, 0, 0
        for tensor in tensors:
            if tensor is None:
                continue
            dense = _grad_tensor(tensor)
            dtype = _dtype_name(dense)
            by_dtype[dtype] = by_dtype.get(dtype, 0) + 1
            elements = _element_count(dense)
            total_elements += elements
            total_bytes += elements * _dtype_bytes(dense)
        return {"count": sum(by_dtype.values()), "by_dtype": by_dtype,
                "elements": total_elements, "bytes": total_bytes}

    report = {
        "trainable": inventory(trainable),
        "gradients": inventory(gradients) if gradients is not None else None,
        "none_gradients": (sum(1 for g in gradients if g is None)
                           if gradients is not None else None),
        "largest_trainable": sorted(
            ((_element_count(v), getattr(v, "path", None) or v.name, tuple(v.shape),
              _dtype_name(v)) for v in trainable), reverse=True)[:10],
        "optimizer": None,
        "jit": {
            "model_jit_compile": getattr(model, "jit_compile", None),
            "global_jit": None,
            "XLA_FLAGS": os.environ.get("XLA_FLAGS"),
            "TF_XLA_FLAGS": os.environ.get("TF_XLA_FLAGS"),
        },
        "dtype_policy": getattr(getattr(model, "dtype_policy", None), "name", None),
        "global_policy": None,
        "run_eagerly": getattr(model, "run_eagerly", None),
        "steps_per_execution": None,
    }
    try:
        report["jit"]["global_jit"] = tf.config.optimizer.get_jit()
    except Exception:  # noqa: BLE001
        pass
    try:
        report["global_policy"] = tf.keras.mixed_precision.global_policy().name
    except Exception:  # noqa: BLE001
        pass
    try:
        value = getattr(model, "steps_per_execution", None)
        report["steps_per_execution"] = int(value) if value is not None else None
    except Exception:  # noqa: BLE001
        pass

    if optimizer is not None:
        slots = list(getattr(optimizer, "variables", []) or [])
        slot_by_dtype = {}
        slot_bytes = 0
        for variable in slots:
            dtype = _dtype_name(variable)
            slot_by_dtype[dtype] = slot_by_dtype.get(dtype, 0) + 1
            slot_bytes += _element_count(variable) * _dtype_bytes(variable)
        report["optimizer"] = {
            "type": type(optimizer).__name__,
            "inner_optimizer": type(getattr(optimizer, "inner_optimizer", None)).__name__
                               if getattr(optimizer, "inner_optimizer", None) is not None else None,
            "loss_scale_factor": getattr(optimizer, "loss_scale_factor", None),
            "clipnorm": getattr(optimizer, "clipnorm", None),
            "clipvalue": getattr(optimizer, "clipvalue", None),
            "global_clipnorm": getattr(optimizer, "global_clipnorm", None),
            "use_ema": getattr(optimizer, "use_ema", None),
            "slot_variables": len(slots),
            "slot_variables_by_dtype": slot_by_dtype,
            "slot_bytes": slot_bytes,
            "iterations": None,
        }
        try:
            report["optimizer"]["iterations"] = int(optimizer.iterations)
        except Exception:  # noqa: BLE001
            pass
    return report


def profile_optimizer_breakdown(model, batch, batches=DEFAULT_BATCHES):
    """PHASE F -- see this section's header. One already-materialized batch, reused, so no dataset
    work happens and every number is model/optimizer time. Returns a plain dict.

    The `apply_gradients` section deliberately re-applies the SAME gradient tensors: the ops, their
    shapes and their memory traffic are identical to a real step, and holding the gradients fixed
    is what isolates the optimizer from the backward pass that would otherwise be recomputed inside
    the timed region. The resulting weights are meaningless, which is why they are restored."""
    (inputs, labels) = batch
    # `tf.data` yields the three model inputs as a TUPLE. Keras reads a tuple `x` as a nested
    # structure in some paths and as `(x, y)` in others, so normalize to a list once here --
    # the tensors themselves are untouched.
    if isinstance(inputs, tuple):
        inputs = list(inputs)
    saved_weights = model.get_weights()
    optimizer = model.optimizer
    loss_fn = model.loss
    trainable = list(model.trainable_variables)

    report = {"batches": batches, "sections": {}, "telemetry": {}, "notes": []}

    def taped_gradients():
        with tf.GradientTape() as tape:
            outputs = model(inputs, training=True)
            loss = loss_fn(labels, outputs)
        return outputs, loss, tape.gradient(loss, trainable)

    # --- warm-up: graph build, cuDNN autotune, slot creation. Never measured. ---------------
    warm = time.perf_counter()
    outputs, loss, gradients = taped_gradients()
    optimizer.apply_gradients(zip(gradients, trainable))
    _drain(trainable)
    report["warmup_seconds"] = time.perf_counter() - warm
    report["stack"] = describe_training_stack(model, gradients)

    # --- the cost of the drain itself, on tensors that are ALREADY materialized -------------
    # Subtractable overhead: whatever this costs is inside every drained section below.
    drain_grads = _Section("drain_gradients")
    drain_vars = _Section("drain_variables")
    for _ in range(max(3, min(batches, 5))):
        with drain_grads:
            _drain(gradients)
        with drain_vars:
            _drain(trainable)
    report["sections"]["drain_gradients_overhead"] = drain_grads.summary()
    report["sections"]["drain_variables_overhead"] = drain_vars.summary()

    # --- F1: forward only -------------------------------------------------------------------
    forward = _Section("forward")
    sampler = _TelemetrySampler().start()
    for _ in range(batches):
        with forward:
            out = model(inputs, training=True)
            _drain([out])
    report["telemetry"]["forward"] = sampler.stop()
    report["sections"]["forward"] = forward.summary()

    # --- F2: forward + gradient computation --------------------------------------------------
    forward_grad = _Section("forward_and_gradients")
    sampler = _TelemetrySampler().start()
    for _ in range(batches):
        with forward_grad:
            _outputs, _loss, gradients = taped_gradients()
            _drain(gradients)
    report["telemetry"]["forward_and_gradients"] = sampler.stop()
    report["sections"]["forward_and_gradients"] = forward_grad.summary()

    # --- F3: apply_gradients, enqueue-only vs drained ---------------------------------------
    # `apply_enqueue` measures the host returning from apply_gradients with device work possibly
    # still in flight -- pure CPU/dispatch cost. `apply_drain` is what remained on the device.
    # Their sum is the honest total; their ratio is the CPU-bound / GPU-bound answer.
    apply_enqueue = _Section("apply_gradients_enqueue")
    apply_drain = _Section("apply_gradients_drain")
    apply_total = _Section("apply_gradients_total")
    iterations_before = None
    try:
        iterations_before = int(optimizer.iterations)
    except Exception:  # noqa: BLE001
        pass
    sampler = _TelemetrySampler().start()
    for _ in range(batches):
        with apply_total:
            with apply_enqueue:
                optimizer.apply_gradients(zip(gradients, trainable))
            with apply_drain:
                _drain(trainable)
    report["telemetry"]["apply_gradients"] = sampler.stop()
    report["sections"]["apply_gradients_enqueue"] = apply_enqueue.summary()
    report["sections"]["apply_gradients_drain"] = apply_drain.summary()
    report["sections"]["apply_gradients_total"] = apply_total.summary()
    try:
        report["optimizer_iterations"] = {
            "before": iterations_before, "after": int(optimizer.iterations),
            "applies_measured": batches,
        }
    except Exception:  # noqa: BLE001
        report["optimizer_iterations"] = None

    # --- F4: reading optimizer.iterations (a host<-device read on every fit() step) ----------
    iteration_read = _Section("iterations_read")
    for _ in range(max(3, min(batches, 5))):
        with iteration_read:
            try:
                int(optimizer.iterations)
            except Exception:  # noqa: BLE001
                break
    report["sections"]["iterations_read"] = iteration_read.summary()

    # --- F5: attribution cross-check -- the SAME work, drained ONCE at the very end ----------
    single = _Section("full_step_single_drain")
    sampler = _TelemetrySampler().start()
    for _ in range(batches):
        with single:
            _outputs, _loss, grads_once = taped_gradients()
            optimizer.apply_gradients(zip(grads_once, trainable))
            _drain(trainable)
    report["telemetry"]["full_step_single_drain"] = sampler.stop()
    report["sections"]["full_step_single_drain"] = single.summary()

    # --- F6: the COMPILED path -- what model.fit() actually executes -------------------------
    compiled = _Section("train_on_batch_compiled")
    compiled_error = None
    try:
        model.train_on_batch(inputs, labels)  # builds the tf.function; not measured
        sampler = _TelemetrySampler().start()
        for _ in range(batches):
            with compiled:
                model.train_on_batch(inputs, labels)
        report["telemetry"]["train_on_batch_compiled"] = sampler.stop()
    except Exception as error:  # noqa: BLE001
        compiled_error = repr(error)
    report["sections"]["train_on_batch_compiled"] = compiled.summary()
    report["compiled_error"] = compiled_error

    model.set_weights(saved_weights)  # leave the model numerically where we found it
    report["tf_gpu_memory"] = tf_gpu_memory()
    report["classification"] = classify_optimizer_breakdown(report)
    return report


def classify_optimizer_breakdown(report):
    """Turns the measured sections into the specific verdicts this diagnostic exists to produce.
    Every branch cites the numbers it used; nothing is asserted that was not measured."""
    sections = report["sections"]
    verdicts = []

    def value(name, key="wall_seconds"):
        entry = sections.get(name)
        return entry[key] if entry else None

    enqueue = value("apply_gradients_enqueue")
    drain = value("apply_gradients_drain")
    drain_overhead = value("drain_variables_overhead") or 0.0
    if enqueue is not None and drain is not None:
        device = max(0.0, drain - drain_overhead)
        total = enqueue + device
        if total > 0:
            host_fraction = enqueue / total
            verdicts.append({
                "question": "Is the optimizer CPU-bound or GPU-bound?",
                "answer": ("CPU-BOUND" if host_fraction >= 0.6 else
                           "GPU-BOUND" if host_fraction <= 0.4 else "MIXED"),
                "evidence": ("apply_gradients returned to Python in %.1f ms; %.1f ms of device "
                             "work remained (drain %.1f ms minus %.1f ms drain overhead). Host "
                             "share %.0f%% of %.1f ms."
                             % (enqueue * 1e3, device * 1e3, drain * 1e3,
                                drain_overhead * 1e3, host_fraction * 100, total * 1e3)),
            })

    per_section = value("forward_and_gradients")
    apply_total = value("apply_gradients_total")
    single = value("full_step_single_drain")
    samples = value("full_step_single_drain", "samples") or 0
    if per_section is not None and apply_total is not None and single is not None:
        summed = per_section + apply_total
        delta = summed - single
        relative = abs(delta) / single if single else None
        thin = samples < 5
        verdicts.append({
            "question": "Are the per-section boundaries real, or is async work misattributed?",
            "answer": ("TOO FEW SAMPLES TO JUDGE (%d)" % samples if thin else
                       "SOUND" if relative is not None and relative <= 0.15 else
                       "SUSPECT -- the decomposition does not add up to the full step"),
            "evidence": ("sum of drained sections %.0f ms vs one-drain full step %.0f ms "
                         "(difference %+.0f ms, %.0f%%), over %d post-warmup sample(s). A gap "
                         "here means work is landing outside the section it is attributed to, or "
                         "that the sections interact (allocator churn, memory pressure) -- "
                         "re-run with more batches before drawing a conclusion."
                         % (summed * 1e3, single * 1e3, delta * 1e3,
                            (relative * 100) if relative is not None else float("nan"), samples)),
        })

    eager = single
    compiled = value("train_on_batch_compiled")
    if eager is not None and compiled is not None and compiled > 0:
        verdicts.append({
            "question": "How much of the step is eager dispatch rather than real computation?",
            "answer": ("COMPILED IS %.2fx FASTER" % (eager / compiled) if compiled < eager
                       else "COMPILED IS NOT FASTER (%.2fx)" % (eager / compiled)),
            "evidence": ("eager taped step %.0f ms vs compiled train_on_batch %.0f ms. "
                         "model.fit() uses the compiled path, so that is the one that bounds a "
                         "real epoch. The comparison is conservative: train_on_batch ALSO "
                         "computes the compiled QWK metric, which the eager loop does not."
                         % (eager * 1e3, compiled * 1e3)),
        })

    stack = report.get("stack") or {}
    optimizer = stack.get("optimizer") or {}
    if optimizer:
        clipping = [name for name in ("clipnorm", "clipvalue", "global_clipnorm")
                    if optimizer.get(name) is not None]
        verdicts.append({
            "question": "Is there gradient processing/clipping time to account for?",
            "answer": ("YES: " + ", ".join(clipping)) if clipping else "NO -- none configured",
            "evidence": ("clipnorm=%r clipvalue=%r global_clipnorm=%r use_ema=%r"
                         % (optimizer.get("clipnorm"), optimizer.get("clipvalue"),
                            optimizer.get("global_clipnorm"), optimizer.get("use_ema"))),
        })
    policy = stack.get("global_policy")
    model_policy = stack.get("dtype_policy")
    if policy is not None:
        verdicts.append({
            "question": "Is mixed precision actually in effect for this model?",
            "answer": ("YES" if model_policy and "float16" in str(model_policy) else
                       "NO -- the model is %s" % model_policy),
            "evidence": ("global policy %r, model dtype policy %r, optimizer %r "
                         "(loss_scale_factor=%r). Keras captures the dtype policy when a layer is "
                         "BUILT, so a policy set after the model was constructed does not apply "
                         "to it." % (policy, model_policy, optimizer.get("type"),
                                     optimizer.get("loss_scale_factor"))),
        })
    return verdicts


def print_optimizer_breakdown(report):
    print("=" * 78)
    print("PHASE F -- STEP BREAKDOWN WITH EXPLICIT SYNCHRONIZATION")
    print("=" * 78)
    stack = report.get("stack") or {}
    trainable = stack.get("trainable") or {}
    gradients = stack.get("gradients") or {}
    optimizer = stack.get("optimizer") or {}
    print("  trainable tensors    : %s (%s parameters, %s)"
          % (trainable.get("count"), format(trainable.get("elements") or 0, ","),
             _gib(trainable.get("bytes"))))
    print("    by dtype           : %s" % trainable.get("by_dtype"))
    print("  gradient tensors     : %s (%s elements, %s); None gradients: %s"
          % (gradients.get("count"), format(gradients.get("elements") or 0, ","),
             _gib(gradients.get("bytes")), stack.get("none_gradients")))
    print("    by dtype           : %s" % gradients.get("by_dtype"))
    print("  optimizer            : %s (inner=%s, loss_scale_factor=%s)"
          % (optimizer.get("type"), optimizer.get("inner_optimizer"),
             optimizer.get("loss_scale_factor")))
    print("    slot variables     : %s (%s), by dtype %s"
          % (optimizer.get("slot_variables"), _gib(optimizer.get("slot_bytes")),
             optimizer.get("slot_variables_by_dtype")))
    print("    clipping           : clipnorm=%s clipvalue=%s global_clipnorm=%s use_ema=%s"
          % (optimizer.get("clipnorm"), optimizer.get("clipvalue"),
             optimizer.get("global_clipnorm"), optimizer.get("use_ema")))
    print("  dtype policy         : model=%s global=%s"
          % (stack.get("dtype_policy"), stack.get("global_policy")))
    print("  jit/XLA              : model.jit_compile=%s global_jit=%s XLA_FLAGS=%s"
          % (stack.get("jit", {}).get("model_jit_compile"),
             stack.get("jit", {}).get("global_jit"), stack.get("jit", {}).get("XLA_FLAGS")))
    print("  run_eagerly=%s  steps_per_execution=%s  warmup=%.2fs"
          % (stack.get("run_eagerly"), stack.get("steps_per_execution"),
             report.get("warmup_seconds") or 0.0))
    if report.get("optimizer_iterations"):
        print("  optimizer.iterations : %s -> %s over %s measured applies"
              % (report["optimizer_iterations"]["before"],
                 report["optimizer_iterations"]["after"],
                 report["optimizer_iterations"]["applies_measured"]))
    print("")
    print("  %-30s %10s %10s %8s %10s %10s %4s"
          % ("section", "wall ms", "cpu ms", "cpu/wall", "min ms", "max ms", "n"))
    print("  " + "-" * 88)
    for name in ("forward", "forward_and_gradients", "apply_gradients_enqueue",
                 "apply_gradients_drain", "apply_gradients_total", "full_step_single_drain",
                 "train_on_batch_compiled", "drain_gradients_overhead",
                 "drain_variables_overhead", "iterations_read"):
        entry = report["sections"].get(name)
        if not entry:
            continue
        print("  %-30s %10.1f %10.1f %8.2f %10.1f %10.1f %4d"
              % (name, entry["wall_seconds"] * 1e3, entry["cpu_seconds"] * 1e3,
                 entry["cpu_over_wall"] if entry["cpu_over_wall"] is not None else float("nan"),
                 entry["min_wall_seconds"] * 1e3, entry["max_wall_seconds"] * 1e3,
                 entry["samples"]))
    print("  cpu/wall near 1.0 = the host is busy; near 0.0 = the host is waiting on the device.")
    print("")
    print("  GPU utilization sampled DURING each section:")
    for name, telemetry in (report.get("telemetry") or {}).items():
        if not telemetry:
            continue
        gpu = telemetry.get("gpu_util_percent") or {}
        cpu = telemetry.get("cpu_percent") or {}
        memory = telemetry.get("gpu_mem_used_mb") or {}
        print("    %-28s gpu mean %5.1f%% max %5.1f%%   cpu mean %5.1f%%   "
              "gpu mem %6.0f MiB   samples %d"
              % (name, gpu.get("mean") or 0.0, gpu.get("max") or 0.0, cpu.get("mean") or 0.0,
                 memory.get("mean") or 0.0, gpu.get("samples") or 0))
    if report.get("compiled_error"):
        print("")
        print("  compiled path NOT measured: %s" % report["compiled_error"])
    print("")
    print("  VERDICTS (each cites the measurement it rests on):")
    for verdict in report.get("classification") or []:
        print("    Q: %s" % verdict["question"])
        print("       -> %s" % verdict["answer"])
        print("          %s" % verdict["evidence"])
    print("")
    print("  This advanced the optimizer's slot state. Re-run the model construction cell before")
    print("  any real training run.")


# =====================================================================
# 5. Per-artifact I/O breakdown from a recorder
# =====================================================================

def io_breakdown(recorder, batches, batch_size):
    """Per-artifact I/O, normalized by the batches CONSUMED.

    Caveat reported alongside the numbers rather than hidden: `prefetch(AUTOTUNE)` lets the
    generator run ahead of the consumer, so the recorded operation counts cover every sample the
    generator PRODUCED, which is generally more than `batches * batch_size`. `reads_per_sample`
    below makes the discrepancy visible -- it should be ~4 (vessel, lesion, rgb, reliability) if
    the counts and the batch total refer to the same samples, and higher when prefetch ran ahead.
    The per-batch time figures are therefore an upper bound on what the consumed batches cost."""
    samples = max(1, batches * batch_size)
    breakdown = {}
    for artifact in jcd.ARTIFACTS:
        reads = recorder.matching(op="read", artifact=artifact)
        breakdown[artifact] = {
            "reads": len(reads),
            "read_seconds": sum(op.seconds for op in reads),
            "bytes": sum(op.nbytes for op in reads),
            "drive_reads": recorder.count(op="read", artifact=artifact, is_persistent=True),
            "local_reads": recorder.count(op="read", artifact=artifact, is_persistent=False),
            "ms_per_sample": (sum(op.seconds for op in reads) / samples) * 1000.0,
        }
    all_reads = recorder.matching(op="read")
    return {
        "per_artifact": breakdown,
        "totals": {
            "local_stats": recorder.count(op="stat", is_persistent=False),
            "drive_stats": recorder.count(op="stat", is_persistent=True),
            "local_reads": recorder.count(op="read", is_persistent=False),
            "drive_reads": recorder.count(op="read", is_persistent=True),
            "local_writes": recorder.count(op="write", is_persistent=False),
            "drive_writes": recorder.count(op="write", is_persistent=True),
            "cache_read_seconds": sum(op.seconds for op in all_reads),
            "bytes_read": sum(op.nbytes for op in all_reads),
            "bytes_per_batch": sum(op.nbytes for op in all_reads) / max(1, batches),
            "reads_per_sample": recorder.count(op="read") / float(samples),
            "raw_image_loads": recorder.call_count("lfed._load_raw_bgr"),
            "raw_image_seconds": recorder.call_seconds("lfed._load_raw_bgr"),
            "stage02_calls": recorder.call_count("lfed._resolve_processed_rgb"),
            "rgb_recomputations": recorder.call_count("_resize_rgb_01"),
            "vessel_recomputations": recorder.call_count("predict_vessel_mask"),
            "lesion_recomputations": recorder.call_count("racaf.tta_views"),
        },
        "drive_paths_touched": sorted({op.path for op in recorder.matching(is_persistent=True)}),
        "drive_write_paths": [op.path for op in recorder.matching(op="write", is_persistent=True)],
    }


def measure_augmentation_cost(cache_dir, racaf_cache_dir, image_dir, entries, image_size,
                              vessel_model, stage4_model, repeats=10):
    """Augmentation and the Stage 06 resize, timed on their own against real cached samples --
    the part of per-sample cost that is pure CPU rather than I/O. Read-only."""
    import local_feature_extraction_dataset as lfed
    rng = np.random.default_rng(0)
    build_times, augment_times, resize_times = [], [], []
    for id_code, diagnosis in list(entries)[:repeats]:
        start = time.perf_counter()
        try:
            sample = jtd._build_joint_sample(
                id_code, diagnosis, image_dir, cache_dir, racaf_cache_dir,
                vessel_model, stage4_model, False, None, image_size=image_size,
            )
        except Exception:  # noqa: BLE001 -- e.g. an empty-FOV image; skip it
            continue
        build_times.append(time.perf_counter() - start)

        tensor = sample["stage5_input"]
        start = time.perf_counter()
        lfed._augment_spatial(np.array(tensor, copy=True), rng)
        augment_times.append(time.perf_counter() - start)

        start = time.perf_counter()
        lfed._resize_input(tensor[..., :3], jtd.STAGE6_IMAGE_SIZE)
        resize_times.append(time.perf_counter() - start)

    def mean(values):
        return (sum(values) / len(values)) if values else None
    return {
        "samples": len(build_times),
        "build_joint_sample_seconds": mean(build_times),
        "augment_spatial_seconds": mean(augment_times),
        "stage6_resize_seconds": mean(resize_times),
    }


# =====================================================================
# 6. Orchestration
# =====================================================================

def run_profile(model, train_ds, cache_dir, racaf_cache_dir, image_dir,
                persistent_cache_dir=None, persistent_racaf_cache_dir=None,
                batch_size=2, batches=DEFAULT_BATCHES, entries=None,
                vessel_model=None, stage4_model=None, image_size=None,
                drive_mount="/content/drive", extra_paths=None,
                preflight_sample_limit=None, abort_if_drive_fallback=True):
    """Runs the storage survey, the configuration snapshot and all three measurement phases, and
    returns a plain-dict report (`print_report` renders it).

    `train_ds` must be the REAL training dataset the run uses, and `model` the REAL compiled joint
    model; its weights are restored after every phase that takes gradient steps."""
    image_size = image_size if image_size is not None else jtd.STAGE5_IMAGE_SIZE
    paths = {
        "/content": "/content",
        "local_cache": cache_dir,
        "local_racaf_cache": racaf_cache_dir,
        "staged_datasets": "/content/datasets",
        "staged_images": image_dir,
        "repo": "/content/diabetic_retinoplasty",
        "drive_mount": drive_mount,
        "persistent_cache": persistent_cache_dir,
        "persistent_racaf_cache": persistent_racaf_cache_dir,
        "tmp": "/tmp",
        "root": "/",
    }
    if extra_paths:
        paths.update(extra_paths)

    report = {"batch_size": batch_size, "requested_batches": batches}
    print("[1/6] storage survey ...")
    report["storage"] = survey_storage(paths)
    print("[2/6] configuration snapshot ...")
    report["config"] = snapshot_configuration(model)

    # Establish whether the local cache is complete for the entries about to be iterated, BEFORE
    # iterating them. A local miss is what sends the pipeline to Drive, and doing that thousands
    # of times in a row is what took the mount down on the first attempt.
    if entries is not None:
        print("[3/6] preflight per-image cache audit ...")
        report["preflight"] = preflight_cache_audit(
            entries, cache_dir, racaf_cache_dir, image_size,
            persistent_cache_dir=persistent_cache_dir,
            persistent_racaf_cache_dir=persistent_racaf_cache_dir,
            sample_limit=preflight_sample_limit,
        )
        audit = report["preflight"]
        print("      %d/%d entries fully local; %d would fall back to Drive; %d missing everywhere"
              % (audit["fully_local"], audit["entries_checked"], audit["would_read_drive"],
                 audit["missing_everywhere"]))
        if audit["drive_unreachable"]:
            print("      Drive is UNREACHABLE right now -- see the report below.")
        if audit["would_read_drive"] and abort_if_drive_fallback:
            report["classification"] = classify(report)
            report["aborted"] = (
                "Aborted before Phase C: %d of %d entries have a missing LOCAL artifact and would "
                "read Drive during measurement. That is what took the mount down last time, and it "
                "would also make the timings measure Drive latency rather than the training step. "
                "Complete the local cache first (Phase 1 mirrors persistent entries locally), or "
                "re-run with abort_if_drive_fallback=False to measure anyway."
                % (audit["would_read_drive"], audit["entries_checked"]))
            print("")
            print(report["aborted"])
            return report

    roots = _roots_for(cache_dir, racaf_cache_dir, persistent_cache_dir,
                       persistent_racaf_cache_dir, image_dir, drive_mount)

    print("[4/6] PHASE C -- input pipeline only, no model ...")
    input_only = profile_input_only(train_ds, roots, batches=batches)
    report["input_only"] = {k: v for k, v in input_only.items() if k != "recorder"}
    report["input_io"] = io_breakdown(input_only["recorder"], input_only["batches"], batch_size)
    if input_only.get("failure"):
        # Phase C could not complete. Running D and E now would only produce numbers that cannot
        # be compared against a missing Phase C baseline, so stop and report the classified cause.
        report["classification"] = classify(report)
        report["aborted"] = ("Phase C failed: %s"
                             % input_only["failure"].get("meaning", "see the failure block"))
        print("")
        print("PHASE C FAILED -- %s" % input_only["failure"].get("category"))
        print(input_only["failure"].get("message", ""))
        return report

    print("[5/6] PHASE D -- model compute only, one reused batch ...")
    warm_batch = None
    for batch in train_ds.take(1):
        warm_batch = batch
    if warm_batch is None:
        report["compute_only"] = {"error": "could not materialize a batch from train_ds"}
    else:
        report["compute_only"] = profile_compute_only(model, warm_batch, batches=batches)

    print("[6/6] PHASE E -- combined dataset + train step ...")
    combined = profile_combined(model, train_ds, roots, batches=batches)
    report["combined"] = {k: v for k, v in combined.items() if k != "recorder"}
    report["combined_io"] = io_breakdown(combined["recorder"], combined["batches"], batch_size)

    if entries is not None and vessel_model is not None and stage4_model is not None:
        print("      per-sample CPU breakdown (augmentation / resize) ...")
        report["cpu_breakdown"] = measure_augmentation_cost(
            cache_dir, racaf_cache_dir, image_dir, entries, image_size,
            vessel_model, stage4_model,
        )
    report["classification"] = classify(report)
    return report


def classify(report):
    """A/B/C/D/E/F/G/H, from the measurements only. Every verdict carries its numbers."""
    verdicts, evidence = [], []
    combined = report.get("combined") or {}
    compute = report.get("compute_only") or {}
    input_only = report.get("input_only") or {}
    totals = (report.get("combined_io") or {}).get("totals", {})

    batch = combined.get("batch_seconds")
    wait = combined.get("input_wait_seconds")
    step = combined.get("compute_seconds")
    input_ceiling = input_only.get("steady_mean_seconds")
    compute_ceiling = compute.get("per_step_seconds")

    if batch and wait is not None and step is not None:
        evidence.append(
            "combined batch %.0f ms = input wait %.0f ms (%.0f%%) + train step %.0f ms (%.0f%%)"
            % (batch * 1000, wait * 1000, 100 * wait / batch, step * 1000, 100 * step / batch))
    if input_ceiling:
        evidence.append("input-only ceiling %.0f ms/batch" % (input_ceiling * 1000))
    if compute_ceiling:
        evidence.append("compute-only ceiling %.0f ms/step" % (compute_ceiling * 1000))

    gpu = (combined.get("telemetry") or {}).get("gpu_util_percent")
    if gpu:
        evidence.append("GPU utilization during the combined loop: mean %.0f%%, max %.0f%% over "
                        "%d samples" % (gpu["mean"], gpu["max"], gpu["samples"]))
    cpu = (combined.get("telemetry") or {}).get("cpu_percent")
    if cpu:
        evidence.append("CPU utilization: mean %.0f%%, max %.0f%%" % (cpu["mean"], cpu["max"]))

    if totals.get("drive_reads") or totals.get("drive_stats") or totals.get("drive_writes"):
        verdicts.append(("C. DRIVE I/O",
                         "the training loop touched Drive: %d stat(s), %d read(s), %d write(s)"
                         % (totals.get("drive_stats", 0), totals.get("drive_reads", 0),
                            totals.get("drive_writes", 0))))
    else:
        evidence.append("zero Drive stats, reads and writes during the combined training loop")

    if batch and wait is not None and step is not None:
        if wait > step:
            verdicts.append(("B. INPUT PIPELINE",
                             "input wait (%.0f ms) exceeds train step (%.0f ms): the GPU is "
                             "waiting for data" % (wait * 1000, step * 1000)))
        elif step > wait * 2:
            verdicts.append(("A. GPU COMPUTE",
                             "train step (%.0f ms) dominates input wait (%.0f ms): the pipeline "
                             "keeps up" % (step * 1000, wait * 1000)))

    cache_seconds = totals.get("cache_read_seconds")
    if cache_seconds is not None and combined.get("total_seconds"):
        share = 100.0 * cache_seconds / max(1e-9, combined["total_seconds"])
        evidence.append("cache file reads took %.2fs = %.1f%% of the combined loop's wall clock; "
                        "%.1f MB read per batch"
                        % (cache_seconds, share, totals.get("bytes_per_batch", 0) / 1e6))
        if share >= 25.0:
            verdicts.append(("D. LOCAL SSD I/O",
                             "cache reads alone are %.0f%% of training wall clock" % share))

    cpu_break = report.get("cpu_breakdown") or {}
    if cpu_break.get("build_joint_sample_seconds"):
        evidence.append("_build_joint_sample %.0f ms/sample -> %.0f ms/batch at batch_size=%d"
                        % (cpu_break["build_joint_sample_seconds"] * 1000,
                           cpu_break["build_joint_sample_seconds"] * 1000 * report["batch_size"],
                           report["batch_size"]))

    content = (report.get("storage") or {}).get("paths", {}).get("/content", {})
    vfs = content.get("statvfs")
    if vfs and vfs["total_bytes"]:
        free_gib = vfs["free_bytes"] / float(1024 ** 3)
        pct_used = 100.0 * (1 - vfs["free_bytes"] / float(vfs["total_bytes"]))
        evidence.append("/content %.0f%% full, %.1f GiB free" % (pct_used, free_gib))
        if free_gib < 5.0:
            verdicts.append(("F. STORAGE PRESSURE", "only %.1f GiB free on /content" % free_gib))
        if vfs.get("inodes_total") and vfs.get("inodes_free") is not None:
            inode_pct = 100.0 * (1 - vfs["inodes_free"] / float(vfs["inodes_total"]))
            evidence.append("inode usage %.1f%%" % inode_pct)
            if inode_pct > 90.0:
                verdicts.append(("F. STORAGE PRESSURE", "inode usage %.0f%%" % inode_pct))

    if batch and input_ceiling and compute_ceiling:
        ceiling = max(input_ceiling, compute_ceiling)
        if batch > ceiling * 1.35:
            verdicts.append(("H. MIXED / CONTENTION",
                             "combined %.0f ms/batch is %.2fx the slower of the two isolated "
                             "ceilings (%.0f ms) -- the phases contend rather than overlap"
                             % (batch * 1000, batch / ceiling, ceiling * 1000)))
    return {"verdicts": verdicts, "evidence": evidence}


# =====================================================================
# 7. Rendering
# =====================================================================

def _ms(seconds):
    return "n/a" if seconds is None else "%.1f ms" % (seconds * 1000.0)


def _gib(num_bytes):
    return "n/a" if num_bytes is None else "%.2f GiB" % (num_bytes / float(1024 ** 3))


def print_report(report):
    line = "=" * 78
    print(line)
    print("JOINT TRAINING STEP PROFILE -- MEASUREMENT ONLY (no Trainer, no callbacks, no fit)")
    print(line)
    print("batch_size=%d, batches per phase=%d" % (report["batch_size"], report["requested_batches"]))

    # --- A. environment / storage -------------------------------------
    print("")
    print("A. ENVIRONMENT AND STORAGE")
    print("-" * 78)
    print("%-22s %10s %10s %10s  %-12s %s" % ("path", "size", "free", "files", "fstype", "location"))
    for label, entry in report["storage"]["paths"].items():
        if not entry["exists"]:
            print("%-22s %10s" % (label, "ABSENT"))
            continue
        filesystem = entry["filesystem"]
        vfs = entry["statvfs"]
        print("%-22s %10s %10s %10s  %-12s %s" % (
            label,
            _gib(entry["size_bytes"]) if entry["size_bytes"] is not None else "-",
            _gib(vfs["free_bytes"]) if vfs else "-",
            entry["file_count"] if entry["file_count"] is not None else "-",
            filesystem["fstype"] if filesystem else "?",
            entry["path"]))
    if report["storage"].get("df"):
        print("")
        print("df -h:")
        print(report["storage"]["df"])
    if report["storage"].get("largest_dirs"):
        print("")
        print("largest directories under /content (Drive excluded):")
        print(report["storage"]["largest_dirs"])

    # --- B. configuration ---------------------------------------------
    print("")
    print("B. RUNTIME CONFIGURATION (read from the live process)")
    print("-" * 78)
    config = report["config"]
    for key in ("tensorflow_version", "keras_version", "gpu_devices", "mixed_precision_policy",
                "eager_execution", "intra_op_threads", "inter_op_threads", "cpu_count",
                "memory_growth", "shuffle_buffer_size", "nvidia_smi"):
        print("  %-24s %s" % (key + ":", config.get(key)))
    if config.get("gpu_details"):
        print("  %-24s %s" % ("gpu_details:", config["gpu_details"]))
    if config.get("tf_gpu_memory"):
        print("  %-24s current %.0f MB, peak %.0f MB" % (
            "tf_gpu_memory:", config["tf_gpu_memory"]["current_mb"],
            config["tf_gpu_memory"]["peak_mb"]))
    print("  env: %s" % {k: v for k, v in config.get("env", {}).items() if v is not None})
    if config.get("model"):
        print("  model:")
        for key, value in config["model"].items():
            print("      %-20s %s" % (key + ":", value))

    # --- preflight -------------------------------------------------------
    audit = report.get("preflight")
    if audit:
        print("")
        print("B2. PREFLIGHT PER-IMAGE CACHE AUDIT (profiler's own probes, not pipeline ops)")
        print("-" * 78)
        print("  entries checked            : %d" % audit["entries_checked"])
        print("  fully local (no Drive need): %d" % audit["fully_local"])
        print("  would fall back to Drive   : %d   <-- each of these is a Drive read during training"
              % audit["would_read_drive"])
        print("  missing local AND persistent: %d" % audit["missing_everywhere"])
        print("  probes performed           : %d (excluded from every pipeline counter)"
              % audit["probe_count"])
        if audit["drive_unreachable"]:
            print("  DRIVE UNREACHABLE: %s" % audit["drive_error"])
        missing_local = {k: v for k, v in audit["per_artifact_local_missing"].items() if v}
        if missing_local:
            print("  local misses by artifact   : %s" % missing_local)
        missing_persistent = {k: v for k, v in audit["per_artifact_persistent_missing"].items() if v}
        if missing_persistent:
            print("  persistent misses by artifact: %s" % missing_persistent)
        for id_code, artifacts in audit["examples_needing_drive"][:5]:
            print("      would read Drive: %s -> %s" % (id_code, artifacts))
        for id_code, artifacts in audit["examples_missing_everywhere"][:5]:
            print("      missing everywhere: %s -> %s" % (id_code, artifacts))
        if audit["would_read_drive"] == 0 and not audit["drive_unreachable"]:
            print("  VERDICT: the local cache is complete for these entries. Training will not")
            print("           read Drive, so a Drive outage cannot affect these measurements.")

    if report.get("aborted"):
        print("")
        print("=" * 78)
        print("RUN STOPPED EARLY")
        print("=" * 78)
        print(report["aborted"])
        for phase_name in ("input_only", "combined"):
            failure = (report.get(phase_name) or {}).get("failure")
            if failure:
                print("")
                print("  %s failure category: %s" % (phase_name, failure.get("category")))
                for key in ("artifact", "id_code", "path", "errno", "local_path", "local_exists",
                            "persistent", "detail"):
                    if failure.get(key) is not None:
                        print("      %-14s %s" % (key + ":", failure[key]))
                print("      meaning: %s" % failure.get("meaning"))
                print("      raw: %s" % str(failure.get("message"))[:600])
        print("")
        print("Nothing was recomputed and no cache file was modified.")
        return

    # --- C. the metric table -------------------------------------------
    combined = report.get("combined") or {}
    compute = report.get("compute_only") or {}
    input_only = report.get("input_only") or {}
    totals = (report.get("combined_io") or {}).get("totals", {})
    per_artifact = (report.get("combined_io") or {}).get("per_artifact", {})
    cpu_break = report.get("cpu_breakdown") or {}
    batch_size = report["batch_size"]

    def artifact_ms(name):
        entry = per_artifact.get(name)
        return None if not entry else entry["read_seconds"] / max(1, combined.get("batches", 1))

    telemetry = combined.get("telemetry") or {}
    gpu_util = telemetry.get("gpu_util_percent")
    gpu_mem = telemetry.get("gpu_mem_used_mb")
    cpu_util = telemetry.get("cpu_percent")
    content_vfs = (report["storage"]["paths"].get("/content") or {}).get("statvfs")

    print("")
    print("C. PER-BATCH PROFILE (combined loop -- the real training shape)")
    print("-" * 78)
    print("%-30s %s" % ("METRIC", "RESULT"))
    print("-" * 78)
    rows = [
        ("batch wall time", _ms(combined.get("batch_seconds"))),
        ("input wait", _ms(combined.get("input_wait_seconds"))),
        ("train step (fwd+bwd+opt)", _ms(combined.get("compute_seconds"))),
        ("_build_joint_sample (per sample)", _ms(cpu_break.get("build_joint_sample_seconds"))),
        ("cache I/O (per batch)", _ms(totals.get("cache_read_seconds", 0) / max(1, combined.get("batches", 1)))),
        ("  vessel I/O", _ms(artifact_ms("vessel"))),
        ("  lesion I/O", _ms(artifact_ms("lesion"))),
        ("  RGB I/O", _ms(artifact_ms("rgb"))),
        ("  RACAF reliability I/O", _ms(artifact_ms("reliability"))),
        ("augmentation (per sample)", _ms(cpu_break.get("augment_spatial_seconds"))),
        ("stage6 resize (per sample)", _ms(cpu_break.get("stage6_resize_seconds"))),
        ("forward pass", _ms(compute.get("forward_seconds"))),
        ("backward pass", _ms(compute.get("backward_seconds"))),
        ("optimizer step", _ms(compute.get("optimizer_seconds"))),
        ("", ""),
        ("input-only ceiling /batch", _ms(input_only.get("steady_mean_seconds"))),
        ("compute-only ceiling /step", _ms(compute.get("per_step_seconds"))),
        ("first batch (one-time cost)", _ms(combined.get("first_batch_seconds"))),
        ("graph warm-up (one-time)", _ms(compute.get("warmup_seconds"))),
        ("", ""),
        ("Drive reads", totals.get("drive_reads")),
        ("Drive stats", totals.get("drive_stats")),
        ("Drive writes", totals.get("drive_writes")),
        ("local reads", totals.get("local_reads")),
        ("local writes", totals.get("local_writes")),
        ("bytes read per batch", "%.1f MB" % (totals.get("bytes_per_batch", 0) / 1e6)),
        ("reads per sample", "%.1f  (4.0 = exactly one read of each artifact; higher means "
                             "prefetch ran ahead of the consumed batches)"
                             % totals.get("reads_per_sample", 0)),
        ("raw image loads", totals.get("raw_image_loads")),
        ("cache recomputations (rgb)", totals.get("rgb_recomputations")),
        ("cache recomputations (vessel)", totals.get("vessel_recomputations")),
        ("cache recomputations (lesion)", totals.get("lesion_recomputations")),
        ("", ""),
        ("GPU utilization", "mean %.0f%%, max %.0f%%" % (gpu_util["mean"], gpu_util["max"]) if gpu_util else "n/a"),
        ("GPU memory used", "%.2f GiB / %.2f GiB" % (gpu_mem["mean"] / 1024.0, (telemetry.get("gpu_mem_total_mb") or 0) / 1024.0) if gpu_mem else "n/a"),
        ("TF GPU memory (cur/peak)", "%.0f / %.0f MB" % (combined["tf_gpu_memory"]["current_mb"], combined["tf_gpu_memory"]["peak_mb"]) if combined.get("tf_gpu_memory") else "n/a"),
        ("CPU utilization", "mean %.0f%%, max %.0f%%" % (cpu_util["mean"], cpu_util["max"]) if cpu_util else "n/a"),
        ("RAM used", "%.1f GiB" % telemetry["ram_used_gb"]["max"] if telemetry.get("ram_used_gb") else "n/a"),
        ("disk free (/content)", _gib(content_vfs["free_bytes"]) if content_vfs else "n/a"),
    ]
    for label, value in rows:
        if label == "":
            print("")
        else:
            print("%-30s %s" % (label, value))

    # --- extrapolation --------------------------------------------------
    if combined.get("batch_seconds"):
        print("")
        print("Extrapolated from the measured steady-state batch time:")
        print("  %.2f s/step x 1461 steps = %.0f s/epoch (%.2f h)" % (
            combined["batch_seconds"], combined["batch_seconds"] * 1461,
            combined["batch_seconds"] * 1461 / 3600.0))
        print("  (train steps only -- excludes validation, checkpointing and TensorBoard, none of")
        print("   which this diagnostic runs)")

    # --- D. Drive evidence -----------------------------------------------
    print("")
    print("D. DRIVE ACCESS EVIDENCE")
    print("-" * 78)
    drive_paths = (report.get("combined_io") or {}).get("drive_paths_touched") or []
    if not drive_paths:
        print("  No path under any persistent root was stat'ed, read or written during the")
        print("  combined training loop. Training is reading local SSD only.")
    else:
        print("  %d distinct Drive path(s) touched during training:" % len(drive_paths))
        for path in drive_paths[:25]:
            print("    %s" % path)
    writes = (report.get("combined_io") or {}).get("drive_write_paths") or []
    print("  Drive WRITE tripwire: %s" % ("PASS -- none" if not writes else "FAILED: %s" % writes))

    # --- E. classification -----------------------------------------------
    print("")
    print("E. CLASSIFICATION (from measurements only)")
    print("-" * 78)
    classification = report.get("classification") or {}
    print("Evidence:")
    for item in classification.get("evidence", []):
        print("  - %s" % item)
    print("")
    print("Verdict(s):")
    if not classification.get("verdicts"):
        print("  none triggered -- no single component crossed its threshold; read the table above")
    for name, why in classification["verdicts"]:
        print("  %s" % name)
        print("      %s" % why)
    print("")
    print("This reports what was measured. It does not name a root cause on its own.")
    print("")
    print("NOTE: this diagnostic took real gradient steps. Model weights were snapshotted and")
    print("restored, but the optimizer's slot/iteration state was advanced -- rebuild and")
    print("recompile the model (re-run the model construction cell) before the real training run.")


# =====================================================================
# PHASE G -- what does model.fit() ACTUALLY cost?
#
# Phases C-F all measure eager, hand-written steps. `model.fit()` does not run
# an eager step: it runs a compiled `train_function` (`jit_compile`, XLA where
# available, `steps_per_execution` batches per call). Phase F measured a
# compiled `train_on_batch` at ~78 ms against a ~4215 ms eager taped step on the
# same batch and the same GPU -- a 54x gap -- which means the eager number
# cannot be read as "the training step costs 4.2 s", and no root cause may be
# named from it.
#
# But the historical real run displayed ~4-5 s per step, and that WAS
# `model.fit()`. Something outside the compiled step accounts for the
# difference. This phase measures the four paths that between them isolate it:
#
#   A. dataset only        -- iterate the real tf.data pipeline, no model
#   B. compiled step only  -- train_on_batch on ONE materialized batch, reused
#   C. model.fit(), minimal callbacks
#   D. model.fit(), the REAL callback stack, redirected to a LOCAL directory
#
#   D - C   = callback overhead (TensorBoard histograms, CSV, checkpoint writes)
#   C - B   = what fit() adds around the compiled step, which for a prefetching
#             pipeline is dominated by input starvation whenever A > B
#   A       = the ceiling the input pipeline can sustain, measured against a
#             consumer that is NOT artificially slow
#
# Safety: weights are snapshotted and restored; nothing is written outside the
# caller-supplied local directory; the model is flagged
# `_dr_diagnostic_dirty` afterwards so `training.Trainer.fit()` refuses it until
# it has been rebuilt and recompiled.
# =====================================================================

DEFAULT_FIT_STEPS = 25


class _StepTimer(tf.keras.callbacks.Callback):
    """Per-batch wall time as `model.fit()` itself sees it. `on_train_batch_end`
    fires after the compiled `train_function` returns, which under TensorFlow's
    asynchronous execution means the batch has been ENQUEUED, not necessarily
    finished -- so these are the same numbers Keras' own progress bar reports,
    which is exactly the quantity under investigation."""

    def __init__(self):
        super().__init__()
        self.batch_seconds = []
        self.epoch_seconds = None
        self._batch_start = None
        self._epoch_start = None

    def on_epoch_begin(self, epoch, logs=None):
        self._epoch_start = time.perf_counter()

    def on_train_batch_begin(self, batch, logs=None):
        self._batch_start = time.perf_counter()

    def on_train_batch_end(self, batch, logs=None):
        if self._batch_start is not None:
            self.batch_seconds.append(time.perf_counter() - self._batch_start)

    def on_epoch_end(self, epoch, logs=None):
        if self._epoch_start is not None:
            self.epoch_seconds = time.perf_counter() - self._epoch_start

    def summary(self):
        times = self.batch_seconds
        steady = times[1:] if len(times) > 1 else []
        return {
            "steps": len(times),
            "first_step_seconds": times[0] if times else None,
            "steady_mean_seconds": (sum(steady) / len(steady)) if steady else None,
            "steady_max_seconds": max(steady) if steady else None,
            "steady_min_seconds": min(steady) if steady else None,
            "epoch_seconds": self.epoch_seconds,
        }


def _mean(values):
    return (sum(values) / len(values)) if values else None


def _fit_configuration(model):
    steps_per_execution = getattr(model, "steps_per_execution", None)
    if steps_per_execution is not None:
        try:
            steps_per_execution = int(tf.keras.backend.get_value(steps_per_execution))
        except (TypeError, ValueError):
            steps_per_execution = None
    optimizer = getattr(model, "optimizer", None)
    inner = getattr(optimizer, "inner_optimizer", None)
    return {
        "jit_compile": getattr(model, "jit_compile", None),
        "run_eagerly": getattr(model, "run_eagerly", None),
        "steps_per_execution": steps_per_execution,
        "optimizer": type(optimizer).__name__ if optimizer is not None else None,
        "inner_optimizer": type(inner).__name__ if inner is not None else None,
        "loss_scaled": inner is not None,
        "model_dtype_policy": getattr(getattr(model, "dtype_policy", None), "name", None),
        "global_dtype_policy": tf.keras.mixed_precision.global_policy().name,
        "trainable_tensors": len(model.trainable_variables),
        "trainable_parameters": int(sum(int(np.prod(v.shape)) for v in model.trainable_variables)),
    }


def profile_dataset_only(dataset, roots, steps=DEFAULT_FIT_STEPS):
    """PATH A -- pull `steps` batches from the real pipeline with no model
    attached, recording every filesystem operation. This is the number Phase E's
    1.3 ms "input wait" could not reveal: with `prefetch`, input wait only shows
    the pipeline is faster than the CONSUMER, and Phase E's consumer was a 4.1 s
    eager step."""
    recorder = jcd._Recorder(roots)
    telemetry = _TelemetrySampler().start()
    per_batch = []
    iterator = iter(dataset)
    start_all = time.perf_counter()
    with jcd._instrument(recorder):
        for _ in range(steps):
            start = time.perf_counter()
            try:
                next(iterator)
            except StopIteration:
                break
            per_batch.append(time.perf_counter() - start)
    total = time.perf_counter() - start_all
    steady = per_batch[1:] if len(per_batch) > 1 else []
    return {
        "steps": len(per_batch),
        "total_seconds": total,
        "first_batch_seconds": per_batch[0] if per_batch else None,
        "steady_mean_seconds": _mean(steady),
        "steady_max_seconds": max(steady) if steady else None,
        "per_batch_seconds": per_batch,
        "telemetry": telemetry.stop(),
        "recorder": recorder,
    }


def profile_compiled_step(model, batch, steps=DEFAULT_FIT_STEPS):
    """PATH B -- the COMPILED training step, via `train_on_batch`, on one
    already-materialized batch reused every iteration. Zero dataset work, and
    the same `train_function` `model.fit()` calls."""
    inputs, labels = batch
    # A multi-input model expects a LIST; `tf.data` hands the inputs over as a tuple.
    if isinstance(inputs, tuple):
        inputs = list(inputs)
    telemetry = _TelemetrySampler().start()

    warm_start = time.perf_counter()
    model.train_on_batch(inputs, labels)          # builds/compiles train_function
    warmup_seconds = time.perf_counter() - warm_start

    per_step = []
    start_all = time.perf_counter()
    for _ in range(steps):
        start = time.perf_counter()
        model.train_on_batch(inputs, labels)
        per_step.append(time.perf_counter() - start)
    total = time.perf_counter() - start_all
    steady = per_step[1:] if len(per_step) > 1 else []
    return {
        "steps": len(per_step),
        "warmup_seconds": warmup_seconds,
        "total_seconds": total,
        "first_step_seconds": per_step[0] if per_step else None,
        "steady_mean_seconds": _mean(steady),
        "per_step_seconds": per_step,
        "telemetry": telemetry.stop(),
        "tf_gpu_memory": tf_gpu_memory(),
    }


def profile_fit(model, dataset, roots, steps=DEFAULT_FIT_STEPS, callbacks=None, label="fit"):
    """PATH C / D -- one short `model.fit()` over the real pipeline.

    `steps_per_epoch=steps` and `epochs=1` keep it to the requested handful of
    batches. No validation data is passed: validation is a separate cost, and
    mixing it in would make the per-step figure unattributable."""
    recorder = jcd._Recorder(roots)
    timer = _StepTimer()
    telemetry = _TelemetrySampler().start()
    start = time.perf_counter()
    with jcd._instrument(recorder):
        model.fit(
            dataset,
            epochs=1,
            steps_per_epoch=steps,
            callbacks=list(callbacks or []) + [timer],
            verbose=0,
        )
    total = time.perf_counter() - start
    summary = timer.summary()
    summary.update({
        "label": label,
        "total_seconds": total,
        "per_step_seconds": timer.batch_seconds,
        "telemetry": telemetry.stop(),
        "recorder": recorder,
    })
    return summary


def profile_fit_paths(model, dataset, roots, steps=DEFAULT_FIT_STEPS,
                      diagnostic_dir=None, monitor="val_QWK", mode="max",
                      real_callbacks=True):
    """Run paths A-D and return one comparable report.

    Pass the SAME `tf.data.Dataset` the training cell built. Every path takes its
    own fresh iterator -- `iter(dataset)` restarts the generator, and
    `model.fit()` makes its own -- so no path inherits a partly consumed one.
    Deliberately NOT a factory: `load_joint_training_datasets()` loads the frozen
    Stage 03 (PyTorch) and Stage 04 (Keras) models on every call, so rebuilding
    the pipeline per path would load them four times over for no benefit.

    `diagnostic_dir` must be LOCAL (e.g. `/content/fit_diagnostic`). Nothing is
    written to Drive: that is the point of redirecting the real callback stack
    rather than pointing it at the experiment directory."""
    diagnostic_dir = diagnostic_dir or os.path.join(
        tempfile.gettempdir(), "joint_fit_diagnostic")
    shutil.rmtree(diagnostic_dir, ignore_errors=True)
    os.makedirs(diagnostic_dir, exist_ok=True)

    configuration = _fit_configuration(model)
    saved_weights = model.get_weights()
    report = {"configuration": configuration, "steps_requested": steps,
              "diagnostic_dir": diagnostic_dir}

    try:
        # --- A. dataset only ------------------------------------------------
        report["dataset_only"] = profile_dataset_only(dataset, roots, steps=steps)

        # --- B. compiled step only -----------------------------------------
        batch = next(iter(dataset))
        report["compiled_step"] = profile_compiled_step(model, batch, steps=steps)

        # --- C. fit() with minimal callbacks --------------------------------
        report["fit_minimal"] = profile_fit(
            model, dataset, roots, steps=steps, callbacks=[], label="fit_minimal")

        # --- D. fit() with the real callback stack, LOCAL output ------------
        if real_callbacks:
            from training.callbacks import build_callbacks
            from training.checkpointing import CheckpointOptions

            callbacks, callback_paths = build_callbacks(
                checkpoint_dir=os.path.join(diagnostic_dir, "checkpoints"),
                log_dir=os.path.join(diagnostic_dir, "logs"),
                monitor=monitor, mode=mode,
                checkpoint_options=CheckpointOptions(
                    experiment_id="fit-diagnostic",
                    staging_dir=os.path.join(diagnostic_dir, "_staging"),
                    verbose=0,
                ),
            )
            report["fit_real_callbacks"] = profile_fit(
                model, dataset, roots, steps=steps, callbacks=callbacks,
                label="fit_real_callbacks")
            report["callback_paths"] = callback_paths
    finally:
        model.set_weights(saved_weights)
        # The optimizer's slots and `iterations` were advanced by real gradient
        # steps; weights alone are not enough to make the model safe to train.
        setattr(model, "_dr_diagnostic_dirty", True)

    report["comparison"] = _compare_fit_paths(report)
    return report


def _compare_fit_paths(report):
    """Turn the four paths into the differences that actually answer the
    question. Every entry is arithmetic on measured numbers -- no attribution
    is asserted that the measurements do not support."""
    dataset = (report.get("dataset_only") or {}).get("steady_mean_seconds")
    compiled = (report.get("compiled_step") or {}).get("steady_mean_seconds")
    minimal = (report.get("fit_minimal") or {}).get("steady_mean_seconds")
    real = (report.get("fit_real_callbacks") or {}).get("steady_mean_seconds")

    comparison = {
        "dataset_only_seconds": dataset,
        "compiled_step_seconds": compiled,
        "fit_minimal_seconds": minimal,
        "fit_real_callbacks_seconds": real,
        "callback_overhead_seconds": (real - minimal) if (real and minimal) else None,
        "fit_overhead_over_compiled_seconds": (minimal - compiled) if (minimal and compiled) else None,
        "input_bound": (dataset is not None and compiled is not None and dataset > compiled),
        "historical_slow_step_reproduced": (minimal is not None and minimal >= 3.0),
    }

    evidence = []
    if dataset is not None and compiled is not None:
        ratio = dataset / compiled if compiled else None
        evidence.append(
            "dataset-only %.0f ms/batch vs compiled step %.0f ms/batch (%s)"
            % (dataset * 1000, compiled * 1000,
               ("input pipeline is %.1fx slower -- prefetch cannot hide it" % ratio)
               if ratio and ratio > 1 else "compute-bound"))
    if minimal is not None and compiled is not None:
        evidence.append(
            "model.fit() %.0f ms/step vs compiled step %.0f ms/step -- fit() adds %.0f ms/step "
            "outside the compiled train_function" % (minimal * 1000, compiled * 1000,
                                                     (minimal - compiled) * 1000))
    if real is not None and minimal is not None:
        evidence.append(
            "the real callback stack adds %.0f ms/step over minimal callbacks"
            % ((real - minimal) * 1000))
    if minimal is not None:
        evidence.append(
            "historical ~4-5 s/step %s in this run (model.fit() steady state %.2f s/step)"
            % ("REPRODUCED" if minimal >= 3.0 else "NOT reproduced", minimal))
    comparison["evidence"] = evidence
    return comparison



def drive_operation_counts(recorder):
    """Persistent-root (Drive) filesystem operations a recorder observed. The
    training path is required to stay at zero: any Drive read inside the loop is
    a per-step FUSE round trip, and any write is the tripwire this profiler
    exists to trip."""
    return {
        "stats": recorder.count(op="stat", is_persistent=True),
        "reads": recorder.count(op="read", is_persistent=True),
        "writes": recorder.count(op="write", is_persistent=True),
        "mkdirs": recorder.count(op="mkdir", is_persistent=True),
        "seconds": recorder.seconds(is_persistent=True),
        "write_paths": [op.path for op in recorder.matching(op="write", is_persistent=True)],
    }


def print_fit_paths(report):
    configuration = report.get("configuration") or {}
    comparison = report.get("comparison") or {}
    print("=" * 78)
    print("PHASE G -- model.fit() vs the compiled step vs the dataset")
    print("=" * 78)
    print("Configuration (read off the live model, not assumed):")
    print("  jit_compile=%s  run_eagerly=%s  steps_per_execution=%s"
          % (configuration.get("jit_compile"), configuration.get("run_eagerly"),
             configuration.get("steps_per_execution")))
    print("  optimizer=%s%s  loss_scaled=%s"
          % (configuration.get("optimizer"),
             "(inner=%s)" % configuration["inner_optimizer"] if configuration.get("inner_optimizer") else "",
             configuration.get("loss_scaled")))
    print("  model dtype policy=%s  global dtype policy=%s"
          % (configuration.get("model_dtype_policy"), configuration.get("global_dtype_policy")))
    print("  trainable: %s tensors, %s parameters"
          % (configuration.get("trainable_tensors"),
             format(configuration.get("trainable_parameters") or 0, ",")))
    print("")

    header = "%-26s %10s %12s %12s %12s" % ("path", "steps", "first", "steady mean", "total")
    print(header)
    print("-" * len(header))
    for key, label in (("dataset_only", "A. dataset only"),
                       ("compiled_step", "B. compiled train_on_batch"),
                       ("fit_minimal", "C. model.fit() minimal"),
                       ("fit_real_callbacks", "D. model.fit() real cbs")):
        section = report.get(key)
        if not section:
            continue
        first = section.get("first_step_seconds") or section.get("first_batch_seconds")
        print("%-26s %10s %12s %12s %12s"
              % (label, section.get("steps"), _ms(first),
                 _ms(section.get("steady_mean_seconds")), _ms(section.get("total_seconds"))))
    print("")
    print("Derived:")
    print("  callback overhead (D - C)          : %s" % _ms(comparison.get("callback_overhead_seconds")))
    print("  fit() overhead over compiled (C - B): %s" % _ms(comparison.get("fit_overhead_over_compiled_seconds")))
    print("  input-bound (A > B)                : %s" % comparison.get("input_bound"))
    print("  historical 4-5 s/step reproduced   : %s" % comparison.get("historical_slow_step_reproduced"))
    print("")
    print("Evidence:")
    for line in comparison.get("evidence", []):
        print("  - %s" % line)

    print("")
    print("Drive (persistent-root) operations, per path -- the training path must stay at zero:")
    for key, label in (("dataset_only", "A"), ("fit_minimal", "C"), ("fit_real_callbacks", "D")):
        section = report.get(key)
        recorder = (section or {}).get("recorder")
        if recorder is None:
            continue
        drive = drive_operation_counts(recorder)
        print("  path %s: %d stat, %d read, %d write, %d mkdir (%.2f s total)"
              % (label, drive["stats"], drive["reads"], drive["writes"], drive["mkdirs"],
                 drive["seconds"]))
        if drive["write_paths"]:
            print("      WRITE TRIPWIRE FAILED -- wrote to: %s" % drive["write_paths"][:5])

    print("")
    print("Telemetry (mean/max where sampled):")
    for key, label in (("compiled_step", "B"), ("fit_minimal", "C"), ("fit_real_callbacks", "D")):
        section = report.get(key)
        telemetry = (section or {}).get("telemetry") or {}
        gpu = telemetry.get("gpu_util_percent") or {}
        cpu = telemetry.get("cpu_percent") or {}
        if gpu or cpu:
            print("  path %s: GPU util mean=%s max=%s | CPU mean=%s max=%s"
                  % (label, gpu.get("mean"), gpu.get("max"), cpu.get("mean"), cpu.get("max")))

    print("")
    print("NOTE: this diagnostic took real gradient steps. Model weights were restored, but the")
    print("optimizer's slot/iteration state was advanced and the model is now flagged")
    print("`_dr_diagnostic_dirty` -- training.Trainer.fit() will refuse it until the model is")
    print("rebuilt and recompiled (joint_training_model.build_and_compile_joint_model(...)).")


# =====================================================================
# PHASE H -- what does one checkpoint actually cost?
# =====================================================================

CHECKPOINT_COST_WORKSPACE_PREFIX = "checkpoint_cost_"
CHECKPOINT_COST_OWNER_MARKER = ".checkpoint_cost_owner"
DEFAULT_CHECKPOINT_COST_LOCATION = os.path.join(tempfile.gettempdir(), "joint_checkpoint_cost")

_REPO_ROOT = os.path.dirname(os.path.abspath(__file__))

#: Never a diagnostic location, whatever OS this runs on. Compared as POSIX
#: strings so they are refused on a Windows host too, where `/content` would
#: otherwise resolve to `<drive>:\content`.
_PROTECTED_POSIX_PATHS = (
    "/", "/content", "/content/drive", "/content/drive/MyDrive",
    "/content/drive/Shareddrives", "/content/drive/.shortcut-targets-by-id",
)

#: `experiment_manager.create_experiment()`'s layout, repeated here so this
#: measurement-only module does not import the Colab infrastructure.
_EXPERIMENT_METADATA_FILENAME = "metadata.json"
_EXPERIMENT_SUBFOLDERS = ("checkpoints", "logs", "tensorboard", "evaluation", "predictions")


class UnsafeDiagnosticPathError(RuntimeError):
    """A diagnostic was pointed at a location it must not write to or delete."""


class CheckpointCostCleanupError(RuntimeError):
    """The diagnostic's own workspace could not be removed. Raised, never
    swallowed: a silent cleanup failure leaves ~500 MiB behind per run on Drive."""


def _delete_tree(path):
    """The single deletion primitive in this diagnostic. No `ignore_errors`:
    a failure must reach the caller (see `CheckpointCostCleanupError`)."""
    shutil.rmtree(path)


def _canonical(path):
    return os.path.normcase(os.path.realpath(os.path.abspath(path)))


def _is_same_or_ancestor(ancestor, path):
    return path == ancestor or path.startswith(ancestor.rstrip(os.sep) + os.sep)


def _protected_reason(path):
    """Why `path` is a system-critical location, or None."""
    raw = str(path).replace("\\", "/")
    if raw and posixpath.normpath(raw) in _PROTECTED_POSIX_PATHS:
        return f"{path} is a protected system/Drive root"
    canonical = _canonical(path)
    if os.path.dirname(canonical) == canonical:
        return f"{path} is a filesystem root"
    home = _canonical(os.path.expanduser("~"))
    if _is_same_or_ancestor(canonical, home):
        return f"{path} is the home directory or one of its ancestors"
    repo = _canonical(_REPO_ROOT)
    if _is_same_or_ancestor(canonical, repo) or _is_same_or_ancestor(repo, canonical):
        return f"{path} is the repository root, inside it, or one of its ancestors"
    return None


def _experiment_reason(path):
    """Why `path` belongs to an experiment or holds checkpoint state, or None.

    Walks `path` and every existing ancestor for the `create_experiment()`
    layout, so `<experiment>/checkpoints`, `<experiment>` itself and anything
    nested inside an experiment are all refused -- even an experiment that has not
    written its first checkpoint yet. Also refuses a bare checkpoint directory
    (generations / `latest.json` / `best/`) with no experiment around it."""
    from training import checkpointing as ckpt

    current = _canonical(path)
    while True:
        if (os.path.isfile(os.path.join(current, _EXPERIMENT_METADATA_FILENAME))
                and any(os.path.isdir(os.path.join(current, sub))
                        for sub in _EXPERIMENT_SUBFOLDERS)):
            return f"{current} is an experiment directory (metadata.json + experiment subfolders)"
        parent = os.path.dirname(current)
        if parent == current:
            break
        current = parent
    evidence = ckpt.checkpoint_evidence(path)
    if evidence:
        return f"{path} already holds checkpoint state ({', '.join(evidence)})"
    return None


def _assert_safe_workspace_location(location):
    reason = _protected_reason(location) or _experiment_reason(location)
    if reason:
        raise UnsafeDiagnosticPathError(
            f"Refusing to run the checkpoint-cost diagnostic at {location}: {reason}. Point it at "
            "a dedicated location such as /content/checkpoint_cost, or "
            "experiments/FinalClassification/_checkpoint_cost_probe for a Drive measurement -- "
            "never at an experiment, a checkpoint directory, or a system root. Nothing was "
            "created or deleted."
        )


def _remove_owned_workspace(workspace, location, token):
    """Delete `workspace`, and only if every ownership check passes.

    It must be a direct child of `location`, carry this module's prefix, hold
    the ownership marker written by THIS invocation (matching `token`), and not
    be a protected path. Anything else is refused -- so no bug or bad argument
    upstream can turn cleanup into deleting a caller's directory."""
    reason = _protected_reason(workspace)
    if reason:
        raise UnsafeDiagnosticPathError(f"Refusing to delete {workspace}: {reason}.")
    if os.path.dirname(_canonical(workspace)) != _canonical(location):
        raise UnsafeDiagnosticPathError(
            f"Refusing to delete {workspace}: it is not a direct child of the diagnostic "
            f"location {location}.")
    if not os.path.basename(workspace).startswith(CHECKPOINT_COST_WORKSPACE_PREFIX):
        raise UnsafeDiagnosticPathError(
            f"Refusing to delete {workspace}: it is not a checkpoint-cost workspace.")
    marker = os.path.join(workspace, CHECKPOINT_COST_OWNER_MARKER)
    if not os.path.isfile(marker):
        raise UnsafeDiagnosticPathError(
            f"Refusing to delete {workspace}: it has no ownership marker, so this invocation "
            "did not create it.")
    with open(marker) as handle:
        if handle.read().strip() != token:
            raise UnsafeDiagnosticPathError(
                f"Refusing to delete {workspace}: it belongs to a different invocation.")
    try:
        _delete_tree(workspace)
    except OSError as error:
        raise CheckpointCostCleanupError(
            f"Could not remove the checkpoint-cost workspace {workspace}: {error}. Nothing "
            "outside that workspace was touched; remove it by hand.") from error
    if os.path.exists(workspace):
        raise CheckpointCostCleanupError(
            f"The checkpoint-cost workspace {workspace} still exists after deletion.")


def measure_checkpoint_cost(model, checkpoint_dir=None, staging_dir=None, restore=True,
                            cleanup=True):
    """Time one full checkpoint save (weights + optimizer + state + manifest +
    validation + seal) and one full restore, and report the byte sizes.

    Deliberately writes to a LOCAL directory by default: this measures the
    checkpoint machinery, not Drive latency. Drive cost has to be measured on
    Colab against a real Drive path, and is reported separately as such.

    `checkpoint_dir` is the LOCATION to measure at, not a directory this
    function owns. It is never deleted, and neither is anything above it or
    beside the workspace: every call creates a fresh, uniquely named
    `checkpoint_cost_<time>_<id>/` inside it (created exclusively, so it cannot
    be a pre-existing directory), marks it with an ownership token, writes the
    generation there, and -- with `cleanup=True` -- removes exactly that
    workspace afterwards. Locations that are, or sit inside, an experiment or a
    checkpoint directory are refused, as are system roots, the home directory and
    the repository. A cleanup failure raises `CheckpointCostCleanupError`."""
    from training import checkpointing as ckpt

    requested = checkpoint_dir or DEFAULT_CHECKPOINT_COST_LOCATION
    # Check the caller's own spelling BEFORE abspath(): on a Windows host
    # abspath("/content") becomes "<drive>:\content", which no longer matches the
    # protected POSIX roots. Then check the absolute form too. Both run before
    # anything is created.
    _assert_safe_workspace_location(requested)
    location = os.path.abspath(requested)
    _assert_safe_workspace_location(location)
    os.makedirs(location, exist_ok=True)

    token = uuid.uuid4().hex
    workspace = os.path.join(
        location, f"{CHECKPOINT_COST_WORKSPACE_PREFIX}{time.strftime('%Y%m%d-%H%M%S')}_{token[:8]}")
    os.makedirs(workspace)                           # exist_ok=False: this call created it
    with open(os.path.join(workspace, CHECKPOINT_COST_OWNER_MARKER), "w") as handle:
        handle.write(token)

    try:
        report = _measure_checkpoint_cost_in(model, os.path.join(workspace, "checkpoints"),
                                             staging_dir, restore)
    except BaseException:
        if cleanup:
            try:
                _remove_owned_workspace(workspace, location, token)
            except Exception as cleanup_error:  # noqa: BLE001 -- reported, original re-raised
                print(f"WARNING: the measurement failed AND its workspace {workspace} could not "
                      f"be removed ({cleanup_error}). Remove it by hand.")
        raise

    report.update({"workspace": workspace, "workspace_location": location, "cleaned_up": False})
    if cleanup:
        _remove_owned_workspace(workspace, location, token)
        report["cleaned_up"] = True
    return report


def _measure_checkpoint_cost_in(model, checkpoint_dir, staging_dir, restore):
    from training import checkpointing as ckpt

    state = ckpt.TrainingState(
        experiment_id="checkpoint-cost", completed_epoch=1, best_epoch=0, best_metric=0.0,
    )
    start = time.perf_counter()
    generation_dir = ckpt.save_generation(checkpoint_dir, model, state, staging_dir=staging_dir)
    save_seconds = time.perf_counter() - start

    sizes = ckpt.checkpoint_size_report(generation_dir)

    start = time.perf_counter()
    validation = ckpt.validate_generation(generation_dir)
    validate_seconds = time.perf_counter() - start

    restore_seconds = None
    if restore:
        start = time.perf_counter()
        ckpt.restore_training_state(model, generation_dir, verbose=0)
        restore_seconds = time.perf_counter() - start

    return {
        "generation_dir": generation_dir,
        "save_seconds": save_seconds,
        "validate_seconds": validate_seconds,
        "restore_seconds": restore_seconds,
        "validation_ok": validation.ok,
        "sizes": sizes,
        "trainable_parameters": int(sum(int(np.prod(v.shape)) for v in model.trainable_variables)),
        "optimizer_variables": len(model.optimizer.variables) if model.optimizer else None,
    }


def print_checkpoint_cost(report):
    print("=" * 78)
    print("PHASE H -- checkpoint cost")
    print("=" * 78)
    print("  trainable parameters : %s" % format(report.get("trainable_parameters") or 0, ","))
    print("  optimizer variables  : %s" % report.get("optimizer_variables"))
    print("  save (build+validate+copy+seal): %s" % _ms(report.get("save_seconds")))
    print("  validate (sha256 re-read)      : %s" % _ms(report.get("validate_seconds")))
    print("  restore (slots+optimizer+weights): %s" % _ms(report.get("restore_seconds")))
    print("  integrity validation           : %s" % ("PASS" if report.get("validation_ok") else "FAIL"))
    print("  workspace                      : %s (%s)"
          % (report.get("workspace"), "removed" if report.get("cleaned_up") else "kept"))
    sizes = report.get("sizes") or {}
    for name, size in (sizes.get("files") or {}).items():
        print("    %-24s %14s bytes (%8.2f MiB)" % (name, format(size, ","), size / 1024 ** 2))
    total = sizes.get("total_bytes") or 0
    print("    %-24s %14s bytes (%8.2f MiB)" % ("TOTAL", format(total, ","), total / 1024 ** 2))
    print("")
    print("Local disk only. Drive/FUSE write cost is NOT measured here and must be measured on")
    print("Colab against a real Drive path before any crash-safety claim is made about Drive.")
