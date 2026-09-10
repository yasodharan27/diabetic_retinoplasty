"""
Model-agnostic Keras callback construction: checkpointing, early stopping,
LR scheduling, TensorBoard, resume-training state, and CSV metric logging.

Nothing here builds a model or loads a dataset -- callers pass in an already
compiled model and an already-loaded dataset elsewhere (see `training.trainer`);
this module only wires up the callbacks that watch training as it runs.

Two checkpoint modes live here:

* the original weights-only pair (`best.weights.h5` / `last.weights.h5` +
  `epoch_state.json`), still the default so every existing caller behaves
  exactly as before; and
* the robust generation-based mode from `training.checkpointing`, enabled by
  passing a `CheckpointOptions` -- which additionally persists optimizer state,
  the GLOBAL best metric, and the stateful callbacks' counters, so a run split
  across many independent runtime sessions continues its trajectory instead of
  restarting it. See `training/checkpointing.py`'s docstring for the format and
  its crash-safety protocol.
"""

import json
import os

import tensorflow as tf

from . import checkpointing as ckpt
from .checkpointing import CheckpointOptions


class EpochStateLogger(tf.keras.callbacks.Callback):
    """Writes the last completed epoch to a JSON file after every epoch, so a
    later run can resume training at the correct `initial_epoch`.

    The write goes to a temporary file in the same directory and is then moved
    into place, so a runtime killed mid-write leaves the previous epoch's state
    readable rather than a truncated file that `get_resume_epoch()` cannot
    parse. (On a Drive/FUSE mount the move is not guaranteed atomic; it is still
    strictly better than truncating the live file, and the generation-based
    checkpoints in `training.checkpointing` do not depend on this file at all.)"""

    def __init__(self, state_path):
        super().__init__()
        self.state_path = state_path

    def on_epoch_end(self, epoch, logs=None):
        temporary = f"{self.state_path}.tmp"
        with open(temporary, "w") as f:
            json.dump({"last_completed_epoch": epoch + 1}, f)
            f.flush()
            os.fsync(f.fileno())
        os.replace(temporary, self.state_path)


def get_resume_epoch(state_path):
    """Return the epoch to resume from (0 if no prior run state exists)."""
    if not os.path.exists(state_path):
        return 0
    with open(state_path) as f:
        state = json.load(f)
    return state.get("last_completed_epoch", 0)


class TrainingStateCheckpoint(tf.keras.callbacks.Callback):
    """The authoritative owner of a run's checkpoint and training state.

    Replaces the two `ModelCheckpoint`s when robust checkpointing is enabled, and
    additionally does the three things `ModelCheckpoint` structurally cannot:

    1. **Global best.** `ModelCheckpoint.best` starts at `None` in every new
       process, so the first epoch of every resumed session always "improves" and
       overwrites `best.weights.h5`. This callback reads the best metric back
       from the previous generation's `state.json`, so the best is global across
       the whole experiment. That record -- not any Keras-internal attribute --
       is authoritative.
    2. **Optimizer state.** Each generation carries the full Adam state
       (iterations, learning rate, every momentum/velocity slot), so a resumed
       session continues the optimization trajectory instead of restarting it
       with zeroed moments.
    3. **Stateful callback counters.** `EarlyStopping.on_train_begin` and
       `ReduceLROnPlateau.on_train_begin` both reset their counters. This
       callback restores `best`/`wait`/`stopped_epoch`/`cooldown_counter`
       afterwards, which is why `build_callbacks()` places it AFTER both of them
       in the callback list -- ordering that `tests/test_checkpoint_resume.py`
       pins down explicitly.

    LAST vs BEST: the generation written every epoch is LAST, the trajectory a
    later session resumes from. BEST is a separate, weights-only copy of the
    globally best epoch, the intended delivery model, and is never resumed from.
    Both are written in `on_epoch_end`, before `EarlyStopping(restore_best_weights
    =True)` rewinds the in-memory model in `on_train_end`, so LAST always holds
    the real end-of-epoch trajectory.
    """

    def __init__(self, checkpoint_dir, monitor="val_QWK", mode="max", options=None,
                 early_stopping=None, reduce_lr=None, repo_dir=None, restore_state=True):
        super().__init__()
        if mode not in ("min", "max"):
            raise ValueError(f"mode must be 'min' or 'max', got {mode!r}")
        self.checkpoint_dir = checkpoint_dir
        self.monitor = monitor
        self.mode = mode
        self.options = options or CheckpointOptions()
        self.early_stopping = early_stopping
        self.reduce_lr = reduce_lr
        self.repo_dir = repo_dir
        #: Adopt a previous generation's global best and callback counters at
        #: `on_train_begin`. `Trainer` sets this from `TrainingConfig.resume`: a run
        #: that is NOT resuming must not inherit its predecessor's best, or a fresh
        #: run pointed at a populated directory would start from epoch 0 with fresh
        #: weights while claiming a best it never achieved.
        self.restore_state = restore_state

        self.best_metric = None
        self.best_epoch = None
        self.completed_epoch = 0
        self.restored_from = None
        self.last_generation_dir = None
        self._warned_missing_monitor = False

    # -- state helpers ----------------------------------------------------

    def _is_improvement(self, current, reference):
        if current is None:
            return False
        if reference is None:
            return True
        return current > reference if self.mode == "max" else current < reference

    def _current_learning_rate(self):
        optimizer = getattr(self.model, "optimizer", None)
        if optimizer is None:
            return None
        return float(tf.keras.backend.get_value(optimizer.learning_rate))

    def _capture_early_stopping(self):
        cb = self.early_stopping
        if cb is None:
            return {}
        return {
            "best": _as_float(getattr(cb, "best", None)),
            "wait": int(getattr(cb, "wait", 0) or 0),
            "stopped_epoch": int(getattr(cb, "stopped_epoch", 0) or 0),
            "best_epoch": int(getattr(cb, "best_epoch", 0) or 0),
            "patience": getattr(cb, "patience", None),
            "baseline": _as_float(getattr(cb, "baseline", None)),
            "start_from_epoch": getattr(cb, "start_from_epoch", None),
            "restore_best_weights": getattr(cb, "restore_best_weights", None),
            "monitor": getattr(cb, "monitor", None),
            "mode": getattr(cb, "mode", None),
        }

    def _capture_reduce_lr(self):
        cb = self.reduce_lr
        if cb is None:
            return {}
        return {
            "best": _as_float(getattr(cb, "best", None)),
            "wait": int(getattr(cb, "wait", 0) or 0),
            "cooldown_counter": int(getattr(cb, "cooldown_counter", 0) or 0),
            "cooldown": getattr(cb, "cooldown", None),
            "factor": getattr(cb, "factor", None),
            "patience": getattr(cb, "patience", None),
            "min_lr": getattr(cb, "min_lr", None),
            "monitor": getattr(cb, "monitor", None),
            "mode": getattr(cb, "mode", None),
        }

    def _restore_callback_counters(self, state):
        """Push the persisted counters back into the live callbacks.

        Only the counters are restored, never `min_delta`: Keras mutates
        `min_delta` in `_set_monitor_op()` (`min_delta *= -1` for `mode='min'`),
        so writing a persisted value back would double-apply that sign flip. The
        configuration fields are persisted for diagnostics only."""
        early = state.early_stopping or {}
        if self.early_stopping is not None and early:
            self.early_stopping.best = early.get("best")
            self.early_stopping.wait = int(early.get("wait", 0) or 0)
            self.early_stopping.stopped_epoch = int(early.get("stopped_epoch", 0) or 0)
            self.early_stopping.best_epoch = int(early.get("best_epoch", 0) or 0)

        reduce_lr = state.reduce_lr or {}
        if self.reduce_lr is not None and reduce_lr:
            self.reduce_lr.best = reduce_lr.get("best")
            self.reduce_lr.wait = int(reduce_lr.get("wait", 0) or 0)
            self.reduce_lr.cooldown_counter = int(reduce_lr.get("cooldown_counter", 0) or 0)

    # -- Keras hooks ------------------------------------------------------

    def on_train_begin(self, logs=None):
        generation_dir = ckpt.find_resumable_generation(self.checkpoint_dir)
        if generation_dir is None:
            return
        if not self.restore_state:
            if self.options.verbose:
                print(f"{os.path.basename(generation_dir)} exists but this run is not resuming: "
                      "its global best and callback counters are NOT adopted. New generations "
                      "continue the numbering, so nothing already on disk is overwritten.")
            return
        state = ckpt.read_state(generation_dir)
        self.best_metric = state.best_metric
        self.best_epoch = state.best_epoch
        self.completed_epoch = state.completed_epoch
        self.restored_from = generation_dir
        self.last_generation_dir = generation_dir
        self._restore_callback_counters(state)
        if self.options.verbose:
            print(f"Training state restored from {os.path.basename(generation_dir)}: "
                  f"{state.completed_epoch} epoch(s) completed, global best "
                  f"{self.monitor}={state.best_metric} at epoch {state.best_epoch}.")

    def on_epoch_end(self, epoch, logs=None):
        logs = logs or {}
        current = logs.get(self.monitor)
        if current is None and not self._warned_missing_monitor:
            self._warned_missing_monitor = True
            print(f"TrainingStateCheckpoint: '{self.monitor}' is not in this epoch's logs "
                  f"(available: {sorted(logs)}). The checkpoint will still be written, but the "
                  "global best cannot advance until the monitored metric appears.")
        current = _as_float(current)

        improved = self._is_improvement(current, self.best_metric)
        if improved:
            self.best_metric = current
            self.best_epoch = epoch
        self.completed_epoch = epoch + 1

        state = ckpt.TrainingState(
            experiment_id=self.options.experiment_id,
            completed_epoch=self.completed_epoch,
            best_epoch=self.best_epoch,
            best_metric=self.best_metric,
            monitor=self.monitor,
            monitor_mode=self.mode,
            learning_rate=self._current_learning_rate(),
            early_stopping=self._capture_early_stopping(),
            reduce_lr=self._capture_reduce_lr(),
            config_hash=self.options.config_hash,
            dataset_version=self.options.dataset_version,
            extra=dict(self.options.extra_metadata or {}, epoch_logs=_json_logs(logs)),
        )

        generation_dir = ckpt.save_generation(
            self.checkpoint_dir, self.model, state,
            staging_dir=self.options.staging_dir,
            keep_generations=self.options.keep_generations,
            repo_dir=self.repo_dir,
            verbose=self.options.verbose,
        )
        self.last_generation_dir = generation_dir

        if improved:
            ckpt.save_best(self.checkpoint_dir, generation_dir, state,
                           staging_dir=self.options.staging_dir,
                           verbose=self.options.verbose)


def _as_float(value):
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _json_logs(logs):
    """Keras logs hold NumPy scalars; keep only what round-trips through JSON."""
    out = {}
    for key, value in logs.items():
        coerced = _as_float(value)
        if coerced is not None:
            out[str(key)] = coerced
    return out


def checkpoint_paths(checkpoint_dir, save_weights_only=True, robust=False):
    """Resolve the standard checkpoint/state file paths used by build_callbacks().
    File extension follows Keras' format requirement: `.weights.h5` for
    weights-only checkpoints, `.keras` for complete-model checkpoints.

    `robust=True` additionally resolves the generation-based locations from
    `training.checkpointing` (`latest.json` and the `best/` directory); the
    legacy keys are still returned either way so nothing that reads them breaks."""
    ext = "weights.h5" if save_weights_only else "keras"
    paths = {
        "checkpoint_dir": checkpoint_dir,
        "best_weights": os.path.join(checkpoint_dir, f"best.{ext}"),
        "last_weights": os.path.join(checkpoint_dir, f"last.{ext}"),
        "epoch_state": os.path.join(checkpoint_dir, "epoch_state.json"),
    }
    if robust:
        paths["latest_json"] = ckpt.latest_path(checkpoint_dir)
        paths["best_dir"] = ckpt.best_dir(checkpoint_dir)
        paths["best_generation_weights"] = os.path.join(
            ckpt.best_dir(checkpoint_dir), ckpt.MODEL_WEIGHTS_FILENAME)
    return paths


def build_callbacks(
    checkpoint_dir,
    log_dir,
    monitor="val_loss",
    mode="min",
    early_stopping_patience=8,
    reduce_lr_patience=4,
    reduce_lr_factor=0.5,
    min_lr=1e-6,
    save_weights_only=True,
    extra_callbacks=None,
    checkpoint_options=None,
    repo_dir=None,
    restore_checkpoint_state=True,
):
    """
    Build the standard callback set: best/last checkpointing, EarlyStopping,
    ReduceLROnPlateau, TensorBoard, CSV metric logging, and epoch-state
    tracking for resume support.

    Returns (callbacks, paths) where `paths` includes the resolved
    checkpoint/log/metrics-log file locations, so the caller (typically
    `training.trainer.Trainer`) doesn't need to re-derive them.

    Passing `checkpoint_options` (a `training.checkpointing.CheckpointOptions`)
    switches the two `ModelCheckpoint`s and `EpochStateLogger` for a single
    `TrainingStateCheckpoint`, which persists optimizer and callback state as
    well as weights. Without it the callback set is exactly what it has always
    been.

    `extra_callbacks` are placed FIRST so a caller-supplied callback can enrich
    the shared `logs` dict before the standard callbacks read it -- Keras hands
    the same dict to every callback in list order, and a monitored value added
    after `EarlyStopping` has already looked would be invisible to it.
    """
    os.makedirs(checkpoint_dir, exist_ok=True)
    os.makedirs(log_dir, exist_ok=True)

    robust = checkpoint_options is not None
    paths = checkpoint_paths(checkpoint_dir, save_weights_only=save_weights_only, robust=robust)
    paths["log_dir"] = log_dir
    paths["metrics_csv"] = os.path.join(checkpoint_dir, "metrics.csv")

    early_stopping = tf.keras.callbacks.EarlyStopping(
        monitor=monitor, mode=mode, patience=early_stopping_patience,
        restore_best_weights=True, verbose=1,
    )
    reduce_lr = tf.keras.callbacks.ReduceLROnPlateau(
        monitor=monitor, mode=mode, factor=reduce_lr_factor,
        patience=reduce_lr_patience, min_lr=min_lr, verbose=1,
    )

    callbacks = list(extra_callbacks) if extra_callbacks else []

    if not robust:
        callbacks.extend([
            tf.keras.callbacks.ModelCheckpoint(
                paths["best_weights"], monitor=monitor, mode=mode,
                save_best_only=True, save_weights_only=save_weights_only, verbose=1,
            ),
            tf.keras.callbacks.ModelCheckpoint(
                paths["last_weights"], save_best_only=False,
                save_weights_only=save_weights_only, verbose=0,
            ),
        ])

    callbacks.extend([
        early_stopping,
        reduce_lr,
        tf.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1),
        tf.keras.callbacks.CSVLogger(paths["metrics_csv"], append=True),
    ])

    if robust:
        # Must come after EarlyStopping/ReduceLROnPlateau: their on_train_begin
        # resets exactly the counters this callback restores.
        callbacks.append(TrainingStateCheckpoint(
            checkpoint_dir=checkpoint_dir, monitor=monitor, mode=mode,
            options=checkpoint_options, early_stopping=early_stopping,
            reduce_lr=reduce_lr, repo_dir=repo_dir,
            restore_state=restore_checkpoint_state,
        ))
    else:
        callbacks.append(EpochStateLogger(paths["epoch_state"]))

    return callbacks, paths
