"""
Model-agnostic Keras callback construction: checkpointing, early stopping,
LR scheduling, TensorBoard, resume-training state, and CSV metric logging.

Nothing here builds a model or loads a dataset -- callers pass in an already
compiled model and an already-loaded dataset elsewhere (see `training.trainer`);
this module only wires up the callbacks that watch training as it runs.
"""

import json
import os

import tensorflow as tf


class EpochStateLogger(tf.keras.callbacks.Callback):
    """Writes the last completed epoch to a JSON file after every epoch, so a
    later run can resume training at the correct `initial_epoch`."""

    def __init__(self, state_path):
        super().__init__()
        self.state_path = state_path

    def on_epoch_end(self, epoch, logs=None):
        with open(self.state_path, "w") as f:
            json.dump({"last_completed_epoch": epoch + 1}, f)


def get_resume_epoch(state_path):
    """Return the epoch to resume from (0 if no prior run state exists)."""
    if not os.path.exists(state_path):
        return 0
    with open(state_path) as f:
        state = json.load(f)
    return state.get("last_completed_epoch", 0)


def checkpoint_paths(checkpoint_dir, save_weights_only=True):
    """Resolve the standard checkpoint/state file paths used by build_callbacks().
    File extension follows Keras' format requirement: `.weights.h5` for
    weights-only checkpoints, `.keras` for complete-model checkpoints."""
    ext = "weights.h5" if save_weights_only else "keras"
    return {
        "best_weights": os.path.join(checkpoint_dir, f"best.{ext}"),
        "last_weights": os.path.join(checkpoint_dir, f"last.{ext}"),
        "epoch_state": os.path.join(checkpoint_dir, "epoch_state.json"),
    }


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
):
    """
    Build the standard callback set: best/last ModelCheckpoint, EarlyStopping,
    ReduceLROnPlateau, TensorBoard, CSV metric logging, and epoch-state
    tracking for resume support.

    Returns (callbacks, paths) where `paths` includes the resolved
    checkpoint/log/metrics-log file locations, so the caller (typically
    `training.trainer.Trainer`) doesn't need to re-derive them.
    """
    os.makedirs(checkpoint_dir, exist_ok=True)
    os.makedirs(log_dir, exist_ok=True)

    paths = checkpoint_paths(checkpoint_dir, save_weights_only=save_weights_only)
    paths["log_dir"] = log_dir
    paths["metrics_csv"] = os.path.join(checkpoint_dir, "metrics.csv")

    callbacks = [
        tf.keras.callbacks.ModelCheckpoint(
            paths["best_weights"], monitor=monitor, mode=mode,
            save_best_only=True, save_weights_only=save_weights_only, verbose=1,
        ),
        tf.keras.callbacks.ModelCheckpoint(
            paths["last_weights"], save_best_only=False,
            save_weights_only=save_weights_only, verbose=0,
        ),
        tf.keras.callbacks.EarlyStopping(
            monitor=monitor, mode=mode, patience=early_stopping_patience,
            restore_best_weights=True, verbose=1,
        ),
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor=monitor, mode=mode, factor=reduce_lr_factor,
            patience=reduce_lr_patience, min_lr=min_lr, verbose=1,
        ),
        tf.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1),
        tf.keras.callbacks.CSVLogger(paths["metrics_csv"], append=True),
        EpochStateLogger(paths["epoch_state"]),
    ]

    if extra_callbacks:
        callbacks.extend(extra_callbacks)

    return callbacks, paths
