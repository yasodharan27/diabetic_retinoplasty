"""
Model-agnostic training orchestrator. `Trainer` wires together mixed
precision, callbacks (checkpointing, early stopping, LR scheduling,
TensorBoard, resume support), and `model.fit()` -- it never builds a model
or loads a dataset itself; both are supplied by the caller.

Usage (once a module's model + dataset loading exist):

    from training import Trainer, TrainingConfig

    config = TrainingConfig(run_dir="/content/training_runs/vessel_segmentation", epochs=50)
    trainer = Trainer(config)
    history = trainer.fit(model, train_ds, val_ds)
    trainer.export_best_weights("models/vessel_segmentation/best_model.weights.h5")
"""

import os
import shutil
from dataclasses import dataclass, field
from typing import List, Optional

import tensorflow as tf

from .callbacks import build_callbacks, get_resume_epoch


@dataclass
class TrainingConfig:
    """Generic training run configuration -- no model- or dataset-specific fields."""
    run_dir: str
    epochs: int = 50
    monitor: str = "val_loss"
    mode: str = "min"
    early_stopping_patience: int = 8
    reduce_lr_patience: int = 4
    reduce_lr_factor: float = 0.5
    min_lr: float = 1e-6
    mixed_precision: bool = True
    resume: bool = False
    save_weights_only: bool = True
    extra_callbacks: Optional[List[tf.keras.callbacks.Callback]] = field(default=None)

    @property
    def checkpoint_dir(self):
        return os.path.join(self.run_dir, "checkpoints")

    @property
    def log_dir(self):
        return os.path.join(self.run_dir, "logs")


def check_gpu():
    """Print GPU availability and enable memory growth. Returns the GPU device list."""
    gpus = tf.config.list_physical_devices("GPU")
    if gpus:
        print(f"GPU available: {[g.name for g in gpus]}")
        for g in gpus:
            try:
                tf.config.experimental.set_memory_growth(g, True)
            except RuntimeError as e:
                print(f"Could not set memory growth on {g.name}: {e}")
    else:
        print("No GPU detected -- training will run on CPU.")
    return gpus


def enable_mixed_precision(enabled=True):
    """Enable mixed_float16 when a GPU is present and `enabled` is True;
    otherwise keep float32 (mixed precision only helps on GPU/TPU)."""
    gpus = tf.config.list_physical_devices("GPU")
    if enabled and gpus:
        policy = tf.keras.mixed_precision.Policy("mixed_float16")
        tf.keras.mixed_precision.set_global_policy(policy)
        print(f"Mixed precision enabled: {policy.name}")
    else:
        if enabled and not gpus:
            print("Mixed precision requested but no GPU detected -- keeping float32.")
        tf.keras.mixed_precision.set_global_policy("float32")
    return tf.keras.mixed_precision.global_policy()


class Trainer:
    """Orchestrates a single model.fit() run with the standard callback set,
    mixed precision, and resume-training support. Model-agnostic and
    dataset-agnostic by design -- both are supplied by the caller, so the
    same Trainer works for Image Quality Assessment, Vessel Segmentation,
    Lesion Segmentation, and Final Classification alike."""

    def __init__(self, config: TrainingConfig):
        self.config = config
        self.callbacks = None
        self.paths = None
        self.history = None

    def prepare(self):
        """Set up mixed precision and build the callback set. Called
        automatically by `fit()`, but exposed separately so a caller can
        inspect `self.paths` (checkpoint/log locations) beforehand."""
        enable_mixed_precision(self.config.mixed_precision)
        self.callbacks, self.paths = build_callbacks(
            checkpoint_dir=self.config.checkpoint_dir,
            log_dir=self.config.log_dir,
            monitor=self.config.monitor,
            mode=self.config.mode,
            early_stopping_patience=self.config.early_stopping_patience,
            reduce_lr_patience=self.config.reduce_lr_patience,
            reduce_lr_factor=self.config.reduce_lr_factor,
            min_lr=self.config.min_lr,
            save_weights_only=self.config.save_weights_only,
            extra_callbacks=self.config.extra_callbacks,
        )
        return self.paths

    def resolve_initial_epoch(self):
        """Determine the epoch to resume from, per `config.resume`."""
        if not self.config.resume:
            return 0
        initial_epoch = get_resume_epoch(self.paths["epoch_state"])
        if initial_epoch > 0 and os.path.exists(self.paths["last_weights"]):
            print(f"Resuming from epoch {initial_epoch} using {self.paths['last_weights']}")
            return initial_epoch
        print("resume=True but no prior checkpoint was found; starting from scratch.")
        return 0

    def fit(self, model, train_ds, val_ds, class_weight=None):
        """Run training. `model` must already be built and compiled by the
        caller; `train_ds`/`val_ds` must already be loaded (any type
        `model.fit()` accepts). Neither is constructed here. `class_weight`
        is passed straight through to `model.fit()` for modules trained on
        imbalanced classes (e.g. Image Quality Assessment's EyeQ labels)."""
        if self.callbacks is None:
            self.prepare()

        initial_epoch = self.resolve_initial_epoch()
        if initial_epoch > 0:
            model.load_weights(self.paths["last_weights"])

        self.history = model.fit(
            train_ds,
            validation_data=val_ds,
            epochs=self.config.epochs,
            initial_epoch=initial_epoch,
            callbacks=self.callbacks,
            class_weight=class_weight,
        )
        return self.history

    def evaluate(self, model, val_ds):
        """Run model.evaluate() and print each metric. Purely a reporting
        convenience -- does not compute or assume any particular metric set."""
        results = model.evaluate(val_ds, return_dict=True, verbose=1)
        print("Evaluation results:")
        for k, v in results.items():
            print(f"  {k}: {v:.4f}")
        return results

    def best_weights_path(self):
        if self.paths is None:
            raise RuntimeError("prepare()/fit() must run before best_weights_path().")
        return self.paths["best_weights"]

    def export_best_weights(self, destination_path):
        """Copy the best checkpoint to an arbitrary destination path (e.g.
        the repository's `models/<module>/` folder). A plain file copy --
        no git or Colab-specific behavior, keeping this framework
        independent of any particular environment."""
        best_path = self.best_weights_path()
        if not os.path.exists(best_path):
            raise FileNotFoundError(f"No best checkpoint found at {best_path} -- did training run?")
        os.makedirs(os.path.dirname(destination_path) or ".", exist_ok=True)
        shutil.copy2(best_path, destination_path)
        print(f"Exported best weights to {destination_path}")
        return destination_path
