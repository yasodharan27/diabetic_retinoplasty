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

Precision ordering
------------------
Keras 3 captures a layer's dtype policy when the layer is CONSTRUCTED, and
decides whether to wrap the optimizer in a `LossScaleOptimizer` when the model
is COMPILED. `Trainer` receives an already-built, already-compiled model, so
calling `enable_mixed_precision()` here can no longer affect it -- the global
policy changes, the model does not. A caller that builds its model before
handing it to `Trainer` and expects `mixed_precision=True` to do something
therefore trains in float32 while believing it is in float16.

`prepare()`/`fit()` now detect exactly that and report it (see
`verify_model_precision` and `TrainingConfig.precision_check`). The fix on the
caller's side is one line: establish the policy BEFORE building the model, e.g.
`joint_training_model.build_and_compile_joint_model(mixed_precision=True)`.
"""

import os
import shutil
from dataclasses import dataclass, field
from typing import List, Optional

import tensorflow as tf

from . import checkpointing as ckpt
from .callbacks import build_callbacks, get_resume_epoch
from .checkpointing import CheckpointOptions

#: Set on a model by measurement-only diagnostics that take real gradient steps.
#: `Trainer.fit()` refuses such a model, because its optimizer slots and
#: `iterations` no longer correspond to any epoch of the real run.
DIAGNOSTIC_DIRTY_ATTRIBUTE = "_dr_diagnostic_dirty"


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
    #: "warn" (default, preserves every existing caller's behaviour), "error", or
    #: "off". Governs what happens when the model handed to `fit()` was built
    #: under a different dtype policy than `mixed_precision` asks for.
    precision_check: str = "warn"
    #: Supplying a `CheckpointOptions` switches this run to the generation-based,
    #: optimizer- and callback-state-preserving checkpoints in
    #: `training.checkpointing`. `None` keeps the original weights-only pair.
    checkpoint_options: Optional[CheckpointOptions] = None
    #: Passed through to the checkpoint manifest's git-commit lookup.
    repo_dir: Optional[str] = None

    @property
    def checkpoint_dir(self):
        return os.path.join(self.run_dir, "checkpoints")

    @property
    def log_dir(self):
        return os.path.join(self.run_dir, "logs")

    @property
    def robust_checkpointing(self):
        return self.checkpoint_options is not None


def check_gpu():
    """Print GPU availability and enable memory growth. Returns the GPU device list.

    Safe to call repeatedly, including after TensorFlow has already initialized the device.
    `set_memory_growth` raises `RuntimeError: Physical devices cannot be modified after being
    initialized`, so this checks the CURRENT setting first and only calls the setter when it would
    actually change something. Once growth is already on -- which is the normal case, since the
    first caller in a session turns it on before any op touches the GPU -- later calls are a
    silent no-op instead of printing a warning that reads like a failure but is not one."""
    gpus = tf.config.list_physical_devices("GPU")
    if gpus:
        print(f"GPU available: {[g.name for g in gpus]}")
        for g in gpus:
            try:
                if tf.config.experimental.get_memory_growth(g):
                    continue  # already enabled -- nothing to do, and setting it again would raise
            except (RuntimeError, ValueError):
                pass  # cannot be queried on this build; fall through and try to set it
            try:
                tf.config.experimental.set_memory_growth(g, True)
            except RuntimeError as e:
                # The device is already initialized and growth is NOT on. Worth saying once, but
                # it is not fatal: TF simply keeps the allocator it already built.
                print(f"Memory growth could not be enabled on {g.name} (device already "
                      f"initialized); continuing with the existing allocator: {e}")
    else:
        print("No GPU detected -- training will run on CPU.")
    return gpus


def enable_mixed_precision(enabled=True):
    """Enable mixed_float16 when a GPU is present and `enabled` is True;
    otherwise keep float32 (mixed precision only helps on GPU/TPU).

    Call this BEFORE constructing the model. Keras 3 reads the global policy in
    each layer's constructor, so a policy set afterwards changes nothing about an
    already-built model -- see this module's docstring and
    `verify_model_precision`."""
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


def expected_policy_name(mixed_precision_enabled):
    """The dtype policy `enable_mixed_precision(mixed_precision_enabled)` would
    actually produce on this machine. On a CPU-only host that is always
    `float32`, whatever the flag says."""
    gpus = tf.config.list_physical_devices("GPU")
    return "mixed_float16" if (mixed_precision_enabled and gpus) else "float32"


def _iter_layers(layer):
    yield layer
    for sub in getattr(layer, "layers", None) or ():
        yield from _iter_layers(sub)


def model_precision_policies(model):
    """Every distinct dtype policy actually captured by `model`'s weighted layers.

    Reads the sub-layers, not just `model.dtype_policy`: a Functional wrapper
    records the policy in force when the wrapper was created, which is not
    necessarily the one its layers were built with."""
    policies = set()
    for layer in _iter_layers(model):
        if getattr(layer, "weights", None):
            policy = getattr(layer, "dtype_policy", None)
            name = getattr(policy, "name", None)
            if name:
                policies.add(name)
    if not policies:
        name = getattr(getattr(model, "dtype_policy", None), "name", None)
        if name:
            policies.add(name)
    return policies


def precision_is_consistent(expected, actual):
    """Does the set of policies `actual` satisfy a run configured for `expected`?

    The rule is deliberately asymmetric, because the two directions are not the
    same kind of event:

    * `expected == "mixed_float16"`: at least one weighted layer must actually be
      mixed_float16. Individual `dtype="float32"` layer overrides are part of this
      project's design -- `racaf_output`, `fused_embedding` and
      `AdaptiveBranchFusion`'s branch-weight projection are all deliberately
      float32 for numerical stability -- so their presence is not a fault. A model
      with NO float16 layer at all is the fault: mixed precision is simply absent.
    * `expected == "float32"`: no layer may be float16. A float16 layer in a run
      that asked for float32 means the model was built under a policy nobody in
      this run selected, and (if it were compiled then too) with loss scaling
      nobody asked for."""
    if not actual:
        return True
    if expected == "mixed_float16":
        return "mixed_float16" in actual
    return set(actual) <= {expected}


def verify_model_precision(model, mixed_precision_enabled, precision_check="warn"):
    """Check that `model` was actually built under the policy this run asks for.

    Returns a report dict. `precision_check="error"` raises on a mismatch,
    `"warn"` prints an unmissable message and continues, `"off"` does neither.

    The mismatch that matters is a model built in float32 for a run configured
    with `mixed_precision=True`: training then runs entirely in float32 with a
    plain `Adam` (no `LossScaleOptimizer`), silently forfeiting the T4's
    float16 throughput. The reverse -- a float16 model in a float32-configured
    run -- is reported too, since it means the optimizer/precision pair was not
    the one intended."""
    expected = expected_policy_name(mixed_precision_enabled)
    actual = model_precision_policies(model)
    optimizer = getattr(model, "optimizer", None)
    report = {
        "expected_policy": expected,
        "model_policies": sorted(actual),
        "global_policy": tf.keras.mixed_precision.global_policy().name,
        "optimizer": type(optimizer).__name__ if optimizer is not None else None,
        "loss_scaled": optimizer is not None and hasattr(optimizer, "inner_optimizer"),
        "consistent": precision_is_consistent(expected, actual),
    }
    if report["consistent"] or precision_check == "off":
        return report

    message = (
        f"Precision policy mismatch: this run is configured for "
        f"mixed_precision={mixed_precision_enabled} (expected layer policy '{expected}'), but the "
        f"model's weighted layers were built with {sorted(actual)}. Keras 3 captures the dtype "
        "policy when each layer is CONSTRUCTED and decides on LossScaleOptimizer when the model "
        "is COMPILED, so setting the policy here cannot change an already-built model -- training "
        f"would run at {sorted(actual)} with optimizer {report['optimizer']}. Fix the ORDER: call "
        "training.enable_mixed_precision(...) (or joint_training_model."
        "build_and_compile_joint_model(mixed_precision=...)) BEFORE building the model."
    )
    if precision_check == "error":
        raise RuntimeError(message)
    print("WARNING: " + message)
    return report


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
        self.precision_report = None
        self.resume_generation_dir = None
        self.resume_state = None
        self._precision_verified = False

    def prepare(self, model=None):
        """Set up mixed precision and build the callback set. Called
        automatically by `fit()`, but exposed separately so a caller can
        inspect `self.paths` (checkpoint/log locations) beforehand.

        Passing `model` additionally verifies that the model was built under the
        dtype policy this run is configured for -- see `verify_model_precision`."""
        enable_mixed_precision(self.config.mixed_precision)
        if model is not None:
            self._verify_precision(model)
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
            checkpoint_options=self.config.checkpoint_options,
            repo_dir=self.config.repo_dir,
            restore_checkpoint_state=self.config.resume,
        )
        return self.paths

    def _verify_precision(self, model):
        self.precision_report = verify_model_precision(
            model, self.config.mixed_precision, self.config.precision_check)
        self._precision_verified = True
        return self.precision_report

    def resolve_initial_epoch(self):
        """Determine the epoch to resume from, per `config.resume`.

        With robust checkpointing the authoritative source is the newest
        generation's `state.json` (`completed_epoch`), reached through
        `find_resumable_generation()`, so a damaged newest generation
        automatically falls back to the previous known-good one. Without it, the
        original `epoch_state.json` + `last.weights.h5` pair is used unchanged."""
        if not self.config.resume:
            return 0

        if self.config.robust_checkpointing:
            generation_dir = ckpt.find_resumable_generation(self.paths["checkpoint_dir"])
            if generation_dir is None:
                print("resume=True but no resumable checkpoint generation was found; "
                      "starting from scratch.")
                return 0
            state = ckpt.read_state(generation_dir)
            self.resume_generation_dir = generation_dir
            self.resume_state = state
            print(f"Resuming from epoch {state.completed_epoch} using "
                  f"{os.path.basename(generation_dir)}")
            return state.completed_epoch

        initial_epoch = get_resume_epoch(self.paths["epoch_state"])
        if initial_epoch > 0 and os.path.exists(self.paths["last_weights"]):
            print(f"Resuming from epoch {initial_epoch} using {self.paths['last_weights']}")
            return initial_epoch
        print("resume=True but no prior checkpoint was found; starting from scratch.")
        return 0

    def restore(self, model, generation_dir=None):
        """Restore model weights + full optimizer state from a checkpoint
        generation, in the order the design fixes (slots built, optimizer
        restored, then weights). Returns the `TrainingState`."""
        if not self.config.robust_checkpointing:
            raise ckpt.CheckpointError(
                "Trainer.restore() needs generation-based checkpoints -- pass a CheckpointOptions "
                "via TrainingConfig(checkpoint_options=...). The weights-only path resumes via "
                "model.load_weights(paths['last_weights']) instead, and cannot restore optimizer "
                "or callback state at all."
            )
        generation_dir = generation_dir or self.resume_generation_dir
        if generation_dir is None:
            raise ckpt.CheckpointError("No checkpoint generation to restore from.")
        expected = self.config.checkpoint_options.expected_for(model)
        self.resume_state = ckpt.restore_training_state(model, generation_dir, expected=expected)
        return self.resume_state

    def fit(self, model, train_ds, val_ds, class_weight=None):
        """Run training. `model` must already be built and compiled by the
        caller; `train_ds`/`val_ds` must already be loaded (any type
        `model.fit()` accepts). Neither is constructed here. `class_weight`
        is passed straight through to `model.fit()` for modules trained on
        imbalanced classes (e.g. Image Quality Assessment's EyeQ labels)."""
        if getattr(model, DIAGNOSTIC_DIRTY_ATTRIBUTE, False):
            raise RuntimeError(
                "This model was used by a measurement-only diagnostic that took real gradient "
                "steps, so its optimizer slots and `iterations` no longer correspond to any epoch "
                "of a real run. Rebuild and recompile the model before training "
                "(joint_training_model.build_and_compile_joint_model(...))."
            )

        if self.callbacks is None:
            self.prepare(model)
        elif not self._precision_verified:
            self._verify_precision(model)

        initial_epoch = self.resolve_initial_epoch()
        if initial_epoch > 0:
            if self.config.robust_checkpointing:
                self.restore(model)
            else:
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
        """Path to the run's BEST (delivery) weights.

        With robust checkpointing that is `checkpoints/best/model.weights.h5`,
        which holds the globally best monitored epoch across every session of the
        experiment -- not merely the best of the most recent one."""
        if self.paths is None:
            raise RuntimeError("prepare()/fit() must run before best_weights_path().")
        if self.config.robust_checkpointing:
            return self.paths["best_generation_weights"]
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
