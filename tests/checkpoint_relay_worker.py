"""One leg of the three-process checkpoint/resume relay, run as its OWN OS process.

Not a test module (the `test_*.py` discovery pattern deliberately does not match this
filename) -- it is the worker `tests/test_checkpoint_resume.py::ThreeProcessRelayTests`
spawns via `subprocess`, because "resume works in a fresh process" cannot be proven
inside the process that saved the checkpoint.

Uses the REAL `training.Trainer`/`TrainingConfig`/`CheckpointOptions` path -- nothing is
stubbed except the model (small, so a leg runs in seconds) and the monitored metric,
which is injected from a fixed script so the best/wait/cooldown assertions are exact
rather than dependent on how a random tiny model happens to converge.

usage: checkpoint_relay_worker.py <run_dir> <leg> <target_epochs> <resume:0|1> <qwk_csv>
"""

import json
import os
import sys

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

import numpy as np  # noqa: E402
import tensorflow as tf  # noqa: E402

from training import Trainer, TrainingConfig  # noqa: E402
from training.checkpointing import CheckpointOptions  # noqa: E402
import training.checkpointing as ckpt  # noqa: E402


class ScriptedMonitor(tf.keras.callbacks.Callback):
    """Writes a fixed `val_QWK` value into the epoch's `logs` dict before any other
    callback reads it. Keras hands the SAME dict object to every callback in list order,
    so placing this first makes the monitored series deterministic across all legs."""

    def __init__(self, values, monitor="val_QWK"):
        super().__init__()
        self.values = values
        self.monitor = monitor

    def on_epoch_end(self, epoch, logs=None):
        if logs is None or epoch >= len(self.values):
            return
        logs[self.monitor] = float(self.values[epoch])


def build_model():
    tf.keras.utils.set_random_seed(1234)
    model = tf.keras.Sequential([
        tf.keras.layers.Input((8,)),
        tf.keras.layers.Dense(6, activation="relu", name="d1"),
        tf.keras.layers.Dense(1, name="d2"),
    ])
    model.compile(optimizer=tf.keras.optimizers.Adam(0.01), loss="mse")
    return model


def main():
    run_dir, leg, target_epochs, resume_flag, qwk_csv = sys.argv[1:6]
    target_epochs = int(target_epochs)
    resume = resume_flag == "1"
    scripted = [float(v) for v in qwk_csv.split(",")]

    rng = np.random.RandomState(0)
    x = rng.rand(32, 8).astype("float32")
    y = (x[:, :1] * 2.0 - 0.5).astype("float32")
    train_ds = tf.data.Dataset.from_tensor_slices((x, y)).batch(8)
    val_ds = tf.data.Dataset.from_tensor_slices((x[:16], y[:16])).batch(8)

    model = build_model()

    config = TrainingConfig(
        run_dir=run_dir,
        epochs=target_epochs,
        monitor="val_QWK",
        mode="max",
        mixed_precision=False,
        resume=resume,
        early_stopping_patience=8,
        reduce_lr_patience=2,
        reduce_lr_factor=0.5,
        checkpoint_options=CheckpointOptions(
            experiment_id="relay-test",
            config_hash="cfg-abc",
            dataset_version="ds-1",
            staging_dir=os.path.join(run_dir, "_staging"),
        ),
        extra_callbacks=[ScriptedMonitor(scripted)],
    )

    trainer = Trainer(config)
    paths = trainer.prepare(model)
    initial_epoch = trainer.resolve_initial_epoch()

    observed = {"leg": leg, "initial_epoch": initial_epoch}

    # Read state BEFORE fit(), i.e. as the fresh process actually finds it on disk.
    generation_dir = ckpt.find_resumable_generation(paths["checkpoint_dir"])
    if generation_dir is not None:
        prior = ckpt.read_state(generation_dir)
        observed["state_before"] = {
            "generation": prior.generation,
            "completed_epoch": prior.completed_epoch,
            "best_metric": prior.best_metric,
            "best_epoch": prior.best_epoch,
            "optimizer_iterations": prior.optimizer["iterations"],
            "early_stopping_wait": prior.early_stopping.get("wait"),
            "reduce_lr_wait": prior.reduce_lr.get("wait"),
            "learning_rate": prior.learning_rate,
        }
    else:
        observed["state_before"] = None

    trainer.fit(model, train_ds, val_ds)

    observed["optimizer_iterations_after"] = int(
        tf.keras.backend.get_value(model.optimizer.iterations)
    )
    observed["learning_rate_after"] = float(
        tf.keras.backend.get_value(model.optimizer.learning_rate)
    )

    final_dir = ckpt.find_resumable_generation(paths["checkpoint_dir"])
    final = ckpt.read_state(final_dir)
    observed["state_after"] = {
        "generation": final.generation,
        "completed_epoch": final.completed_epoch,
        "best_metric": final.best_metric,
        "best_epoch": final.best_epoch,
        "optimizer_iterations": final.optimizer["iterations"],
        "early_stopping_wait": final.early_stopping.get("wait"),
        "reduce_lr_wait": final.reduce_lr.get("wait"),
        "learning_rate": final.learning_rate,
    }
    observed["generations_on_disk"] = [n for n, _ in ckpt.list_generations(paths["checkpoint_dir"])]
    observed["best_dir_exists"] = os.path.isdir(ckpt.best_dir(paths["checkpoint_dir"]))
    best_state_path = os.path.join(ckpt.best_dir(paths["checkpoint_dir"]), ckpt.STATE_FILENAME)
    if os.path.exists(best_state_path):
        with open(best_state_path) as handle:
            best_state = json.load(handle)
        observed["best_checkpoint"] = {
            "best_metric": best_state.get("best_metric"),
            "best_epoch": best_state.get("best_epoch"),
        }

    print("RELAY_RESULT " + json.dumps(observed))


if __name__ == "__main__":
    main()
