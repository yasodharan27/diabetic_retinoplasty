"""
Tests for the robust checkpoint/resume infrastructure (`training/checkpointing.py`
+ `training.callbacks.TrainingStateCheckpoint` + `training.Trainer`'s resume path).

Written BEFORE the implementation, per the task's Phase 1 requirement, and kept as
the regression suite afterwards. Every test here exercises the real modules -- the
only stand-in is the model itself (small, so the suite runs in seconds), because
every mechanism under test (generation layout, manifest integrity, optimizer-slot
restoration, callback-state persistence, LAST-vs-BEST) is model-agnostic by
construction. Fidelity at the real 43.3M-parameter scale is a separate, explicitly
labelled measurement, not something a unit test should pay for.

`ThreeProcessRelayTests` really does spawn three separate OS processes
(`tests/checkpoint_relay_worker.py`) -- "resume works in a fresh process" is not
provable inside the process that wrote the checkpoint.
"""

import hashlib
import json
import os
import shutil
import subprocess
import sys
import tempfile
import unittest

import numpy as np
import tensorflow as tf

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

import training.checkpointing as ckpt
from training import Trainer, TrainingConfig
from training.callbacks import TrainingStateCheckpoint
from training.checkpointing import CheckpointOptions


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------

def build_tiny_model(seed=1234, learning_rate=0.01, units=6):
    tf.keras.utils.set_random_seed(seed)
    model = tf.keras.Sequential([
        tf.keras.layers.Input((8,)),
        tf.keras.layers.Dense(units, activation="relu", name="d1"),
        tf.keras.layers.Dense(1, name="d2"),
    ])
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate), loss="mse")
    return model


def tiny_data(n=32, seed=0):
    rng = np.random.RandomState(seed)
    x = rng.rand(n, 8).astype("float32")
    y = (x[:, :1] * 2.0 - 0.5).astype("float32")
    return x, y


def train_a_few_steps(model, steps=3):
    x, y = tiny_data()
    model.fit(x, y, epochs=steps, batch_size=8, verbose=0)
    return model


def weight_signature(model):
    return hashlib.sha256(
        b"".join(np.ascontiguousarray(v.numpy()).tobytes() for v in model.weights)
    ).hexdigest()


def optimizer_signature(optimizer):
    return [round(float(np.abs(np.asarray(v)).sum()), 6) for v in optimizer.variables]


def make_state(**overrides):
    base = dict(
        experiment_id="exp-1",
        generation=1,
        completed_epoch=1,
        best_epoch=0,
        best_metric=0.5,
        monitor="val_QWK",
        monitor_mode="max",
        learning_rate=0.01,
        config_hash="cfg-abc",
        dataset_version="ds-1",
    )
    base.update(overrides)
    return ckpt.TrainingState(**base)


class ScriptedMonitor(tf.keras.callbacks.Callback):
    """Injects a deterministic `val_QWK` series into the shared epoch `logs` dict."""

    def __init__(self, values, monitor="val_QWK"):
        super().__init__()
        self.values = values
        self.monitor = monitor

    def on_epoch_end(self, epoch, logs=None):
        if logs is None or epoch >= len(self.values):
            return
        logs[self.monitor] = float(self.values[epoch])


class TempDirTestCase(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.mkdtemp(prefix="ckpt_test_")
        self.addCleanup(shutil.rmtree, self.tmp, True)
        self.checkpoint_dir = os.path.join(self.tmp, "checkpoints")
        os.makedirs(self.checkpoint_dir, exist_ok=True)

    def save_one(self, model, **state_overrides):
        state = make_state(**state_overrides)
        return ckpt.save_generation(self.checkpoint_dir, model, state,
                                    staging_dir=os.path.join(self.tmp, "_staging"))


# ===========================================================================
# 7, 8 -- model weight persistence + same-process save/load
# ===========================================================================

class SameProcessSaveLoadTests(TempDirTestCase):
    def test_generation_layout_matches_specification(self):
        model = train_a_few_steps(build_tiny_model())
        generation_dir = self.save_one(model)

        self.assertEqual(os.path.basename(generation_dir), "gen_00001")
        for filename in (ckpt.MODEL_WEIGHTS_FILENAME, ckpt.OPTIMIZER_FILENAME,
                         ckpt.STATE_FILENAME, ckpt.MANIFEST_FILENAME, ckpt.READY_FILENAME):
            self.assertTrue(os.path.exists(os.path.join(generation_dir, filename)),
                            f"missing {filename}")
        self.assertTrue(os.path.exists(os.path.join(self.checkpoint_dir, ckpt.LATEST_FILENAME)))

    def test_model_weights_round_trip_exactly(self):
        model = train_a_few_steps(build_tiny_model())
        expected = weight_signature(model)
        generation_dir = self.save_one(model)

        fresh = build_tiny_model()
        self.assertNotEqual(weight_signature(fresh), expected)
        ckpt.restore_training_state(fresh, generation_dir)
        self.assertEqual(weight_signature(fresh), expected)

    def test_weights_file_does_not_duplicate_the_optimizer_state(self):
        """Keras 3's `save_weights` includes the optimizer once its slots exist.
        Left alone that puts the whole Adam state in `model.weights.h5` AS WELL AS
        in `optimizer.npz` -- measured at 521,441,264 bytes vs 173.4 MB of real
        model variables on this project's joint model. The weights file must hold
        model variables only."""
        model = train_a_few_steps(build_tiny_model())
        self.assertTrue(model.optimizer.built)
        parameters = sum(int(np.prod(v.shape)) for v in model.weights)
        generation_dir = self.save_one(model)

        size = os.path.getsize(os.path.join(generation_dir, ckpt.MODEL_WEIGHTS_FILENAME))
        # Model variables plus HDF5 overhead, and nowhere near the 3x an
        # optimizer-carrying file would cost.
        self.assertLess(size, parameters * 4 * 2 + 100_000)

    def test_restore_emits_no_skipping_optimizer_warning(self):
        """That warning is benign here (the optimizer came from `optimizer.npz` a
        moment earlier) but is indistinguishable from the silent optimizer reset
        this module exists to prevent, so it must not appear at all."""
        import warnings

        model = train_a_few_steps(build_tiny_model())
        generation_dir = self.save_one(model)
        fresh = build_tiny_model()
        train_a_few_steps(fresh, steps=1)  # build the optimizer slots first

        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            ckpt.restore_training_state(fresh, generation_dir, verbose=0)
        messages = [str(w.message) for w in caught if "Skipping variable loading" in str(w.message)]
        self.assertEqual(messages, [])

    def test_state_json_stays_human_readable(self):
        """The full 820-entry optimizer variable list belongs in `optimizer.npz`,
        where it validates the restore -- not in the state file someone opens to
        see where a run got to."""
        model = train_a_few_steps(build_tiny_model())
        generation_dir = self.save_one(model)
        size = os.path.getsize(os.path.join(generation_dir, ckpt.STATE_FILENAME))
        self.assertLess(size, 8192)

        state = ckpt.read_state(generation_dir)
        self.assertIn("variable_count", state.optimizer)
        self.assertIn("variables_signature", state.optimizer)
        self.assertNotIn("variables", state.optimizer)

    def test_staging_directory_is_left_clean(self):
        model = train_a_few_steps(build_tiny_model())
        staging = os.path.join(self.tmp, "_staging")
        ckpt.save_generation(self.checkpoint_dir, model, make_state(), staging_dir=staging)
        leftovers = os.listdir(staging) if os.path.isdir(staging) else []
        self.assertEqual(leftovers, [], f"staging not cleaned: {leftovers}")


# ===========================================================================
# 5, 6, 18 -- optimizer iteration + Adam slot persistence, and continuation
# ===========================================================================

class OptimizerStatePersistenceTests(TempDirTestCase):
    def test_optimizer_iterations_persist(self):
        model = train_a_few_steps(build_tiny_model(), steps=3)
        iterations = int(tf.keras.backend.get_value(model.optimizer.iterations))
        self.assertGreater(iterations, 0)
        generation_dir = self.save_one(model)

        fresh = build_tiny_model()
        self.assertEqual(int(tf.keras.backend.get_value(fresh.optimizer.iterations)), 0)
        ckpt.restore_training_state(fresh, generation_dir)
        self.assertEqual(int(tf.keras.backend.get_value(fresh.optimizer.iterations)), iterations)

    def test_adam_momentum_and_velocity_slots_persist_exactly(self):
        model = train_a_few_steps(build_tiny_model(), steps=3)
        expected = optimizer_signature(model.optimizer)
        # A trained Adam has non-zero moments; a fresh one does not. Without that, this
        # test could pass on two all-zero optimizers.
        self.assertGreater(sum(1 for s in expected if s > 0), 2)
        generation_dir = self.save_one(model)

        fresh = build_tiny_model()
        ckpt.restore_training_state(fresh, generation_dir)
        self.assertEqual(optimizer_signature(fresh.optimizer), expected)

    def test_restore_builds_optimizer_slots_when_not_yet_built(self):
        model = train_a_few_steps(build_tiny_model(), steps=2)
        generation_dir = self.save_one(model)

        fresh = build_tiny_model()
        self.assertFalse(fresh.optimizer.built)
        ckpt.restore_training_state(fresh, generation_dir)
        self.assertTrue(fresh.optimizer.built)
        self.assertEqual(len(fresh.optimizer.variables), len(model.optimizer.variables))

    def test_optimizer_state_continues_rather_than_resetting(self):
        model = train_a_few_steps(build_tiny_model(), steps=3)
        generation_dir = self.save_one(model)
        before = int(tf.keras.backend.get_value(model.optimizer.iterations))

        fresh = build_tiny_model()
        ckpt.restore_training_state(fresh, generation_dir)
        x, y = tiny_data()
        fresh.fit(x, y, epochs=1, batch_size=8, verbose=0)
        after = int(tf.keras.backend.get_value(fresh.optimizer.iterations))
        self.assertEqual(after, before + 4)  # 32 samples / batch 8 = 4 steps

    def test_incomplete_optimizer_state_fails_loudly_instead_of_resetting(self):
        model = train_a_few_steps(build_tiny_model(), steps=2)
        generation_dir = self.save_one(model)

        # A checkpoint whose optimizer archive is missing variables must NOT silently
        # fall back to a zero-initialized optimizer.
        archive = os.path.join(generation_dir, ckpt.OPTIMIZER_FILENAME)
        loaded = dict(np.load(archive, allow_pickle=False))
        spec = json.loads(str(loaded.pop(ckpt.OPTIMIZER_SPEC_KEY)))
        dropped = sorted(k for k in loaded if k.startswith("v"))[-1]
        del loaded[dropped]
        spec["variables"] = spec["variables"][:-1]
        loaded[ckpt.OPTIMIZER_SPEC_KEY] = np.array(json.dumps(spec))
        np.savez(archive, **loaded)

        fresh = build_tiny_model()
        with self.assertRaises(ckpt.CheckpointIntegrityError):
            ckpt.restore_training_state(fresh, generation_dir, verify_checksums=False)

    def test_optimizer_variable_shape_mismatch_is_refused(self):
        model = train_a_few_steps(build_tiny_model(units=6), steps=2)
        generation_dir = self.save_one(model)

        wider = build_tiny_model(units=9)  # different architecture -> different slot shapes
        with self.assertRaises(ckpt.CheckpointIntegrityError):
            ckpt.restore_training_state(wider, generation_dir)


# ===========================================================================
# 20 -- manifest checksums and sizes are validated
# ===========================================================================

class ManifestIntegrityTests(TempDirTestCase):
    def setUp(self):
        super().setUp()
        self.model = train_a_few_steps(build_tiny_model(), steps=2)
        self.generation_dir = self.save_one(self.model)

    def test_manifest_records_size_and_sha256_for_every_required_file(self):
        with open(os.path.join(self.generation_dir, ckpt.MANIFEST_FILENAME)) as handle:
            manifest = json.load(handle)
        for filename in (ckpt.MODEL_WEIGHTS_FILENAME, ckpt.OPTIMIZER_FILENAME, ckpt.STATE_FILENAME):
            entry = manifest["files"][filename]
            actual = os.path.join(self.generation_dir, filename)
            self.assertEqual(entry["size"], os.path.getsize(actual))
            self.assertEqual(entry["sha256"], ckpt.sha256_file(actual))
            self.assertEqual(len(entry["sha256"]), 64)

    def test_manifest_records_environment_and_configuration_fields(self):
        with open(os.path.join(self.generation_dir, ckpt.MANIFEST_FILENAME)) as handle:
            manifest = json.load(handle)
        for key in ("experiment_id", "generation", "completed_epoch", "best_epoch",
                    "best_metric", "git_commit_hash", "tensorflow_version", "keras_version",
                    "python_version", "optimizer_type", "precision_policy", "config_hash",
                    "dataset_version", "checkpoint_format", "format_version"):
            self.assertIn(key, manifest)

    def test_a_truncated_file_fails_validation(self):
        target = os.path.join(self.generation_dir, ckpt.MODEL_WEIGHTS_FILENAME)
        with open(target, "r+b") as handle:
            handle.truncate(os.path.getsize(target) // 2)
        result = ckpt.validate_generation(self.generation_dir)
        self.assertFalse(result.ok)
        self.assertIn("size", result.reason.lower())

    def test_a_corrupted_but_same_size_file_fails_checksum_validation(self):
        target = os.path.join(self.generation_dir, ckpt.OPTIMIZER_FILENAME)
        with open(target, "rb") as handle:
            data = bytearray(handle.read())
        data[len(data) // 2] ^= 0xFF
        with open(target, "wb") as handle:
            handle.write(bytes(data))
        result = ckpt.validate_generation(self.generation_dir)
        self.assertFalse(result.ok)
        self.assertIn("sha256", result.reason.lower())

    def test_a_tampered_manifest_fails_against_the_ready_marker(self):
        manifest_path = os.path.join(self.generation_dir, ckpt.MANIFEST_FILENAME)
        with open(manifest_path) as handle:
            manifest = json.load(handle)
        manifest["files"][ckpt.MODEL_WEIGHTS_FILENAME]["size"] = 1
        with open(manifest_path, "w") as handle:
            json.dump(manifest, handle)
        result = ckpt.validate_generation(self.generation_dir)
        self.assertFalse(result.ok)

    def test_restore_refuses_a_checkpoint_that_fails_integrity(self):
        target = os.path.join(self.generation_dir, ckpt.MODEL_WEIGHTS_FILENAME)
        with open(target, "r+b") as handle:
            handle.truncate(os.path.getsize(target) // 2)
        fresh = build_tiny_model()
        with self.assertRaises(ckpt.CheckpointIntegrityError):
            ckpt.restore_training_state(fresh, self.generation_dir)


# ===========================================================================
# 10, 11, 15, 16 -- crash / partial / corrupt / READY semantics and fallback
# ===========================================================================

class CrashAndFallbackTests(TempDirTestCase):
    def setUp(self):
        super().setUp()
        self.model = build_tiny_model()
        train_a_few_steps(self.model, steps=2)
        self.gen1 = self.save_one(self.model, generation=1, completed_epoch=1, best_metric=0.60)
        train_a_few_steps(self.model, steps=2)
        self.gen2 = self.save_one(self.model, generation=2, completed_epoch=2, best_metric=0.70)

    def test_generation_without_ready_is_not_resumable(self):
        os.remove(os.path.join(self.gen2, ckpt.READY_FILENAME))
        result = ckpt.validate_generation(self.gen2)
        self.assertFalse(result.ok)
        self.assertIn("ready", result.reason.lower())
        self.assertEqual(ckpt.find_resumable_generation(self.checkpoint_dir), self.gen1)

    def test_partially_written_newest_generation_falls_back_to_previous(self):
        # Simulate a crash mid-copy: files present but READY never written.
        gen3 = os.path.join(self.checkpoint_dir, "gen_00003")
        os.makedirs(gen3, exist_ok=True)
        shutil.copy2(os.path.join(self.gen2, ckpt.STATE_FILENAME), gen3)
        self.assertEqual(ckpt.find_resumable_generation(self.checkpoint_dir), self.gen2)

    def test_incomplete_newest_generation_does_not_destroy_previous_known_good(self):
        gen3 = os.path.join(self.checkpoint_dir, "gen_00003")
        os.makedirs(gen3, exist_ok=True)
        self.assertTrue(ckpt.validate_generation(self.gen2).ok)
        self.assertTrue(ckpt.validate_generation(self.gen1).ok)
        fresh = build_tiny_model()
        state = ckpt.restore_training_state(fresh, ckpt.find_resumable_generation(self.checkpoint_dir))
        self.assertEqual(state.completed_epoch, 2)

    def test_corrupt_newest_generation_falls_back_to_previous_valid_one(self):
        target = os.path.join(self.gen2, ckpt.MODEL_WEIGHTS_FILENAME)
        with open(target, "rb") as handle:
            data = bytearray(handle.read())
        data[len(data) // 2] ^= 0xFF
        with open(target, "wb") as handle:
            handle.write(bytes(data))
        resolved = ckpt.find_resumable_generation(self.checkpoint_dir)
        self.assertEqual(resolved, self.gen1)
        fresh = build_tiny_model()
        state = ckpt.restore_training_state(fresh, resolved)
        self.assertEqual(state.completed_epoch, 1)

    def test_latest_pointer_to_a_missing_generation_falls_back_automatically(self):
        shutil.rmtree(self.gen2)
        # latest.json still names gen_00002, which no longer exists.
        pointer = ckpt.read_latest_pointer(self.checkpoint_dir)
        self.assertEqual(pointer, "gen_00002")
        self.assertEqual(ckpt.find_resumable_generation(self.checkpoint_dir), self.gen1)

    def test_previous_generation_is_retained_and_older_ones_pruned(self):
        train_a_few_steps(self.model, steps=1)
        self.save_one(self.model, generation=3, completed_epoch=3)
        numbers = [n for n, _ in ckpt.list_generations(self.checkpoint_dir)]
        self.assertEqual(numbers, [2, 3], "must keep exactly the latest two known-good generations")

    def test_pruning_never_removes_the_generation_latest_points_at(self):
        for extra in range(4, 7):
            train_a_few_steps(self.model, steps=1)
            self.save_one(self.model, generation=extra, completed_epoch=extra)
        pointer = ckpt.read_latest_pointer(self.checkpoint_dir)
        self.assertTrue(os.path.isdir(os.path.join(self.checkpoint_dir, pointer)))
        self.assertTrue(ckpt.validate_generation(
            os.path.join(self.checkpoint_dir, pointer)).ok)

    def test_no_resumable_generation_returns_none_rather_than_raising(self):
        empty = os.path.join(self.tmp, "empty_checkpoints")
        os.makedirs(empty)
        self.assertIsNone(ckpt.find_resumable_generation(empty))


# ===========================================================================
# 13, 14 -- configuration / metadata mismatch refusal
# ===========================================================================

class CompatibilityRefusalTests(TempDirTestCase):
    def setUp(self):
        super().setUp()
        model = train_a_few_steps(build_tiny_model(), steps=2)
        self.generation_dir = self.save_one(model, config_hash="cfg-abc", dataset_version="ds-1")

    def _manifest(self):
        with open(os.path.join(self.generation_dir, ckpt.MANIFEST_FILENAME)) as handle:
            return json.load(handle)

    def test_matching_configuration_is_accepted(self):
        ckpt.assert_compatible(self._manifest(), {"config_hash": "cfg-abc"})

    def test_configuration_hash_mismatch_is_refused(self):
        with self.assertRaises(ckpt.CheckpointCompatibilityError) as caught:
            ckpt.assert_compatible(self._manifest(), {"config_hash": "cfg-DIFFERENT"})
        self.assertIn("config_hash", str(caught.exception))

    def test_optimizer_type_mismatch_is_refused(self):
        with self.assertRaises(ckpt.CheckpointCompatibilityError):
            ckpt.assert_compatible(self._manifest(), {"optimizer_type": "SGD"})

    def test_precision_policy_mismatch_is_refused(self):
        with self.assertRaises(ckpt.CheckpointCompatibilityError):
            ckpt.assert_compatible(self._manifest(), {"precision_policy": "mixed_float16"})

    def test_checkpoint_format_version_mismatch_is_refused(self):
        manifest = self._manifest()
        manifest["format_version"] = ckpt.CHECKPOINT_FORMAT_VERSION + 99
        with self.assertRaises(ckpt.CheckpointCompatibilityError):
            ckpt.assert_compatible(manifest, {})

    def test_advisory_mismatches_warn_but_do_not_refuse(self):
        report = ckpt.compatibility_report(
            self._manifest(),
            {"git_commit_hash": "deadbeef", "dataset_version": "ds-2"},
        )
        self.assertTrue(report.compatible)
        self.assertTrue(report.warnings)

    def test_restore_refuses_when_expected_configuration_mismatches(self):
        fresh = build_tiny_model()
        with self.assertRaises(ckpt.CheckpointCompatibilityError):
            ckpt.restore_training_state(fresh, self.generation_dir,
                                        expected={"config_hash": "cfg-OTHER"})

    def test_config_hash_is_stable_and_order_independent(self):
        a = ckpt.config_hash({"batch_size": 2, "epochs": 50, "monitor": "val_QWK"})
        b = ckpt.config_hash({"monitor": "val_QWK", "epochs": 50, "batch_size": 2})
        c = ckpt.config_hash({"batch_size": 4, "epochs": 50, "monitor": "val_QWK"})
        self.assertEqual(a, b)
        self.assertNotEqual(a, c)


# ===========================================================================
# 1, 2, 3, 4, 12 -- callback state, global best, LAST vs BEST (in-process)
# ===========================================================================

class CallbackStatePersistenceTests(TempDirTestCase):
    def _run_leg(self, run_dir, epochs, resume, scripted, model=None,
                 reduce_lr_patience=2, config_hash="cfg-abc"):
        model = model or build_tiny_model()
        x, y = tiny_data()
        train_ds = tf.data.Dataset.from_tensor_slices((x, y)).batch(8)
        val_ds = tf.data.Dataset.from_tensor_slices((x[:16], y[:16])).batch(8)
        config = TrainingConfig(
            run_dir=run_dir, epochs=epochs, monitor="val_QWK", mode="max",
            mixed_precision=False, resume=resume,
            early_stopping_patience=8, reduce_lr_patience=reduce_lr_patience,
            reduce_lr_factor=0.5,
            checkpoint_options=CheckpointOptions(
                experiment_id="exp-1", config_hash=config_hash, dataset_version="ds-1",
                staging_dir=os.path.join(self.tmp, "_staging"),
            ),
            extra_callbacks=[ScriptedMonitor(scripted)],
        )
        trainer = Trainer(config)
        trainer.prepare(model)
        trainer.fit(model, train_ds, val_ds)
        return trainer, model

    def test_global_best_survives_a_resume_that_scores_worse(self):
        run_dir = os.path.join(self.tmp, "run")
        # Leg 1: epochs 0-1, best val_QWK 0.72.
        self._run_leg(run_dir, epochs=2, resume=False, scripted=[0.70, 0.72])
        checkpoint_dir = os.path.join(run_dir, "checkpoints")
        state = ckpt.read_state(ckpt.find_resumable_generation(checkpoint_dir))
        self.assertAlmostEqual(state.best_metric, 0.72, places=6)
        self.assertEqual(state.best_epoch, 1)

        # Leg 2: epochs 2-3, both WORSE than 0.72. The global best must not regress.
        self._run_leg(run_dir, epochs=4, resume=True, scripted=[0, 0, 0.70, 0.68])
        state = ckpt.read_state(ckpt.find_resumable_generation(checkpoint_dir))
        self.assertAlmostEqual(state.best_metric, 0.72, places=6)
        self.assertEqual(state.best_epoch, 1)

        # Leg 3: epoch 4 beats it.
        self._run_leg(run_dir, epochs=5, resume=True, scripted=[0, 0, 0, 0, 0.75])
        state = ckpt.read_state(ckpt.find_resumable_generation(checkpoint_dir))
        self.assertAlmostEqual(state.best_metric, 0.75, places=6)
        self.assertEqual(state.best_epoch, 4)

    def test_best_checkpoint_file_is_not_overwritten_by_a_worse_resumed_epoch(self):
        run_dir = os.path.join(self.tmp, "run")
        self._run_leg(run_dir, epochs=2, resume=False, scripted=[0.70, 0.72])
        checkpoint_dir = os.path.join(run_dir, "checkpoints")
        best_weights = os.path.join(ckpt.best_dir(checkpoint_dir), ckpt.MODEL_WEIGHTS_FILENAME)
        digest_before = ckpt.sha256_file(best_weights)

        self._run_leg(run_dir, epochs=4, resume=True, scripted=[0, 0, 0.10, 0.05])
        self.assertEqual(ckpt.sha256_file(best_weights), digest_before,
                         "a worse epoch overwrote the global best checkpoint")

    def test_last_and_best_are_distinct_checkpoints(self):
        run_dir = os.path.join(self.tmp, "run")
        # Best is epoch 0; epochs 1-2 are worse, so LAST must differ from BEST.
        self._run_leg(run_dir, epochs=3, resume=False, scripted=[0.90, 0.20, 0.10])
        checkpoint_dir = os.path.join(run_dir, "checkpoints")
        last_dir = ckpt.find_resumable_generation(checkpoint_dir)

        last_weights = os.path.join(last_dir, ckpt.MODEL_WEIGHTS_FILENAME)
        best_weights = os.path.join(ckpt.best_dir(checkpoint_dir), ckpt.MODEL_WEIGHTS_FILENAME)
        self.assertNotEqual(ckpt.sha256_file(last_weights), ckpt.sha256_file(best_weights))

        last_state = ckpt.read_state(last_dir)
        self.assertEqual(last_state.completed_epoch, 3)
        self.assertEqual(last_state.best_epoch, 0)

    def test_last_checkpoint_holds_the_trajectory_not_the_restored_best_weights(self):
        """EarlyStopping(restore_best_weights=True) rewinds the in-memory model at
        `on_train_end`. LAST is written at `on_epoch_end`, so it must still hold the
        final epoch's weights -- otherwise the next leg would resume from a rewound
        trajectory with a mismatched optimizer state."""
        run_dir = os.path.join(self.tmp, "run")
        self._run_leg(run_dir, epochs=3, resume=False, scripted=[0.90, 0.20, 0.10])
        checkpoint_dir = os.path.join(run_dir, "checkpoints")
        last_dir = ckpt.find_resumable_generation(checkpoint_dir)

        probe = build_tiny_model()
        ckpt.restore_training_state(probe, last_dir)
        best_probe = build_tiny_model()
        best_probe.load_weights(os.path.join(ckpt.best_dir(checkpoint_dir),
                                             ckpt.MODEL_WEIGHTS_FILENAME))
        self.assertNotEqual(weight_signature(probe), weight_signature(best_probe))

    def test_early_stopping_wait_counter_accumulates_across_resumes(self):
        run_dir = os.path.join(self.tmp, "run")
        self._run_leg(run_dir, epochs=2, resume=False, scripted=[0.90, 0.50])
        checkpoint_dir = os.path.join(run_dir, "checkpoints")
        state = ckpt.read_state(ckpt.find_resumable_generation(checkpoint_dir))
        self.assertEqual(state.early_stopping["wait"], 1)
        self.assertAlmostEqual(state.early_stopping["best"], 0.90, places=6)

        self._run_leg(run_dir, epochs=4, resume=True, scripted=[0, 0, 0.40, 0.30])
        state = ckpt.read_state(ckpt.find_resumable_generation(checkpoint_dir))
        self.assertEqual(state.early_stopping["wait"], 3,
                         "EarlyStopping.wait reset at the process/leg boundary")
        self.assertAlmostEqual(state.early_stopping["best"], 0.90, places=6)
        self.assertIn("stopped_epoch", state.early_stopping)

    def test_reduce_lr_state_and_effective_learning_rate_persist(self):
        run_dir = os.path.join(self.tmp, "run")
        # patience=2: epochs 1,2 are non-improving -> LR halves at epoch 2.
        self._run_leg(run_dir, epochs=3, resume=False,
                      scripted=[0.90, 0.50, 0.40], reduce_lr_patience=2)
        checkpoint_dir = os.path.join(run_dir, "checkpoints")
        state = ckpt.read_state(ckpt.find_resumable_generation(checkpoint_dir))
        self.assertAlmostEqual(state.learning_rate, 0.005, places=6)
        self.assertAlmostEqual(state.reduce_lr["best"], 0.90, places=6)
        self.assertIn("cooldown_counter", state.reduce_lr)

        # A resumed leg must start from the REDUCED learning rate, not the original 0.01.
        resumed = build_tiny_model()
        self._run_leg(run_dir, epochs=4, resume=True, scripted=[0, 0, 0, 0.35],
                      model=resumed, reduce_lr_patience=2)
        self.assertAlmostEqual(
            float(tf.keras.backend.get_value(resumed.optimizer.learning_rate)), 0.005, places=6)

    def test_completed_epoch_drives_initial_epoch_on_resume(self):
        run_dir = os.path.join(self.tmp, "run")
        self._run_leg(run_dir, epochs=2, resume=False, scripted=[0.10, 0.20])
        config = TrainingConfig(
            run_dir=run_dir, epochs=4, monitor="val_QWK", mode="max", resume=True,
            mixed_precision=False,
            checkpoint_options=CheckpointOptions(experiment_id="exp-1", config_hash="cfg-abc"),
        )
        trainer = Trainer(config)
        trainer.prepare(build_tiny_model())
        self.assertEqual(trainer.resolve_initial_epoch(), 2)

    def test_resume_with_a_different_configuration_hash_is_refused(self):
        run_dir = os.path.join(self.tmp, "run")
        self._run_leg(run_dir, epochs=2, resume=False, scripted=[0.10, 0.20])
        with self.assertRaises(ckpt.CheckpointCompatibilityError):
            self._run_leg(run_dir, epochs=4, resume=True, scripted=[0, 0, 0.3, 0.4],
                          config_hash="cfg-CHANGED")


# ===========================================================================
# TrainingStateCheckpoint wiring
# ===========================================================================

class TrainingStateCheckpointCallbackTests(TempDirTestCase):
    def test_callback_is_registered_by_build_callbacks_when_options_are_supplied(self):
        from training.callbacks import build_callbacks
        callbacks, paths = build_callbacks(
            checkpoint_dir=self.checkpoint_dir,
            log_dir=os.path.join(self.tmp, "logs"),
            monitor="val_QWK", mode="max",
            checkpoint_options=CheckpointOptions(experiment_id="exp-1"),
        )
        kinds = [type(c).__name__ for c in callbacks]
        self.assertIn("TrainingStateCheckpoint", kinds)
        # It must come AFTER EarlyStopping/ReduceLROnPlateau, whose on_train_begin
        # resets the very counters this callback restores.
        self.assertGreater(kinds.index("TrainingStateCheckpoint"), kinds.index("EarlyStopping"))
        self.assertGreater(kinds.index("TrainingStateCheckpoint"),
                           kinds.index("ReduceLROnPlateau"))
        self.assertIn("latest_json", paths)
        self.assertIn("best_dir", paths)

    def test_default_build_callbacks_is_unchanged_without_options(self):
        from training.callbacks import build_callbacks
        callbacks, paths = build_callbacks(
            checkpoint_dir=self.checkpoint_dir, log_dir=os.path.join(self.tmp, "logs"),
        )
        kinds = [type(c).__name__ for c in callbacks]
        self.assertNotIn("TrainingStateCheckpoint", kinds)
        self.assertEqual(kinds.count("ModelCheckpoint"), 2)
        self.assertIn("best_weights", paths)
        self.assertIn("last_weights", paths)
        self.assertIn("epoch_state", paths)

    def test_a_run_that_is_not_resuming_does_not_adopt_a_previous_global_best(self):
        """A fresh run pointed at a directory that already holds generations must
        start from epoch 0 with fresh weights AND a fresh best -- inheriting the
        predecessor's best would claim a score this run never achieved. Nothing
        already on disk is overwritten either: generation numbering continues."""
        model = train_a_few_steps(build_tiny_model(), steps=1)
        self.save_one(model, best_metric=0.91, best_epoch=0)

        callback = TrainingStateCheckpoint(
            checkpoint_dir=self.checkpoint_dir, monitor="val_QWK", mode="max",
            options=CheckpointOptions(experiment_id="exp-1", verbose=0),
            restore_state=False,
        )
        callback.set_model(model)
        callback.on_train_begin()
        self.assertIsNone(callback.best_metric)

        callback.on_epoch_end(0, {"val_QWK": 0.10})
        numbers = [n for n, _ in ckpt.list_generations(self.checkpoint_dir)]
        self.assertEqual(numbers, [1, 2], "the earlier generation must survive untouched")
        self.assertAlmostEqual(ckpt.read_state(
            os.path.join(self.checkpoint_dir, "gen_00002")).best_metric, 0.10, places=6)
        self.assertAlmostEqual(ckpt.read_state(
            os.path.join(self.checkpoint_dir, "gen_00001")).best_metric, 0.91, places=6)

    def test_a_resuming_run_does_adopt_the_previous_global_best(self):
        model = train_a_few_steps(build_tiny_model(), steps=1)
        self.save_one(model, best_metric=0.91, best_epoch=0)

        callback = TrainingStateCheckpoint(
            checkpoint_dir=self.checkpoint_dir, monitor="val_QWK", mode="max",
            options=CheckpointOptions(experiment_id="exp-1", verbose=0),
            restore_state=True,
        )
        callback.set_model(model)
        callback.on_train_begin()
        self.assertAlmostEqual(callback.best_metric, 0.91, places=6)

    def test_trainer_only_restores_state_when_resume_is_set(self):
        from training.callbacks import build_callbacks
        for resume, expected in ((True, True), (False, False)):
            callbacks, _ = build_callbacks(
                checkpoint_dir=self.checkpoint_dir,
                log_dir=os.path.join(self.tmp, "logs"),
                monitor="val_QWK", mode="max",
                checkpoint_options=CheckpointOptions(experiment_id="exp-1"),
                restore_checkpoint_state=resume,
            )
            state_callback = [c for c in callbacks if isinstance(c, TrainingStateCheckpoint)][0]
            self.assertEqual(state_callback.restore_state, expected)

    def test_trainer_restore_without_checkpoint_options_fails_clearly(self):
        config = TrainingConfig(run_dir=self.tmp, epochs=1, mixed_precision=False)
        trainer = Trainer(config)
        trainer.prepare()
        with self.assertRaises(ckpt.CheckpointError) as caught:
            trainer.restore(build_tiny_model())
        self.assertIn("CheckpointOptions", str(caught.exception))

    def test_callback_records_monitor_name_and_mode_in_state(self):
        model = train_a_few_steps(build_tiny_model(), steps=1)
        callback = TrainingStateCheckpoint(
            checkpoint_dir=self.checkpoint_dir, monitor="val_QWK", mode="max",
            options=CheckpointOptions(experiment_id="exp-1", config_hash="cfg-abc"),
        )
        callback.set_model(model)
        callback.on_train_begin()
        callback.on_epoch_end(0, {"val_QWK": 0.42, "loss": 1.0})
        state = ckpt.read_state(ckpt.find_resumable_generation(self.checkpoint_dir))
        self.assertEqual(state.monitor, "val_QWK")
        self.assertEqual(state.monitor_mode, "max")
        self.assertAlmostEqual(state.best_metric, 0.42, places=6)
        self.assertEqual(state.completed_epoch, 1)


# ===========================================================================
# 9, 17, 19 -- fresh-process resume and the three-process relay
# ===========================================================================

class ThreeProcessRelayTests(unittest.TestCase):
    """A -> B -> C, each a separate OS process, resuming the same experiment."""

    WORKER = os.path.join(REPO_ROOT, "tests", "checkpoint_relay_worker.py")

    def setUp(self):
        self.tmp = tempfile.mkdtemp(prefix="relay_test_")
        self.addCleanup(shutil.rmtree, self.tmp, True)
        self.run_dir = os.path.join(self.tmp, "run")

    def run_leg(self, leg, target_epochs, resume, scripted):
        env = dict(os.environ)
        env["PYTHONPATH"] = REPO_ROOT + os.pathsep + env.get("PYTHONPATH", "")
        completed = subprocess.run(
            [sys.executable, self.WORKER, self.run_dir, leg, str(target_epochs),
             "1" if resume else "0", ",".join(str(v) for v in scripted)],
            capture_output=True, text=True, env=env, cwd=REPO_ROOT, timeout=900,
        )
        marker = [line for line in completed.stdout.splitlines() if line.startswith("RELAY_RESULT ")]
        self.assertTrue(marker, f"leg {leg} produced no result.\n"
                                f"STDOUT:\n{completed.stdout[-4000:]}\n"
                                f"STDERR:\n{completed.stderr[-4000:]}")
        return json.loads(marker[-1][len("RELAY_RESULT "):])

    def test_three_process_relay_continues_optimizer_and_global_best(self):
        a = self.run_leg("A", 2, False, [0.60, 0.72])
        self.assertEqual(a["initial_epoch"], 0)
        self.assertIsNone(a["state_before"])
        self.assertEqual(a["state_after"]["completed_epoch"], 2)
        self.assertAlmostEqual(a["state_after"]["best_metric"], 0.72, places=6)
        self.assertGreater(a["optimizer_iterations_after"], 0)

        # Leg B scores WORSE at both epochs -- the global best must survive.
        b = self.run_leg("B", 4, True, [0, 0, 0.65, 0.61])
        self.assertEqual(b["initial_epoch"], 2)
        self.assertEqual(b["state_before"]["completed_epoch"], 2)
        self.assertEqual(b["state_before"]["optimizer_iterations"],
                         a["state_after"]["optimizer_iterations"])
        self.assertAlmostEqual(b["state_before"]["best_metric"], 0.72, places=6)
        self.assertAlmostEqual(b["state_after"]["best_metric"], 0.72, places=6)
        self.assertEqual(b["state_after"]["best_epoch"], 1)
        self.assertEqual(b["state_after"]["completed_epoch"], 4)
        self.assertGreater(b["state_after"]["optimizer_iterations"],
                           a["state_after"]["optimizer_iterations"])

        # Leg C beats it at epoch 5.
        c = self.run_leg("C", 6, True, [0, 0, 0, 0, 0.70, 0.80])
        self.assertEqual(c["initial_epoch"], 4)
        self.assertAlmostEqual(c["state_before"]["best_metric"], 0.72, places=6)
        self.assertAlmostEqual(c["state_after"]["best_metric"], 0.80, places=6)
        self.assertEqual(c["state_after"]["best_epoch"], 5)
        self.assertEqual(c["state_after"]["completed_epoch"], 6)
        self.assertGreater(c["state_after"]["optimizer_iterations"],
                           b["state_after"]["optimizer_iterations"])
        self.assertEqual(c["best_checkpoint"]["best_epoch"], 5)

        # EarlyStopping's wait counter accumulated across BOTH process boundaries:
        # epochs 2,3 were worse than 0.72 -> wait 1,2; epoch 4 worse -> 3; epoch 5 improved -> 0.
        self.assertEqual(b["state_after"]["early_stopping_wait"], 2)
        self.assertEqual(c["state_before"]["early_stopping_wait"], 2)
        self.assertEqual(c["state_after"]["early_stopping_wait"], 0)

        # Only the latest two generations are retained.
        self.assertEqual(len(c["generations_on_disk"]), 2)
        self.assertTrue(c["best_dir_exists"])

    def test_fresh_process_resume_reports_the_reduced_learning_rate(self):
        """reduce_lr_patience=2 in the worker: epochs 1 and 2 are non-improving, so the
        LR halves during leg A and leg B must START from the halved value."""
        a = self.run_leg("A", 3, False, [0.90, 0.50, 0.40])
        self.assertAlmostEqual(a["learning_rate_after"], 0.005, places=6)
        self.assertAlmostEqual(a["state_after"]["learning_rate"], 0.005, places=6)

        b = self.run_leg("B", 4, True, [0, 0, 0, 0.30])
        self.assertAlmostEqual(b["state_before"]["learning_rate"], 0.005, places=6)
        self.assertLessEqual(b["learning_rate_after"], 0.005 + 1e-9)


if __name__ == "__main__":
    unittest.main()
