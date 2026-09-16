"""Fast, CPU-only unit tests for multiseed_runs.py.

Uses tiny synthetic models and temp directories throughout -- never the real 43M-parameter joint
model (that is exercised in tests/test_no_racaf_model.py) and never real images/caches.
"""
import json
import os
import shutil
import tempfile
import time
import unittest
from unittest import mock

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "-1")

import numpy as np
import tensorflow as tf

import corn
import multiseed_runs as msr
from training import checkpointing as ckpt


def _tiny_model():
    model = tf.keras.Sequential([tf.keras.layers.Dense(2, input_shape=(3,))])
    model.compile(optimizer=tf.keras.optimizers.Adam(0.01), loss="mse")
    return model


class TempDirTestCase(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.mkdtemp(prefix="msr_test_")

    def tearDown(self):
        shutil.rmtree(self.tmp, ignore_errors=True)


class DirectoryLayoutTests(TempDirTestCase):
    def test_run_dir_layout(self):
        path = msr.run_dir(self.tmp, "exp1", "RACAF", 42)
        self.assertEqual(path, os.path.join(self.tmp, "exp1", "RACAF", "seed_42"))

    def test_rejects_unknown_arm(self):
        with self.assertRaises(ValueError):
            msr.run_dir(self.tmp, "exp1", "BOTH", 42)

    def test_rejects_unknown_seed(self):
        with self.assertRaises(ValueError):
            msr.run_dir(self.tmp, "exp1", "RACAF", 999)

    def test_all_run_ids_is_six_matched_pairs(self):
        self.assertEqual(msr.all_run_ids(), [("RACAF", 42), ("NO_RACAF", 42), ("RACAF", 123),
                                             ("NO_RACAF", 123), ("RACAF", 2026), ("NO_RACAF", 2026)])
        self.assertEqual(msr.SPLIT_SEED, 42)
        self.assertEqual((msr.EXPECTED_TRAIN_COUNT, msr.EXPECTED_VAL_COUNT), (2929, 733))

    def test_ensure_run_dir_is_idempotent(self):
        path = msr.run_dir(self.tmp, "exp1", "RACAF", 42)
        msr.ensure_run_dir(path)
        marker = os.path.join(path, "checkpoints", "marker.txt")
        with open(marker, "w") as handle:
            handle.write("keep me")
        msr.ensure_run_dir(path)  # must not delete existing content
        self.assertTrue(os.path.exists(marker))


class RunConfigHashTests(unittest.TestCase):
    def test_hash_is_stable(self):
        a = msr.run_config_hash("exp1", "RACAF", 42, "deadbeef")
        b = msr.run_config_hash("exp1", "RACAF", 42, "deadbeef")
        self.assertEqual(a, b)

    def test_hash_differs_by_arm(self):
        a = msr.run_config_hash("exp1", "RACAF", 42, "deadbeef")
        b = msr.run_config_hash("exp1", "NO_RACAF", 42, "deadbeef")
        self.assertNotEqual(a, b)

    def test_hash_differs_by_seed(self):
        a = msr.run_config_hash("exp1", "RACAF", 42, "deadbeef")
        b = msr.run_config_hash("exp1", "RACAF", 123, "deadbeef")
        self.assertNotEqual(a, b)

    def test_hash_differs_by_experiment_id(self):
        a = msr.run_config_hash("exp1", "RACAF", 42, "deadbeef")
        b = msr.run_config_hash("exp2", "RACAF", 42, "deadbeef")
        self.assertNotEqual(a, b)

    def test_config_mapping_never_contains_a_run_seed_named_split_seed_field_confusion(self):
        mapping = msr.run_config_mapping("exp1", "RACAF", 123, "deadbeef")
        self.assertEqual(mapping["run_seed"], 123)
        self.assertEqual(mapping["split_seed"], msr.SPLIT_SEED)
        self.assertNotEqual(mapping["split_seed"], 123)  # split seed is NEVER the run seed


class RunManifestTests(TempDirTestCase):
    def test_write_then_verify_round_trips(self):
        path = os.path.join(self.tmp, "run")
        os.makedirs(path)
        mapping = msr.run_config_mapping("exp1", "RACAF", 42, "deadbeef")
        config_hash = ckpt.config_hash(mapping)
        msr.write_run_manifest(os.path.join(path, msr.RUN_MANIFEST_FILENAME), mapping, config_hash)
        verified = msr.verify_run_manifest(path, mapping, config_hash)
        self.assertEqual(verified["arm"], "RACAF")

    def test_refuses_to_overwrite_an_existing_manifest(self):
        path = os.path.join(self.tmp, "run")
        os.makedirs(path)
        mapping = msr.run_config_mapping("exp1", "RACAF", 42, "deadbeef")
        manifest_path = os.path.join(path, msr.RUN_MANIFEST_FILENAME)
        msr.write_run_manifest(manifest_path, mapping, "hash1")
        with self.assertRaises(msr.RunConfigurationError):
            msr.write_run_manifest(manifest_path, mapping, "hash2")

    def test_verify_rejects_hash_mismatch(self):
        path = os.path.join(self.tmp, "run")
        os.makedirs(path)
        mapping = msr.run_config_mapping("exp1", "RACAF", 42, "deadbeef")
        msr.write_run_manifest(os.path.join(path, msr.RUN_MANIFEST_FILENAME), mapping, "hash1")
        with self.assertRaises(msr.RunConfigurationError):
            msr.verify_run_manifest(path, mapping, "hash2")

    def test_verify_rejects_seed_mismatch(self):
        path = os.path.join(self.tmp, "run")
        os.makedirs(path)
        mapping = msr.run_config_mapping("exp1", "RACAF", 42, "deadbeef")
        config_hash = ckpt.config_hash(mapping)
        msr.write_run_manifest(os.path.join(path, msr.RUN_MANIFEST_FILENAME), mapping, config_hash)
        other_mapping = msr.run_config_mapping("exp1", "RACAF", 123, "deadbeef")
        with self.assertRaises(msr.RunConfigurationError):
            msr.verify_run_manifest(path, other_mapping, config_hash)

    def test_initialize_run_is_idempotent(self):
        path1, hash1 = msr.initialize_run(self.tmp, "exp1", "RACAF", 42, "deadbeef")
        path2, hash2 = msr.initialize_run(self.tmp, "exp1", "RACAF", 42, "deadbeef")
        self.assertEqual(path1, path2)
        self.assertEqual(hash1, hash2)

    def test_initialize_run_refuses_a_changed_configuration(self):
        msr.initialize_run(self.tmp, "exp1", "RACAF", 42, "deadbeef")
        with self.assertRaises(msr.RunConfigurationError):
            msr.initialize_run(self.tmp, "exp1", "RACAF", 42, "different-split-hash")

    def test_initialize_run_accepts_a_changed_git_commit_and_keeps_the_original_as_provenance(self):
        with mock.patch.object(ckpt, "environment_fingerprint", return_value={"git_commit_hash": "aaa"}):
            msr.initialize_run(self.tmp, "exp1", "RACAF", 42, "deadbeef")
        with mock.patch.object(ckpt, "environment_fingerprint", return_value={"git_commit_hash": "bbb"}):
            path, _hash = msr.initialize_run(self.tmp, "exp1", "RACAF", 42, "deadbeef")
        self.assertEqual(msr.read_run_manifest(path)["environment"]["git_commit_hash"], "aaa")


class PreregistrationTests(TempDirTestCase):
    def test_write_then_load_round_trips(self):
        path = os.path.join(self.tmp, "PREREGISTRATION.json")
        written, sha_written = msr.write_preregistration(path, "exp1")
        loaded, sha_loaded = msr.load_and_verify_preregistration(path)
        self.assertEqual(written, loaded)
        self.assertEqual(sha_written, sha_loaded)

    def test_refuses_to_overwrite(self):
        path = os.path.join(self.tmp, "PREREGISTRATION.json")
        msr.write_preregistration(path, "exp1")
        with self.assertRaises(msr.RunConfigurationError):
            msr.write_preregistration(path, "exp1")

    def test_detects_hand_editing(self):
        path = os.path.join(self.tmp, "PREREGISTRATION.json")
        msr.write_preregistration(path, "exp1")
        with open(path) as handle:
            data = json.load(handle)
        data["max_epochs"] = 999
        with open(path, "w") as handle:
            json.dump(data, handle)  # NOT the canonical indent=2, sort_keys=True form
        with self.assertRaises(msr.RunConfigurationError):
            msr.load_and_verify_preregistration(path)

    def test_delta_definition_is_racaf_minus_no_racaf(self):
        prereg = msr.build_preregistration("exp1")
        self.assertIn("QWK_RACAF - QWK_NO_RACAF", prereg["delta_definition"])

    def test_idrid_is_explicitly_out_of_scope(self):
        prereg = msr.build_preregistration("exp1")
        self.assertIn("out of scope", prereg["idrid_external_evaluation"])


class LockTests(TempDirTestCase):
    def test_acquire_then_release(self):
        msr.acquire_lock(self.tmp, owner_id="me")
        self.assertTrue(os.path.exists(msr._lock_path(self.tmp)))
        msr.release_lock(self.tmp, owner_id="me")
        self.assertFalse(os.path.exists(msr._lock_path(self.tmp)))

    def test_same_owner_can_reacquire(self):
        msr.acquire_lock(self.tmp, owner_id="me")
        msr.acquire_lock(self.tmp, owner_id="me")  # must not raise

    def test_different_owner_is_refused_while_fresh(self):
        msr.acquire_lock(self.tmp, owner_id="runtime-a")
        with self.assertRaises(msr.RunLockedError):
            msr.acquire_lock(self.tmp, owner_id="runtime-b", ttl_seconds=3600)

    def test_different_owner_allowed_after_stale_ttl(self):
        msr.acquire_lock(self.tmp, owner_id="runtime-a")
        msr.acquire_lock(self.tmp, owner_id="runtime-b", ttl_seconds=0)  # instantly stale

    def test_force_overrides_a_fresh_lock(self):
        msr.acquire_lock(self.tmp, owner_id="runtime-a")
        msr.acquire_lock(self.tmp, owner_id="runtime-b", ttl_seconds=3600, force=True)

    def test_heartbeat_refreshes_the_same_owner(self):
        msr.acquire_lock(self.tmp, owner_id="me")
        before = json.load(open(msr._lock_path(self.tmp)))["heartbeat"]
        time.sleep(0.01)
        msr.heartbeat_lock(self.tmp, owner_id="me")
        after = json.load(open(msr._lock_path(self.tmp)))["heartbeat"]
        self.assertGreater(after, before)

    def test_heartbeat_raises_if_taken_over(self):
        msr.acquire_lock(self.tmp, owner_id="runtime-a")
        msr.acquire_lock(self.tmp, owner_id="runtime-b", ttl_seconds=0, force=False)  # stale takeover
        with self.assertRaises(msr.RunLockedError):
            msr.heartbeat_lock(self.tmp, owner_id="runtime-a")

    def test_release_by_non_owner_does_nothing(self):
        msr.acquire_lock(self.tmp, owner_id="runtime-a")
        msr.release_lock(self.tmp, owner_id="runtime-b")
        self.assertTrue(os.path.exists(msr._lock_path(self.tmp)))


class StopDecisionTests(TempDirTestCase):
    def test_none_when_absent(self):
        self.assertIsNone(msr.read_stop_decision(self.tmp))

    def test_write_then_read(self):
        msr.write_stop_decision(self.tmp, 18, "early_stopping")
        decision = msr.read_stop_decision(self.tmp)
        self.assertEqual(decision["epoch"], 18)
        self.assertEqual(decision["stop_reason"], "early_stopping")
        self.assertTrue(decision["stop_decided"])

    def test_rejects_unknown_reason(self):
        with self.assertRaises(ValueError):
            msr.write_stop_decision(self.tmp, 18, "because_i_felt_like_it")


class EpochHistoryTests(TempDirTestCase):
    def _state(self, epoch):
        return ckpt.TrainingState(
            experiment_id="exp1", completed_epoch=epoch, best_epoch=epoch, best_metric=0.5,
            monitor="val_QWK", monitor_mode="max", learning_rate=1e-4,
            extra={"epoch_logs": {"val_QWK": 0.5, "val_loss": 0.3}},
        )

    def test_write_then_read_history(self):
        msr.write_epoch_history(self.tmp, self._state(1))
        msr.write_epoch_history(self.tmp, self._state(2))
        rows = msr.read_history(self.tmp)
        self.assertEqual([r["epoch"] for r in rows], [1, 2])
        self.assertEqual(rows[0]["val_QWK"], 0.5)

    def test_rewriting_the_same_epoch_does_not_duplicate(self):
        msr.write_epoch_history(self.tmp, self._state(1))
        msr.write_epoch_history(self.tmp, self._state(1))
        rows = msr.read_history(self.tmp)
        self.assertEqual(len(rows), 1)


class TwoSlotBestTests(TempDirTestCase):
    def _generation(self, epoch, qwk):
        model = _tiny_model()
        state = ckpt.TrainingState(
            experiment_id="exp1", completed_epoch=epoch, best_epoch=epoch, best_metric=qwk,
            monitor="val_QWK", monitor_mode="max",
        )
        checkpoint_dir = os.path.join(self.tmp, "checkpoints")
        return ckpt.save_generation(checkpoint_dir, model, state,
                                    staging_dir=os.path.join(self.tmp, "staging")), state

    def test_publish_then_read_best(self):
        generation_dir, state = self._generation(1, 0.5)
        slot_dir = msr.publish_best(self.tmp, generation_dir, state)
        read_dir, pointer = msr.read_best(self.tmp)
        self.assertEqual(read_dir, slot_dir)
        self.assertEqual(pointer["val_QWK"], 0.5)

    def test_none_before_any_publish(self):
        slot_dir, pointer = msr.read_best(self.tmp)
        self.assertIsNone(slot_dir)
        self.assertIsNone(pointer)

    def test_second_publish_uses_the_other_slot_and_first_slot_survives_until_flip(self):
        gen1, state1 = self._generation(1, 0.5)
        slot1 = msr.publish_best(self.tmp, gen1, state1)
        self.assertTrue(os.path.basename(slot1) in ("best_a", "best_b"))

        gen2, state2 = self._generation(2, 0.6)
        slot2 = msr.publish_best(self.tmp, gen2, state2)
        self.assertNotEqual(slot1, slot2)
        # the FIRST slot must still physically exist and validate right up until best.json flips
        # -- publish_best only ever deletes the currently-INACTIVE slot, never the active one.
        self.assertTrue(os.path.isdir(slot1))

        read_dir, pointer = msr.read_best(self.tmp)
        self.assertEqual(read_dir, slot2)
        self.assertEqual(pointer["val_QWK"], 0.6)

    def test_a_corrupted_best_pointer_target_raises_not_none(self):
        generation_dir, state = self._generation(1, 0.5)
        slot_dir = msr.publish_best(self.tmp, generation_dir, state)
        os.remove(os.path.join(slot_dir, ckpt.MODEL_WEIGHTS_FILENAME))
        with self.assertRaises(ckpt.CheckpointIntegrityError):
            msr.read_best(self.tmp)

    def test_three_publishes_alternate_slots(self):
        slots = []
        for epoch, qwk in ((1, 0.4), (2, 0.5), (3, 0.6)):
            generation_dir, state = self._generation(epoch, qwk)
            slots.append(os.path.basename(msr.publish_best(self.tmp, generation_dir, state)))
        self.assertEqual(slots, ["best_a", "best_b", "best_a"])


class StatusTests(TempDirTestCase):
    def test_not_started_when_no_manifest(self):
        self.assertEqual(msr.run_status(self.tmp), msr.STATUS_NOT_STARTED)

    def test_created_after_manifest_before_any_epoch(self):
        mapping = msr.run_config_mapping("exp1", "RACAF", 42, "deadbeef")
        msr.write_run_manifest(os.path.join(self.tmp, msr.RUN_MANIFEST_FILENAME), mapping, "h")
        self.assertEqual(msr.run_status(self.tmp), msr.STATUS_CREATED)

    def test_completed_when_stop_decision_present(self):
        mapping = msr.run_config_mapping("exp1", "RACAF", 42, "deadbeef")
        msr.write_run_manifest(os.path.join(self.tmp, msr.RUN_MANIFEST_FILENAME), mapping, "h")
        msr.write_stop_decision(self.tmp, 30, "early_stopping")
        self.assertEqual(msr.run_status(self.tmp), msr.STATUS_COMPLETED)

    def test_running_when_a_fresh_lock_and_a_valid_generation_exist(self):
        mapping = msr.run_config_mapping("exp1", "RACAF", 42, "deadbeef")
        msr.write_run_manifest(os.path.join(self.tmp, msr.RUN_MANIFEST_FILENAME), mapping, "h")
        model = _tiny_model()
        state = ckpt.TrainingState(experiment_id="exp1", completed_epoch=1, monitor="val_QWK", monitor_mode="max")
        ckpt.save_generation(os.path.join(self.tmp, "checkpoints"), model, state,
                             staging_dir=os.path.join(self.tmp, "staging"))
        msr.acquire_lock(self.tmp, owner_id="me")
        self.assertEqual(msr.run_status(self.tmp), msr.STATUS_RUNNING)

    def test_interrupted_when_generation_valid_but_lock_stale(self):
        mapping = msr.run_config_mapping("exp1", "RACAF", 42, "deadbeef")
        msr.write_run_manifest(os.path.join(self.tmp, msr.RUN_MANIFEST_FILENAME), mapping, "h")
        model = _tiny_model()
        state = ckpt.TrainingState(experiment_id="exp1", completed_epoch=1, monitor="val_QWK", monitor_mode="max")
        ckpt.save_generation(os.path.join(self.tmp, "checkpoints"), model, state,
                             staging_dir=os.path.join(self.tmp, "staging"))
        self.assertEqual(msr.run_status(self.tmp, lock_ttl_seconds=1800), msr.STATUS_INTERRUPTED)


def _tiny_arm_model():
    """A trivial 3-input, CORN-shaped (5-grade, 4-threshold) model -- stands in for
    build_arm_model()'s real ~43M-parameter output so train_run()'s ORCHESTRATION (not the real
    architecture, already covered by tests/test_no_racaf_model.py) can be exercised fast, on CPU,
    with no cache/images/GPU models at all."""
    import corn as corn_module
    import weighted_corn as wc

    stage5 = tf.keras.Input(shape=(3,), name="stage5_input")
    stage6 = tf.keras.Input(shape=(2,), name="stage6_input")
    reliability = tf.keras.Input(shape=(1,), name="reliability")
    merged = tf.keras.layers.Concatenate()([stage5, stage6, reliability])
    logits = tf.keras.layers.Dense(corn_module.NUM_THRESHOLDS, name="corn_logits")(merged)
    model = tf.keras.Model([stage5, stage6, reliability], logits, name="tiny_arm_model")
    model.compile(optimizer=tf.keras.optimizers.Adam(1e-2),
                 loss=wc.make_weighted_corn_loss([1.0] * corn_module.NUM_GRADES),
                 metrics=[corn_module.CORNQuadraticWeightedKappa()])
    return model


def _tiny_epoch_dataset(rng_seed):
    rng = np.random.default_rng(rng_seed)
    n = 6
    stage5 = rng.normal(size=(n, 3)).astype(np.float32)
    stage6 = rng.normal(size=(n, 2)).astype(np.float32)
    reliability = rng.uniform(size=(n, 1)).astype(np.float32)
    grades = rng.integers(0, 5, size=n).astype(np.int32)
    return tf.data.Dataset.from_tensor_slices(((stage5, stage6, reliability), grades)).batch(2)


class TrainRunResumeIntegrationTests(TempDirTestCase):
    """The single most important property this module exists for: a run trained partway in one
    `train_run()` call, then handed a FRESH model instance (simulating a brand-new process/
    runtime rebuilding it) and resumed via a second `train_run()` call, continues from the
    correct epoch -- never epoch 0 -- with no duplicated or skipped epoch in its history."""

    def setUp(self):
        super().setUp()
        self.run_dir = os.path.join(self.tmp, "run")
        self.staging_dir = os.path.join(self.tmp, "staging")
        self._patcher = mock.patch(
            "improved_training_data.make_epoch_dataset",
            side_effect=lambda entries, epoch, run_seed, **kw: _tiny_epoch_dataset(
                run_seed * 1000 + epoch),
        )
        self._patcher.start()
        self.addCleanup(self._patcher.stop)

        # The git commit and one extra fingerprint component are controlled per test; the rest of
        # the fingerprint is the REAL one, computed from this repository's real sources.
        self.commit = "commitA"
        self.behavior_change = None
        real_fingerprint = msr.training_behavior_fingerprint

        def fingerprint(*args, **kwargs):
            result = real_fingerprint(*args, **kwargs)
            if self.behavior_change is None:
                return result
            components = dict(result["components"], test_behavior_change=self.behavior_change)
            return {"training_behavior_hash": msr._canonical_hash(components), "components": components}

        for name, replacement in (("_current_git_commit", lambda repo_dir=None: self.commit),
                                  ("training_behavior_fingerprint", fingerprint)):
            patcher = mock.patch.object(msr, name, side_effect=replacement)
            patcher.start()
            self.addCleanup(patcher.stop)

    def _train_run(self, model, **overrides):
        kwargs = dict(
            model=model, run_dir_path=self.run_dir, arm="RACAF", run_seed=42,
            train_entries=[("a", 0), ("b", 1)], val_entries=[("c", 2)],
            cache_dir="x", racaf_cache_dir="x", config_hash_value="testhash", batch_size=2, max_epochs=5,
            early_stopping_patience=100, reduce_lr_patience=100, staging_dir=self.staging_dir,
            precision_check="off", mixed_precision=False, verbose=0,
        )
        kwargs.update(overrides)
        return msr.train_run(**kwargs)

    def test_resumes_from_the_correct_epoch_with_a_brand_new_model_instance(self):
        first_model = _tiny_arm_model()
        first_outcome = self._train_run(first_model, session_epoch_budget=2)
        self.assertFalse(first_outcome.stopped)
        self.assertEqual(first_outcome.completed_epoch, 2)
        self.assertEqual(first_outcome.epochs_trained_this_call, 2)
        self.assertEqual([h["epoch"] for h in msr.read_history(self.run_dir)], [1, 2])

        # A FRESH model instance -- a different Python object, freshly initialized weights,
        # freshly built optimizer -- simulating a brand-new process/runtime that rebuilds the
        # model from scratch before resuming. train_run() must restore weights+optimizer from
        # the checkpoint and continue at epoch 2, never retrain epoch 1 or 2.
        second_model = _tiny_arm_model()
        second_outcome = self._train_run(second_model)
        self.assertTrue(second_outcome.stopped)
        self.assertEqual(second_outcome.stop_reason, "epoch_cap")
        self.assertEqual(second_outcome.completed_epoch, 5)
        self.assertEqual(second_outcome.epochs_trained_this_call, 3)   # epochs 3, 4, 5 only

        history = msr.read_history(self.run_dir)
        self.assertEqual([h["epoch"] for h in history], [1, 2, 3, 4, 5])   # no duplicate, no gap

        # A third call after the run is already COMPLETED must train nothing further at all.
        third_model = _tiny_arm_model()
        third_outcome = self._train_run(third_model)
        self.assertTrue(third_outcome.stopped)
        self.assertEqual(third_outcome.epochs_trained_this_call, 0)
        self.assertEqual(len(msr.read_history(self.run_dir)), 5)

    def test_a_completed_run_stop_decision_survives_and_blocks_further_training(self):
        model = _tiny_arm_model()
        self._train_run(model, max_epochs=2)   # trains straight to the cap in one call
        self.assertIsNotNone(msr.read_stop_decision(self.run_dir))
        history_before = msr.read_history(self.run_dir)

        model2 = _tiny_arm_model()
        outcome = self._train_run(model2, max_epochs=2)
        self.assertEqual(outcome.epochs_trained_this_call, 0)
        self.assertEqual(msr.read_history(self.run_dir), history_before)

    def test_best_is_published_during_a_resumed_run(self):
        model = _tiny_arm_model()
        self._train_run(model, session_epoch_budget=1)
        slot_dir, pointer = msr.read_best(self.run_dir)
        self.assertIsNotNone(slot_dir)   # the first epoch always "improves" over no BEST at all
        self.assertEqual(pointer["epoch"], 0)   # 0-indexed epoch, matching TrainingState.best_epoch

    # --- training-behaviour fingerprint vs git commit -------------------------------------------

    def _superseded(self):
        return msr._superseded_dirs(self.run_dir)

    @staticmethod
    def _iterations(model):
        return int(model.optimizer.iterations.numpy())

    def test_git_commit_change_alone_resumes_normally(self):
        self._train_run(_tiny_arm_model(), session_epoch_budget=2)
        self.commit = "commitB"   # a normal non-training commit + a brand-new runtime/model
        outcome = self._train_run(_tiny_arm_model(), session_epoch_budget=1)

        self.assertEqual(outcome.training_behavior["action"], "resume")
        self.assertEqual(outcome.completed_epoch, 3)
        self.assertEqual([h["epoch"] for h in msr.read_history(self.run_dir)], [1, 2, 3])
        self.assertEqual(self._superseded(), [])
        record = msr.read_training_behavior(self.run_dir)
        self.assertEqual(record["established_git_commit"], "commitA")
        self.assertEqual(record["last_git_commit"], "commitB")
        self.assertEqual(record["events"][-1]["event"], "git_commit_changed_resume_allowed")

    def _assert_clean_restart(self, first_model_epochs=2):
        old_record = msr.read_training_behavior(self.run_dir)
        model = _tiny_arm_model()
        outcome = self._train_run(model, session_epoch_budget=1)
        decision = outcome.training_behavior

        self.assertEqual(decision["action"], "restarted")
        self.assertEqual(outcome.completed_epoch, 1)                      # epoch numbering restarted
        self.assertEqual(self._iterations(model), 3)                      # one epoch of steps, not 6 + 3
        self.assertEqual([h["epoch"] for h in msr.read_history(self.run_dir)], [1])   # old history not active
        self.assertEqual(msr.read_best(self.run_dir)[1]["epoch"], 0)      # BEST belongs to the new trajectory

        superseded = self._superseded()
        self.assertEqual(len(superseded), 1)
        old_generations = [n for n in os.listdir(os.path.join(superseded[0], "checkpoints"))
                           if n.startswith("gen_")]
        self.assertTrue(old_generations)                                 # old checkpoints preserved, not deleted
        self.assertEqual(len(os.listdir(os.path.join(superseded[0], "history"))), first_model_epochs)
        restart_json = json.load(open(os.path.join(superseded[0], "restart.json")))
        self.assertTrue(restart_json["completed"])
        self.assertEqual(restart_json["restart_reason"], decision["restart_reason"])

        record = msr.read_training_behavior(self.run_dir)
        self.assertEqual(record["active_training_behavior_hash"], decision["new_training_behavior_hash"])
        self.assertNotEqual(decision["old_training_behavior_hash"], decision["new_training_behavior_hash"])
        self.assertTrue(record["events"][-1]["training_behavior_changed"])
        if old_record is not None:
            self.assertEqual(decision["old_training_behavior_hash"], old_record["active_training_behavior_hash"])
        return decision

    def test_training_behavior_change_restarts_from_epoch_0_without_old_weights_or_optimizer(self):
        self._train_run(_tiny_arm_model(), session_epoch_budget=2)
        self.behavior_change = "augmentation changed"
        decision = self._assert_clean_restart()
        self.assertEqual(decision["restart_reason"], "training_behavior_fingerprint_changed")
        self.assertEqual((decision["old_git_commit"], decision["new_git_commit"]), ("commitA", "commitA"))
        self.assertIn("test_behavior_change", decision["changed_components"])

    def test_git_commit_and_training_behavior_change_restarts_from_epoch_0(self):
        self._train_run(_tiny_arm_model(), session_epoch_budget=2)
        self.commit, self.behavior_change = "commitB", "optimizer changed"
        decision = self._assert_clean_restart()
        self.assertEqual((decision["old_git_commit"], decision["new_git_commit"]), ("commitA", "commitB"))

    def test_a_trained_run_without_a_recorded_fingerprint_restarts(self):
        self._train_run(_tiny_arm_model(), session_epoch_budget=2)
        os.remove(os.path.join(self.run_dir, msr.TRAINING_BEHAVIOR_FILENAME))   # a pre-fingerprint run
        decision = self._assert_clean_restart()
        self.assertEqual(decision["restart_reason"], "training_behavior_fingerprint_missing")

    def test_created_run_under_a_new_git_commit_initializes_normally(self):
        with mock.patch.object(ckpt, "environment_fingerprint", return_value={"git_commit_hash": "commitA"}):
            self.run_dir, _ = msr.initialize_run(self.tmp, "exp1", "RACAF", 42, "deadbeef")
        self.commit = "commitB"
        outcome = self._train_run(_tiny_arm_model(), session_epoch_budget=1)
        self.assertEqual(outcome.training_behavior["action"], "established")
        self.assertEqual(outcome.completed_epoch, 1)
        self.assertEqual(self._superseded(), [])
        self.assertEqual(msr.read_training_behavior(self.run_dir)["established_git_commit"], "commitB")

    def test_completed_run_is_immutable_under_a_later_commit_and_changed_behavior(self):
        self._train_run(_tiny_arm_model(), max_epochs=2)
        record_before = msr.read_training_behavior(self.run_dir)
        history_before = msr.read_history(self.run_dir)
        best_before = msr.read_best(self.run_dir)[1]

        self.commit, self.behavior_change = "commitB", "architecture changed"
        outcome = self._train_run(_tiny_arm_model(), max_epochs=2)
        self.assertTrue(outcome.stopped)
        self.assertEqual(outcome.epochs_trained_this_call, 0)
        self.assertIsNone(outcome.training_behavior)
        self.assertEqual(self._superseded(), [])
        self.assertEqual(msr.read_training_behavior(self.run_dir), record_before)
        self.assertEqual(msr.read_history(self.run_dir), history_before)
        self.assertEqual(msr.read_best(self.run_dir)[1], best_before)


class TrainingBehaviorFingerprintTests(TempDirTestCase):
    ENTRIES = ([("id_b", 1), ("id_a", 0), ("id_c", 4)], [("id_v", 2)])

    def _copy_sources(self):
        for relative_path, _symbols in msr.TRAINING_BEHAVIOR_SOURCES:
            destination = os.path.join(self.tmp, *relative_path.split("/"))
            os.makedirs(os.path.dirname(destination), exist_ok=True)
            shutil.copyfile(os.path.join(msr.REPO_ROOT, *relative_path.split("/")), destination)

    def _edit(self, relative_path, old, new):
        path = os.path.join(self.tmp, *relative_path.split("/"))
        with open(path, encoding="utf-8") as handle:
            text = handle.read()
        self.assertEqual(text.count(old), 1, f"test anchor not unique in {relative_path}: {old!r}")
        with open(path, "w", encoding="utf-8") as handle:
            handle.write(text.replace(old, new))

    def _fingerprint(self, repo_root=None, **overrides):
        train, val = overrides.pop("entries", self.ENTRIES)
        return msr.training_behavior_fingerprint("RACAF", 42, train, val,
                                                 repo_root=repo_root or self.tmp, **overrides)

    def test_deterministic_and_independent_of_entry_order(self):
        self._copy_sources()
        a = self._fingerprint()
        b = self._fingerprint(entries=(list(reversed(self.ENTRIES[0])), self.ENTRIES[1]))
        self.assertEqual(a, b)
        self.assertEqual(a["components"]["configuration"]["split_seed"], 42)
        self.assertEqual(a["components"]["configuration"]["split_sha256"], msr.EXPECTED_SPLIT_SHA256)

    def test_the_cache_first_data_path_is_part_of_the_fingerprint(self):
        sources = self._fingerprint(repo_root=msr.REPO_ROOT)["components"]["sources"]
        for key in ("improved_training_data.py::load_cached_sample",
                    "improved_training_data.py::make_epoch_dataset",
                    "improved_training_data.py::epoch_training_order",
                    "improved_training_data.py::per_image_augmentation_rng",
                    "joint_training_dataset.py::_build_joint_sample",
                    "joint_cache_diagnostics.py::artifact_paths",
                    "multiseed_runs.py::build_arm_model", "weighted_corn.py", "racaf.py"):
            self.assertIn(key, sources)
        self.assertNotIn("improved_training_data.py::complete_local_cache", sources)
        self.assertNotIn("multiseed_runs.py::train_run", sources)

    def test_comments_docstrings_and_unrelated_code_do_not_change_the_fingerprint(self):
        self._copy_sources()
        before = self._fingerprint()
        self._edit("improved_training_data.py", '"""One joint sample built from the LOCAL cache only',
                   '"""EDITED documentation. One joint sample built from the LOCAL cache only')
        self._edit("improved_training_data.py",
                   "    missing = missing_local_artifacts(id_code, cache_dir, racaf_cache_dir, image_size)\n"
                   "    if missing:",
                   "    # an explanatory comment\n"
                   "    missing = missing_local_artifacts(id_code, cache_dir, racaf_cache_dir, image_size)  # why\n"
                   "\n    if missing:")
        self._edit("improved_training_data.py",   # infrastructure: one-time cache completion
                   '    report["already_local"] = len(entries) - len(not_local)\n',
                   '    report["already_local"] = len(entries) - len(not_local)\n    report["note"] = 1\n')
        self._edit("multiseed_runs.py", "def run_status(run_dir_path, lock_ttl_seconds=DEFAULT_LOCK_TTL_SECONDS):",
                   "def run_status(run_dir_path, lock_ttl_seconds=DEFAULT_LOCK_TTL_SECONDS):  # infra")
        self.assertEqual(self._fingerprint()["training_behavior_hash"], before["training_behavior_hash"])

    def test_a_training_behavior_code_change_changes_the_fingerprint_and_names_the_component(self):
        self._copy_sources()
        before = self._fingerprint()
        self._edit("improved_training_data.py", "        processed_dir=None, image_size=image_size,\n    )",
                   "        processed_dir=None, image_size=(8, 8),\n    )")
        after = self._fingerprint()
        self.assertNotEqual(after["training_behavior_hash"], before["training_behavior_hash"])
        self.assertEqual(msr.changed_behavior_components(before["components"], after["components"]),
                         ["sources.improved_training_data.py::load_cached_sample"])

    def test_configuration_and_population_changes_change_the_fingerprint(self):
        self._copy_sources()
        base = self._fingerprint()["training_behavior_hash"]
        self.assertNotEqual(base, self._fingerprint(batch_size=4)["training_behavior_hash"])
        self.assertNotEqual(base, msr.training_behavior_fingerprint(
            "RACAF", 123, *self.ENTRIES, repo_root=self.tmp)["training_behavior_hash"])
        self.assertNotEqual(base, msr.training_behavior_fingerprint(
            "NO_RACAF", 42, *self.ENTRIES, repo_root=self.tmp)["training_behavior_hash"])
        self.assertNotEqual(base, self._fingerprint(
            entries=(self.ENTRIES[0][:2], self.ENTRIES[1]))["training_behavior_hash"])


class _StaticShapeConsumer(tf.keras.layers.Layer):
    """Stands in for Swin's window reshapes (`swin_transformer.py:323` reads `x.shape[1]`/`[2]`):
    it consumes a STATIC shape, so it raises "as_list() is not defined on an unknown TensorShape"
    if the traced predict signature was relaxed by a rank change."""

    def call(self, x):
        dims = x.shape.as_list()
        return tf.reshape(x, (-1, dims[1]))


def _tiny_eval_model(arm):
    """A tiny model with the SAME reliability contract as the real arms: `Input(shape=(1,))`,
    consumed by the real `InertReliabilityConnection` (NO_RACAF) or by a `Dense` gate (RACAF's
    `reliability_gate`)."""
    import no_racaf_model as nrm

    stage5 = tf.keras.Input(shape=(4,), name="stage5_input")
    stage6 = tf.keras.Input(shape=(2,), name="stage6_input")
    reliability = tf.keras.Input(shape=(1,), name="reliability")
    merged = tf.keras.layers.Concatenate()([_StaticShapeConsumer()(stage5), stage6])
    logits = tf.keras.layers.Dense(corn.NUM_THRESHOLDS, name="corn_logits")(merged)
    if arm == "NO_RACAF":
        outputs = nrm.InertReliabilityConnection(name="no_racaf_inert_reliability")(
            [logits, reliability])
    else:
        gate = tf.keras.layers.Dense(1, activation="sigmoid", name="reliability_gate")(reliability)
        outputs = tf.keras.layers.Multiply()([logits, gate])
    return tf.keras.Model([stage5, stage6, reliability], outputs, name=f"tiny_{arm.lower()}")


def _fake_cached_sample(id_code, diagnosis, cache_dir, racaf_cache_dir, augment, rng,
                        image_size=None):
    value = float(int(id_code[-1]))
    return {"image_id": id_code, "stage5_input": np.full((4,), value, np.float32),
            "stage6_input": np.full((2,), value, np.float32),
            "reliability": np.float32(0.25 * value), "grade": int(diagnosis)}


class EvaluationReliabilityRankTests(unittest.TestCase):
    """Regression for the NO-RACAF evaluation crash: `evaluate_arm_from_disk()` must feed
    `reliability` with the rank the model's `Input(shape=(1,))` declares.

    `build_arm_model()` traces the NO_RACAF model's predict function with rank-2 `(2, 1)` probes
    (`no_racaf_model.verify_no_racaf_model()`); evaluating afterwards with the rank-1 `(N,)` that
    `np.stack` of per-sample scalars produces relaxed the traced signature to an unknown
    TensorShape and crashed with "as_list() is not defined on an unknown TensorShape". With the
    old `np.stack(rel)` this test fails (the NO_RACAF case raises; both cases see shape `(N,)`)."""

    def _evaluate(self, arm, verification_probes):
        model = _tiny_eval_model(arm)
        if verification_probes:
            # Exactly what verify_no_racaf_model() does at build time: rank-2 (B, 1) reliability.
            probe5, probe6 = np.zeros((2, 4), np.float32), np.zeros((2, 2), np.float32)
            model.predict_on_batch([probe5, probe6, np.zeros((2, 1), np.float32)])
            model.predict_on_batch([probe5, probe6, np.ones((2, 1), np.float32)])

        entries = [(f"id{i}", i % 5) for i in range(5)]
        with mock.patch("improved_training_data.load_cached_sample",
                        side_effect=_fake_cached_sample), \
             mock.patch.object(model, "predict_on_batch", wraps=model.predict_on_batch) as spy:
            rows = msr.evaluate_arm_from_disk(model, entries, "cache", "racaf_cache", batch_size=2)
        reliability_shapes = [np.asarray(call.args[0][2]).shape for call in spy.call_args_list]
        return rows, reliability_shapes

    def test_no_racaf_evaluation_succeeds_after_the_rank_2_verification_probes(self):
        rows, reliability_shapes = self._evaluate("NO_RACAF", verification_probes=True)
        self.assertEqual(len(rows), 5)
        self.assertEqual([shape[0] for shape in reliability_shapes], [2, 2, 1])  # batched 2/2/1
        for shape in reliability_shapes:
            self.assertEqual(len(shape), 2, f"reliability must be rank 2, got {shape}")
            self.assertEqual(shape[1], 1, f"reliability must be (N, 1), got {shape}")

    def test_racaf_evaluation_still_works_and_uses_the_same_rank(self):
        rows, reliability_shapes = self._evaluate("RACAF", verification_probes=False)
        self.assertEqual(len(rows), 5)
        for shape in reliability_shapes:
            self.assertEqual(shape[1:], (1,), f"reliability must be (N, 1), got {shape}")


class BuildOptimizerTests(unittest.TestCase):
    def test_excludes_bias_and_norm_params_on_a_small_model(self):
        inputs = tf.keras.Input(shape=(4,))
        x = tf.keras.layers.Dense(3, name="dense_a")(inputs)
        x = tf.keras.layers.BatchNormalization(name="bn_a")(x)
        x = tf.keras.layers.LayerNormalization(name="ln_a")(x)
        outputs = tf.keras.layers.Dense(1, name="dense_b")(x)
        model = tf.keras.Model(inputs, outputs)

        optimizer = msr.build_optimizer()
        variables = model.trainable_variables
        for variable in variables:
            use_decay = optimizer._use_weight_decay(variable)
            if variable.path.endswith("kernel"):
                self.assertTrue(use_decay, f"{variable.path} should NOT be excluded")
            if variable.path.endswith("bias") or "gamma" in variable.path or "beta" in variable.path:
                self.assertFalse(use_decay, f"{variable.path} should be excluded")


if __name__ == "__main__":
    unittest.main()
