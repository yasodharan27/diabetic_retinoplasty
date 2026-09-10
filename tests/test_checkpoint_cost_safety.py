"""
Path-safety tests for `joint_training_profiler.measure_checkpoint_cost()`.

The defect these pin down: the function used to run

    shutil.rmtree(os.path.dirname(checkpoint_dir), ignore_errors=True)

on a CALLER-SUPPLIED path. Passing `<experiment.root>/checkpoints` -- the obvious
argument for "measure this on Drive" -- would have deleted `<experiment.root>`
outright: every checkpoint generation, `best/`, `logs/` and `metadata.json`, with
`ignore_errors=True` suppressing any trace of it.

The diagnostic now creates a uniquely named workspace of its own under the
supplied location, marks it with an ownership token, and removes only that
workspace. It refuses locations that are, or sit inside, an experiment or a
checkpoint directory, and system-critical roots outright. Every test here uses a
real (tiny) model through the real function; nothing is mocked except, in one
test, the deletion primitive -- to prove a cleanup failure is surfaced.
"""

import json
import os
import shutil
import sys
import tempfile
import unittest
from unittest import mock

import numpy as np
import tensorflow as tf

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

import joint_training_profiler as jtp
import training.checkpointing as ckpt


def tiny_model():
    tf.keras.utils.set_random_seed(7)
    model = tf.keras.Sequential([
        tf.keras.layers.Input((8,)),
        tf.keras.layers.Dense(4, activation="relu"),
        tf.keras.layers.Dense(1),
    ])
    model.compile(optimizer=tf.keras.optimizers.Adam(0.01), loss="mse")
    return model


def snapshot(root):
    """Every file and directory under `root`, files mapped to their SHA256."""
    entries = {}
    for directory, dirs, files in os.walk(root):
        for name in dirs:
            entries[os.path.relpath(os.path.join(directory, name), root) + os.sep] = "<dir>"
        for name in files:
            path = os.path.join(directory, name)
            entries[os.path.relpath(path, root)] = ckpt.sha256_file(path)
    return entries


def make_experiment(root, with_checkpoint=True):
    """An experiment directory exactly as `experiment_manager.create_experiment()`
    lays it out, optionally with a real sealed checkpoint generation and BEST."""
    for sub in ("checkpoints", "logs", "tensorboard", "evaluation", "predictions"):
        os.makedirs(os.path.join(root, sub), exist_ok=True)
    with open(os.path.join(root, "metadata.json"), "w") as handle:
        json.dump({"timestamp": "2026-09-10T00:00:00", "batch_size": 2}, handle)
    with open(os.path.join(root, "logs", "events.out.tfevents.probe"), "w") as handle:
        handle.write("tensorboard")
    if with_checkpoint:
        model = tiny_model()
        checkpoint_dir = os.path.join(root, "checkpoints")
        generation = ckpt.save_generation(
            checkpoint_dir, model, ckpt.TrainingState(completed_epoch=1, best_metric=0.8,
                                                      best_epoch=0),
            staging_dir=os.path.join(os.path.dirname(root), "_staging_for_fixture"))
        ckpt.save_best(checkpoint_dir, generation, ckpt.read_state(generation),
                       staging_dir=os.path.join(os.path.dirname(root), "_staging_for_fixture"))
    return root


class CheckpointCostWorkspaceSafetyTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.mkdtemp(prefix="ckpt_cost_safety_")
        self.addCleanup(shutil.rmtree, self.tmp, True)
        self.staging = os.path.join(self.tmp, "_staging")
        self.model = tiny_model()

    def measure(self, checkpoint_dir, **kwargs):
        return jtp.measure_checkpoint_cost(self.model, checkpoint_dir=checkpoint_dir,
                                           staging_dir=self.staging, **kwargs)

    # -- 1. a supplied diagnostic location: only the workspace is cleaned ----

    def test_supplied_checkpoint_cost_location_cleans_only_its_own_workspace(self):
        parent = os.path.join(self.tmp, "checkpoint_cost")
        base = os.path.join(parent, "checkpoints")
        os.makedirs(base)
        with open(os.path.join(base, "keep_me.txt"), "w") as handle:
            handle.write("caller-owned, inside the supplied location")
        with open(os.path.join(parent, "parent_keep_me.txt"), "w") as handle:
            handle.write("caller-owned, in the PARENT the old code deleted")

        report = self.measure(base)

        self.assertTrue(report["validation_ok"])
        self.assertGreater(report["sizes"]["total_bytes"], 0)
        self.assertEqual(os.path.dirname(report["workspace"]), os.path.abspath(base))
        self.assertTrue(os.path.basename(report["workspace"]).startswith(
            jtp.CHECKPOINT_COST_WORKSPACE_PREFIX))
        self.assertTrue(report["cleaned_up"])
        self.assertFalse(os.path.exists(report["workspace"]))
        self.assertTrue(os.path.isfile(os.path.join(base, "keep_me.txt")))
        self.assertTrue(os.path.isfile(os.path.join(parent, "parent_keep_me.txt")))
        self.assertEqual(sorted(os.listdir(base)), ["keep_me.txt"])

    # -- 2, 3. an experiment's checkpoints dir can never cost the experiment -

    def test_experiment_checkpoints_dir_is_refused_and_the_experiment_survives(self):
        experiment = make_experiment(os.path.join(self.tmp, "experiments", "2026-09-10_00-00-00"))
        before = snapshot(experiment)
        self.assertIn(os.path.join("checkpoints", "best", ckpt.MODEL_WEIGHTS_FILENAME), before)

        with self.assertRaises(jtp.UnsafeDiagnosticPathError) as caught:
            self.measure(os.path.join(experiment, "checkpoints"))
        self.assertIn("experiment", str(caught.exception).lower())

        self.assertTrue(os.path.isdir(experiment))
        self.assertEqual(snapshot(experiment), before)

    def test_experiment_root_is_refused_and_its_files_remain(self):
        experiment = make_experiment(os.path.join(self.tmp, "exp_root"))
        before = snapshot(experiment)
        with self.assertRaises(jtp.UnsafeDiagnosticPathError):
            self.measure(experiment)
        self.assertEqual(snapshot(experiment), before)
        self.assertTrue(os.path.isfile(os.path.join(experiment, "metadata.json")))

    def test_experiment_that_has_not_checkpointed_yet_is_still_refused(self):
        """Layout alone (metadata.json + experiment subfolders) identifies an
        experiment; it must not need a checkpoint to be protected."""
        experiment = make_experiment(os.path.join(self.tmp, "young_exp"), with_checkpoint=False)
        before = snapshot(experiment)
        with self.assertRaises(jtp.UnsafeDiagnosticPathError):
            self.measure(os.path.join(experiment, "checkpoints"))
        self.assertEqual(snapshot(experiment), before)

    def test_location_nested_inside_an_experiment_is_refused(self):
        experiment = make_experiment(os.path.join(self.tmp, "nest_exp"), with_checkpoint=False)
        nested = os.path.join(experiment, "evaluation", "probe")
        with self.assertRaises(jtp.UnsafeDiagnosticPathError):
            self.measure(nested)
        self.assertFalse(os.path.exists(nested))

    def test_a_bare_checkpoint_directory_is_refused(self):
        """A checkpoints dir that holds generations but has no experiment layout
        around it -- e.g. `training.Trainer` pointed at an ad-hoc run_dir."""
        checkpoint_dir = os.path.join(self.tmp, "adhoc_run", "checkpoints")
        ckpt.save_generation(checkpoint_dir, tiny_model(), ckpt.TrainingState(completed_epoch=1),
                             staging_dir=self.staging)
        before = snapshot(checkpoint_dir)
        with self.assertRaises(jtp.UnsafeDiagnosticPathError):
            self.measure(checkpoint_dir)
        self.assertEqual(snapshot(checkpoint_dir), before)

    # -- 4, 5, 6. system-critical roots ------------------------------------

    def _assert_refused_without_side_effects(self, path):
        existed = os.path.exists(path)
        listing = sorted(os.listdir(path)) if os.path.isdir(path) else None
        with self.assertRaises(jtp.UnsafeDiagnosticPathError):
            self.measure(path)
        self.assertEqual(os.path.exists(path), existed, f"{path} was created or removed")
        if listing is not None:
            self.assertEqual(sorted(os.listdir(path)), listing, f"{path} was modified")

    def test_repository_root_is_refused(self):
        self._assert_refused_without_side_effects(REPO_ROOT)

    def test_location_inside_the_repository_is_refused(self):
        self._assert_refused_without_side_effects(os.path.join(REPO_ROOT, "tests"))

    def test_content_is_refused(self):
        self._assert_refused_without_side_effects("/content")

    def test_content_drive_is_refused(self):
        self._assert_refused_without_side_effects("/content/drive")
        self._assert_refused_without_side_effects("/content/drive/MyDrive")

    def test_filesystem_root_and_home_are_refused(self):
        self._assert_refused_without_side_effects(os.path.abspath(os.sep))
        self._assert_refused_without_side_effects(os.path.expanduser("~"))

    def test_cleanup_guard_itself_refuses_critical_roots(self):
        """Independent of the entry-point checks: the deletion routine refuses a
        critical root even if called with one directly."""
        for path in (REPO_ROOT, "/content", "/content/drive", os.path.abspath(os.sep),
                     os.path.expanduser("~")):
            with self.assertRaises(jtp.UnsafeDiagnosticPathError):
                jtp._remove_owned_workspace(path, os.path.dirname(path) or path, "token")
        self.assertTrue(os.path.isdir(REPO_ROOT))
        self.assertTrue(os.path.isfile(os.path.join(REPO_ROOT, "joint_training_profiler.py")))

    # -- 7. cleanup removes only what this invocation created ---------------

    def test_cleanup_only_removes_the_workspace_this_invocation_created(self):
        base = os.path.join(self.tmp, "shared_probe_base")
        impostor = os.path.join(base, jtp.CHECKPOINT_COST_WORKSPACE_PREFIX + "left_by_someone")
        os.makedirs(impostor)
        with open(os.path.join(impostor, "important.bin"), "wb") as handle:
            handle.write(b"\x00" * 64)
        with open(os.path.join(base, "notes.txt"), "w") as handle:
            handle.write("caller-owned")

        report = self.measure(base)

        self.assertFalse(os.path.exists(report["workspace"]))
        self.assertTrue(os.path.isfile(os.path.join(impostor, "important.bin")),
                        "a same-prefixed directory this call did not create was deleted")
        self.assertTrue(os.path.isfile(os.path.join(base, "notes.txt")))

    def test_cleanup_guard_refuses_a_directory_it_does_not_own(self):
        base = os.path.join(self.tmp, "guard_base")
        foreign = os.path.join(base, jtp.CHECKPOINT_COST_WORKSPACE_PREFIX + "foreign")
        os.makedirs(foreign)
        open(os.path.join(foreign, "data"), "w").close()

        with self.assertRaises(jtp.UnsafeDiagnosticPathError):      # no ownership marker
            jtp._remove_owned_workspace(foreign, base, "any-token")
        with open(os.path.join(foreign, jtp.CHECKPOINT_COST_OWNER_MARKER), "w") as handle:
            handle.write("someone-elses-token")
        with self.assertRaises(jtp.UnsafeDiagnosticPathError):      # wrong token
            jtp._remove_owned_workspace(foreign, base, "my-token")
        with self.assertRaises(jtp.UnsafeDiagnosticPathError):      # not a child of base
            jtp._remove_owned_workspace(foreign, self.tmp, "someone-elses-token")
        self.assertTrue(os.path.isfile(os.path.join(foreign, "data")))

    def test_cleanup_false_keeps_the_workspace_for_inspection(self):
        report = self.measure(os.path.join(self.tmp, "keep_base"), cleanup=False)
        self.assertFalse(report["cleaned_up"])
        self.assertTrue(os.path.isdir(report["workspace"]))
        self.assertTrue(ckpt.validate_generation(report["generation_dir"]).ok)

    # -- 8. a cleanup failure is surfaced, never swallowed ------------------

    def test_cleanup_failure_is_surfaced(self):
        base = os.path.join(self.tmp, "failing_cleanup")
        with mock.patch.object(jtp, "_delete_tree", side_effect=OSError("device busy")):
            with self.assertRaises(jtp.CheckpointCostCleanupError) as caught:
                self.measure(base)
        message = str(caught.exception)
        self.assertIn("device busy", message)
        leftovers = [n for n in os.listdir(base)
                     if n.startswith(jtp.CHECKPOINT_COST_WORKSPACE_PREFIX)]
        self.assertEqual(len(leftovers), 1, "the error must describe a workspace that exists")
        self.assertIn(os.path.join(os.path.abspath(base), leftovers[0]), message)

    # -- notebook use cases are preserved ------------------------------------

    def test_default_location_is_a_private_workspace_under_the_temp_dir(self):
        report = jtp.measure_checkpoint_cost(self.model, staging_dir=self.staging)
        self.assertTrue(report["workspace"].startswith(os.path.abspath(tempfile.gettempdir())))
        self.assertFalse(os.path.exists(report["workspace"]))
        self.assertTrue(os.path.isdir(tempfile.gettempdir()))

    def test_drive_probe_beside_real_experiments_leaves_them_untouched(self):
        """The notebook's Drive probe lives at
        `experiments/FinalClassification/_checkpoint_cost_probe`, a sibling of
        real experiments. It must measure there and leave its siblings intact."""
        family = os.path.join(self.tmp, "experiments", "FinalClassification")
        sibling = make_experiment(os.path.join(family, "2026-09-01_12-00-00"))
        before = snapshot(sibling)

        report = self.measure(os.path.join(family, "_checkpoint_cost_probe"))

        self.assertTrue(report["validation_ok"])
        self.assertFalse(os.path.exists(report["workspace"]))
        self.assertEqual(snapshot(sibling), before)


if __name__ == "__main__":
    unittest.main()
