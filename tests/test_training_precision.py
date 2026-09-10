"""
Tests for the precision/optimizer initialization order in `training.trainer`
and `joint_training_model.build_and_compile_joint_model`.

The defect these pin down: Keras 3 captures a layer's dtype policy when the
layer is CONSTRUCTED and decides on `LossScaleOptimizer` when the model is
COMPILED, but `training.Trainer` receives an already-built, already-compiled
model and only then calls `enable_mixed_precision()`. In that order the global
policy changes and the model does not -- so a run configured for
`mixed_precision=True` trains entirely in float32, with a bare `Adam` and no
loss scaling, while every flag reports mixed precision as enabled.

Note on what "correct" looks like under `mixed_float16`: trainable VARIABLES stay
float32 (`variable_dtype=float32`) and so do the gradients applied to them. Only
the compute dtype is float16. A float32 variable list is therefore not evidence
of a misconfiguration, and no test here treats it as such.
"""

import os
import shutil
import sys
import tempfile
import unittest

import numpy as np
import tensorflow as tf

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from training import (
    Trainer,
    TrainingConfig,
    enable_mixed_precision,
    expected_policy_name,
    model_precision_policies,
    verify_model_precision,
)
from training.trainer import DIAGNOSTIC_DIRTY_ATTRIBUTE


class PolicyPreservingTestCase(unittest.TestCase):
    """Every test here mutates the process-global dtype policy; none may leak it."""

    def setUp(self):
        self._policy = tf.keras.mixed_precision.global_policy()
        self.addCleanup(tf.keras.mixed_precision.set_global_policy, self._policy)


def tiny_model(units=4):
    return tf.keras.Sequential([
        tf.keras.layers.Input((8,)),
        tf.keras.layers.Dense(units, activation="relu", name="d1"),
        tf.keras.layers.Dense(1, name="d2", dtype="float32"),
    ])


class ModelPrecisionPolicyTests(PolicyPreservingTestCase):
    def test_reports_the_policy_the_weighted_layers_were_built_with(self):
        tf.keras.mixed_precision.set_global_policy("float32")
        self.assertEqual(model_precision_policies(tiny_model()), {"float32"})

        tf.keras.mixed_precision.set_global_policy("mixed_float16")
        # `tiny_model`'s output layer carries an explicit dtype="float32" override,
        # mirroring this project's real numerically-sensitive heads (racaf_output,
        # fused_embedding, AdaptiveBranchFusion's branch-weight projection), so BOTH
        # policies are legitimately present.
        self.assertEqual(model_precision_policies(tiny_model()),
                         {"float32", "mixed_float16"})

    def test_deliberate_float32_layer_overrides_do_not_count_as_a_mismatch(self):
        from training.trainer import precision_is_consistent
        self.assertTrue(precision_is_consistent("mixed_float16", {"float32", "mixed_float16"}))
        self.assertTrue(precision_is_consistent("mixed_float16", {"mixed_float16"}))
        self.assertTrue(precision_is_consistent("float32", {"float32"}))

    def test_absent_mixed_precision_and_stray_float16_are_both_mismatches(self):
        from training.trainer import precision_is_consistent
        self.assertFalse(precision_is_consistent("mixed_float16", {"float32"}))
        self.assertFalse(precision_is_consistent("float32", {"float32", "mixed_float16"}))

    def test_changing_the_global_policy_after_construction_does_not_change_the_model(self):
        """The exact mechanism behind the defect, asserted rather than assumed."""
        tf.keras.mixed_precision.set_global_policy("float32")
        model = tiny_model()
        model.compile(optimizer=tf.keras.optimizers.Adam(), loss="mse")
        self.assertEqual(model_precision_policies(model), {"float32"})
        self.assertIsNone(getattr(model.optimizer, "inner_optimizer", None))

        tf.keras.mixed_precision.set_global_policy("mixed_float16")
        self.assertEqual(tf.keras.mixed_precision.global_policy().name, "mixed_float16")
        self.assertEqual(model_precision_policies(model), {"float32"},
                         "a policy set after construction must not appear to change the model")
        self.assertIsNone(getattr(model.optimizer, "inner_optimizer", None))

    def test_building_under_mixed_float16_wraps_the_optimizer_in_a_loss_scale_optimizer(self):
        tf.keras.mixed_precision.set_global_policy("mixed_float16")
        model = tiny_model()
        model.compile(optimizer=tf.keras.optimizers.Adam(), loss="mse")
        self.assertEqual(type(model.optimizer).__name__, "LossScaleOptimizer")
        self.assertEqual(type(model.optimizer.inner_optimizer).__name__, "Adam")

    def test_mixed_float16_keeps_variables_in_float32_by_design(self):
        """Guards against 'fixing' a non-bug: float32 trainable tensors under
        mixed_float16 are correct, not a symptom."""
        tf.keras.mixed_precision.set_global_policy("mixed_float16")
        model = tiny_model()
        self.assertEqual({str(v.dtype) for v in model.trainable_variables}, {"float32"})
        layer = model.get_layer("d1")
        self.assertEqual(layer.dtype_policy.compute_dtype, "float16")
        self.assertEqual(layer.dtype_policy.variable_dtype, "float32")

    def test_expected_policy_name_follows_gpu_availability(self):
        expected = "mixed_float16" if tf.config.list_physical_devices("GPU") else "float32"
        self.assertEqual(expected_policy_name(True), expected)
        self.assertEqual(expected_policy_name(False), "float32")


class VerifyModelPrecisionTests(PolicyPreservingTestCase):
    def build_float16_model(self):
        tf.keras.mixed_precision.set_global_policy("mixed_float16")
        model = tiny_model()
        model.compile(optimizer=tf.keras.optimizers.Adam(), loss="mse")
        return model

    def test_consistent_configuration_passes(self):
        tf.keras.mixed_precision.set_global_policy("float32")
        model = tiny_model()
        model.compile(optimizer=tf.keras.optimizers.Adam(), loss="mse")
        report = verify_model_precision(model, mixed_precision_enabled=False,
                                        precision_check="error")
        self.assertTrue(report["consistent"])
        self.assertEqual(report["expected_policy"], "float32")

    def test_mismatch_raises_when_precision_check_is_error(self):
        model = self.build_float16_model()
        with self.assertRaises(RuntimeError) as caught:
            verify_model_precision(model, mixed_precision_enabled=False, precision_check="error")
        self.assertIn("mixed_float16", str(caught.exception))
        self.assertIn("BEFORE building the model", str(caught.exception))

    def test_mismatch_only_warns_by_default(self):
        model = self.build_float16_model()
        report = verify_model_precision(model, mixed_precision_enabled=False,
                                        precision_check="warn")
        self.assertFalse(report["consistent"])
        self.assertIn("mixed_float16", report["model_policies"])

    def test_check_can_be_disabled(self):
        model = self.build_float16_model()
        report = verify_model_precision(model, mixed_precision_enabled=False,
                                        precision_check="off")
        self.assertFalse(report["consistent"])

    def test_report_records_the_optimizer_and_loss_scaling(self):
        model = self.build_float16_model()
        report = verify_model_precision(model, mixed_precision_enabled=True, precision_check="off")
        self.assertEqual(report["optimizer"], "LossScaleOptimizer")
        self.assertTrue(report["loss_scaled"])


class TrainerPrecisionIntegrationTests(PolicyPreservingTestCase):
    def setUp(self):
        super().setUp()
        self.tmp = tempfile.mkdtemp(prefix="precision_test_")
        self.addCleanup(shutil.rmtree, self.tmp, True)

    def tiny_dataset(self):
        x = np.random.RandomState(0).rand(16, 8).astype("float32")
        y = (x[:, :1] * 2).astype("float32")
        return tf.data.Dataset.from_tensor_slices((x, y)).batch(8)

    def test_prepare_verifies_the_model_when_one_is_supplied(self):
        tf.keras.mixed_precision.set_global_policy("mixed_float16")
        model = tiny_model()
        model.compile(optimizer=tf.keras.optimizers.Adam(), loss="mse")

        config = TrainingConfig(run_dir=self.tmp, epochs=1, mixed_precision=False,
                                precision_check="error")
        with self.assertRaises(RuntimeError):
            Trainer(config).prepare(model)

    def test_prepare_without_a_model_stays_backwards_compatible(self):
        config = TrainingConfig(run_dir=self.tmp, epochs=1, mixed_precision=False)
        paths = Trainer(config).prepare()
        self.assertIn("best_weights", paths)

    def test_fit_records_a_precision_report(self):
        tf.keras.mixed_precision.set_global_policy("float32")
        model = tiny_model()
        model.compile(optimizer=tf.keras.optimizers.Adam(), loss="mse")
        config = TrainingConfig(run_dir=self.tmp, epochs=1, mixed_precision=False,
                                precision_check="error")
        trainer = Trainer(config)
        trainer.fit(model, self.tiny_dataset(), self.tiny_dataset())
        self.assertIsNotNone(trainer.precision_report)
        self.assertTrue(trainer.precision_report["consistent"])

    def test_fit_refuses_a_model_left_dirty_by_a_diagnostic(self):
        """A profiler run takes real gradient steps: the weights can be restored
        but the optimizer's slots and `iterations` cannot, so the model must be
        rebuilt before real training rather than silently continuing."""
        tf.keras.mixed_precision.set_global_policy("float32")
        model = tiny_model()
        model.compile(optimizer=tf.keras.optimizers.Adam(), loss="mse")
        setattr(model, DIAGNOSTIC_DIRTY_ATTRIBUTE, True)

        config = TrainingConfig(run_dir=self.tmp, epochs=1, mixed_precision=False)
        with self.assertRaises(RuntimeError) as caught:
            Trainer(config).fit(model, self.tiny_dataset(), self.tiny_dataset())
        self.assertIn("rebuild", str(caught.exception).lower())


class EnableMixedPrecisionTests(PolicyPreservingTestCase):
    def test_disabling_always_yields_float32(self):
        self.assertEqual(enable_mixed_precision(False).name, "float32")

    def test_enabling_matches_expected_policy_for_this_host(self):
        self.assertEqual(enable_mixed_precision(True).name, expected_policy_name(True))


class BuildAndCompileJointModelTests(PolicyPreservingTestCase):
    """One real joint-model build. Slow (43M parameters), so exactly one test
    does it -- but the ordering guarantee is worthless if it is only checked on
    a stand-in model."""

    def test_policy_and_optimizer_are_established_before_the_model_is_built(self):
        import joint_training_model as jtm

        tf.keras.mixed_precision.set_global_policy("float32")
        model = jtm.build_and_compile_joint_model(mixed_precision=True, verbose=0)

        from training.trainer import precision_is_consistent
        expected = expected_policy_name(True)
        self.assertTrue(precision_is_consistent(expected, model_precision_policies(model)))
        self.assertEqual(tf.keras.mixed_precision.global_policy().name, expected)

        inner = getattr(model.optimizer, "inner_optimizer", None)
        if expected == "mixed_float16":
            self.assertIsNotNone(inner, "mixed_float16 must be paired with a LossScaleOptimizer")
            self.assertEqual(type(inner).__name__, "Adam")
        else:
            self.assertIsNone(inner)
            self.assertEqual(type(model.optimizer).__name__, "Adam")

        self.assertEqual(model.loss.__name__, "joint_corn_loss")

    def test_float32_request_builds_a_float32_model_with_a_bare_adam(self):
        import joint_training_model as jtm

        tf.keras.mixed_precision.set_global_policy("mixed_float16")
        model = jtm.build_and_compile_joint_model(mixed_precision=False, verbose=0)
        self.assertEqual(model_precision_policies(model), {"float32"},
                         "a float32 run must contain no float16 layer at all")
        self.assertEqual(type(model.optimizer).__name__, "Adam")


if __name__ == "__main__":
    unittest.main()
