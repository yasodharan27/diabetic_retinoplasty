"""Tests for no_racaf_model.py.

Builds the REAL ~43M-parameter joint models (no mocking of the architecture itself is possible
without losing the thing being tested: exact parameter/tensor counts and reliability inertness).
Slower than most unit tests in this project, but still CPU-only and self-contained -- no dataset,
cache, or GPU is touched. Cross-arm initialization parity (matched vs unmatched construction
order) is verified in ISOLATED subprocesses, since Keras 3's per-process layer-name counter and
global RNG state make an in-process second build unreliable evidence either way.
"""
import os
import subprocess
import sys
import unittest

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "-1")

import numpy as np

import corn
import joint_training_model as jtm
import no_racaf_model as nrm

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


class NoRacafModelStructureTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.model = nrm.build_no_racaf_joint_model()
        cls.model.compile(optimizer="adam", loss=jtm.joint_corn_loss,
                          metrics=[corn.CORNQuadraticWeightedKappa()])

    def test_expected_parameter_and_tensor_counts(self):
        parameters = sum(int(np.prod(v.shape)) for v in self.model.trainable_variables)
        tensors = len(self.model.trainable_variables)
        self.assertEqual(tensors, nrm.EXPECTED_TRAINABLE_TENSORS)
        self.assertEqual(parameters, nrm.EXPECTED_TRAINABLE_PARAMETERS)

    def test_delta_against_racaf_reference_is_exactly_racaf(self):
        parameters = sum(int(np.prod(v.shape)) for v in self.model.trainable_variables)
        tensors = len(self.model.trainable_variables)
        self.assertEqual(nrm.REFERENCE_TRAINABLE_PARAMETERS - parameters, nrm.RACAF_PARAMETERS_REMOVED)
        self.assertEqual(nrm.REFERENCE_TRAINABLE_TENSORS - tensors, nrm.RACAF_TENSORS_REMOVED)

    def test_no_racaf_layer_or_variable_present(self):
        layer_names = {layer.name for layer in self.model.layers}
        variable_paths = " ".join(v.path for v in self.model.trainable_variables).lower()
        self.assertNotIn("racaf_fusion", layer_names)
        self.assertNotIn("racaf", variable_paths)
        self.assertNotIn("reliability_gate", variable_paths)
        self.assertNotIn("global_projection", variable_paths)

    def test_reliability_input_is_declared_but_inert(self):
        result = nrm.verify_no_racaf_model(self.model)
        self.assertEqual(result["reliability_max_abs_diff"], 0.0)

    def test_output_shape_is_corn_logits(self):
        self.assertEqual(tuple(self.model.outputs[0].shape), (None, corn.NUM_THRESHOLDS))

    def test_verify_no_racaf_model_raises_on_a_tampered_model(self):
        # A model missing the inert-reliability layer must fail verification loudly, not pass.
        broken = jtm.build_joint_model()  # this IS the RACAF model -- wrong counts, has RACAF
        with self.assertRaises(RuntimeError):
            nrm.verify_no_racaf_model(broken)

    def test_weight_decay_exclusion_matches_real_variable_names(self):
        """multiseed_runs.build_optimizer()'s exclusion list, checked against every trainable
        variable of the REAL (large) model -- not merely in principle. Every bias must be
        excluded; every BatchNorm/LayerNorm gamma/beta must be excluded; every Dense/Conv
        kernel must NOT be excluded."""
        import multiseed_runs as msr

        optimizer = msr.build_optimizer()
        variables = self.model.trainable_variables
        self.assertTrue(any("bias" in v.path for v in variables), "no bias variable found to test")
        self.assertTrue(any("gamma" in v.path or "beta" in v.path for v in variables),
                        "no BatchNorm/LayerNorm gamma/beta variable found to test")

        excluded, included = [], []
        for variable in variables:
            (excluded if not optimizer._use_weight_decay(variable) else included).append(variable.path)

        for path in excluded:
            self.assertTrue(any(name in path for name in msr.WEIGHT_DECAY_EXCLUDE_NAMES),
                            f"{path} was excluded but matches none of {msr.WEIGHT_DECAY_EXCLUDE_NAMES}")
        for path in included:
            self.assertFalse(any(name in path for name in msr.WEIGHT_DECAY_EXCLUDE_NAMES),
                             f"{path} should have been excluded (matches "
                             f"{msr.WEIGHT_DECAY_EXCLUDE_NAMES}) but was not")
        kernel_paths = [v.path for v in variables if v.path.endswith("kernel")]
        self.assertTrue(kernel_paths, "no kernel variable found to test")
        for path in kernel_paths:
            self.assertIn(path, included, f"{path} (a kernel) was incorrectly excluded from weight decay")


_MATCHED_INIT_SCRIPT = """
import os, sys
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "-1")
sys.path.insert(0, {repo_root!r})
import numpy as np
import tensorflow as tf
tf.keras.utils.set_random_seed({seed})
import no_racaf_model as nrm
model = nrm.build_no_racaf_joint_model_matched_init() if {matched} else nrm.build_no_racaf_joint_model()
corn_layer = model.get_layer("corn")
kernel = corn_layer.get_layer("corn_logits").kernel.numpy()
bias = corn_layer.get_layer("corn_logits").bias.numpy()
np.save({kernel_path!r}, kernel)
np.save({bias_path!r}, bias)
"""

_RACAF_INIT_SCRIPT = """
import os, sys
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "-1")
sys.path.insert(0, {repo_root!r})
import numpy as np
import tensorflow as tf
tf.keras.utils.set_random_seed({seed})
import joint_training_model as jtm
model = jtm.build_joint_model()
corn_layer = model.get_layer("corn")
kernel = corn_layer.get_layer("corn_logits").kernel.numpy()
bias = corn_layer.get_layer("corn_logits").bias.numpy()
np.save({kernel_path!r}, kernel)
np.save({bias_path!r}, bias)
"""


def _run_build_in_subprocess(script_template, tmp_dir, tag, seed, matched=None):
    kernel_path = os.path.join(tmp_dir, f"{tag}_kernel.npy")
    bias_path = os.path.join(tmp_dir, f"{tag}_bias.npy")
    script = script_template.format(repo_root=REPO_ROOT, seed=seed, matched=matched,
                                    kernel_path=kernel_path, bias_path=bias_path)
    completed = subprocess.run([sys.executable, "-c", script], capture_output=True, text=True,
                               cwd=REPO_ROOT, timeout=600)
    if completed.returncode != 0:
        raise RuntimeError(f"subprocess failed:\nSTDOUT:\n{completed.stdout}\nSTDERR:\n{completed.stderr}")
    return np.load(kernel_path), np.load(bias_path)


@unittest.skipUnless(os.environ.get("RUN_SLOW_TESTS", "1") != "0",
                     "set RUN_SLOW_TESTS=0 to skip the isolated-process init-parity test")
class CrossArmInitializationParityTests(unittest.TestCase):
    """Proves build_no_racaf_joint_model_matched_init() restores CORN's initial-weight parity
    with the WITH-RACAF model under a shared seed, and that the PLAIN (non-matched) builder does
    NOT have this parity -- so the test is not vacuous."""

    def test_matched_init_gives_identical_corn_initial_weights(self):
        import tempfile
        with tempfile.TemporaryDirectory() as tmp:
            racaf_kernel, racaf_bias = _run_build_in_subprocess(_RACAF_INIT_SCRIPT, tmp, "racaf", seed=777)
            matched_kernel, matched_bias = _run_build_in_subprocess(
                _MATCHED_INIT_SCRIPT, tmp, "matched", seed=777, matched=True)
        np.testing.assert_array_equal(racaf_kernel, matched_kernel)
        np.testing.assert_array_equal(racaf_bias, matched_bias)

    def test_unmatched_init_does_not_give_identical_corn_initial_weights(self):
        import tempfile
        with tempfile.TemporaryDirectory() as tmp:
            racaf_kernel, _racaf_bias = _run_build_in_subprocess(_RACAF_INIT_SCRIPT, tmp, "racaf2", seed=778)
            plain_kernel, _plain_bias = _run_build_in_subprocess(
                _MATCHED_INIT_SCRIPT, tmp, "plain", seed=778, matched=False)
        self.assertFalse(np.array_equal(racaf_kernel, plain_kernel))


if __name__ == "__main__":
    unittest.main()
