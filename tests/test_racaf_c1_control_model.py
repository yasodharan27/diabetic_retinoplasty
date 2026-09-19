"""Tests for racaf_c1_control_model.py.

Builds the REAL ~43M-parameter joint model (no mocking of the architecture itself is possible
without losing the thing being tested: exact parameter/tensor counts and the gate's inertness to
`r`). Slower than most unit tests in this project, but still CPU-only and self-contained -- no
dataset, cache, or GPU is touched. Cross-arm initialization parity (matched vs unmatched
construction order) is verified in ISOLATED subprocesses, exactly as
`tests/test_no_racaf_model.py::CrossArmInitializationParityTests` already does for NO_RACAF.
"""
import os
import subprocess
import sys
import unittest

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "-1")

import numpy as np

import corn
import joint_training_model as jtm
import racaf_c1_control_model as c1m

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


class C1ModelStructureTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.model = c1m.build_c1_joint_model()
        cls.model.compile(optimizer="adam", loss=jtm.joint_corn_loss,
                          metrics=[corn.CORNQuadraticWeightedKappa()])

    def test_expected_parameter_and_tensor_counts(self):
        parameters = sum(int(np.prod(v.shape)) for v in self.model.trainable_variables)
        tensors = len(self.model.trainable_variables)
        self.assertEqual(tensors, c1m.EXPECTED_TRAINABLE_TENSORS)
        self.assertEqual(parameters, c1m.EXPECTED_TRAINABLE_PARAMETERS)

    def test_delta_against_racaf_reference_is_exactly_the_gate_kernel(self):
        parameters = sum(int(np.prod(v.shape)) for v in self.model.trainable_variables)
        tensors = len(self.model.trainable_variables)
        self.assertEqual(c1m.RACAF_REFERENCE_TRAINABLE_PARAMETERS - parameters,
                         c1m.GATE_KERNEL_PARAMETERS_REMOVED)
        self.assertEqual(c1m.RACAF_REFERENCE_TRAINABLE_TENSORS - tensors,
                         c1m.GATE_KERNEL_TENSORS_REMOVED)

    def test_no_racaf_r_dependent_gate_variable_present_but_ghat_pathway_is(self):
        variable_paths = " ".join(v.path for v in self.model.trainable_variables).lower()
        self.assertNotIn("reliability_gate", variable_paths)
        self.assertIn("global_projection", variable_paths)

    def test_gate_is_inert_to_r_but_model_is_not_degenerate(self):
        result = c1m.verify_c1_model(self.model)
        self.assertEqual(result["reliability_max_abs_diff"], 0.0)
        self.assertIsInstance(result["gate_value"], float)
        self.assertGreaterEqual(result["gate_value"], 0.0)
        self.assertLessEqual(result["gate_value"], 1.0)

    def test_output_shape_is_corn_logits(self):
        self.assertEqual(tuple(self.model.outputs[0].shape), (None, corn.NUM_THRESHOLDS))

    def test_verify_c1_model_raises_on_a_tampered_model(self):
        # The actual RACAF model has RACAF's own r-dependent gate and different counts -- must
        # fail C1's verification loudly, not silently pass.
        broken = jtm.build_joint_model()
        with self.assertRaises(RuntimeError):
            c1m.verify_c1_model(broken)

    def test_weight_decay_exclusion_matches_real_variable_names(self):
        """multiseed_runs.build_optimizer()'s exclusion list, checked against every trainable
        variable of the REAL C1 model. Every bias (including the gate's own, named "bias" for
        exactly this reason) must be excluded; every BatchNorm/LayerNorm gamma/beta must be
        excluded; every Dense/Conv kernel must NOT be excluded."""
        import multiseed_runs as msr

        optimizer = msr.build_optimizer()
        variables = self.model.trainable_variables
        gate_bias_paths = [v.path for v in variables if "c1_constant_gate" in v.path]
        self.assertTrue(gate_bias_paths, "C1's gate bias variable not found")

        excluded, included = [], []
        for variable in variables:
            (excluded if not optimizer._use_weight_decay(variable) else included).append(variable.path)

        self.assertTrue(all(path in excluded for path in gate_bias_paths),
                        "C1's gate bias must be excluded from weight decay, like every other bias")
        for path in excluded:
            self.assertTrue(any(name in path for name in msr.WEIGHT_DECAY_EXCLUDE_NAMES),
                            f"{path} was excluded but matches none of {msr.WEIGHT_DECAY_EXCLUDE_NAMES}")
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
import racaf_c1_control_model as c1m
model = c1m.build_c1_joint_model_matched_init() if {matched} else c1m.build_c1_joint_model()
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
    """Proves build_c1_joint_model_matched_init() restores CORN's initial-weight parity with the
    WITH-RACAF model under a shared seed, and that the PLAIN (non-matched) builder does NOT have
    this parity -- so the test is not vacuous. Exactly the same proof structure as
    `tests/test_no_racaf_model.py`'s equivalent RACAF-vs-NO_RACAF test, applied to RACAF-vs-C1."""

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
