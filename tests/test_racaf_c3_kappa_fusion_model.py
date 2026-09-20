"""Tests for racaf_c3_kappa_fusion_model.py and improved_training_data_kappa.py.

Builds the REAL ~43M-parameter joint model (exact parameter counts and the zero-init `F == E`
property cannot be tested on a mock). CPU-only and self-contained: no GPU, no Drive, no dataset.
The loader tests run against a tiny on-disk cache fixture built at a 32x32 canonical size, which
exercises the REAL `joint_training_dataset._build_joint_sample` path -- proving the loader needs
no Stage 04 model, no raw image and no cache write.

Cross-arm initialisation parity is verified in ISOLATED subprocesses, exactly as
`tests/test_no_racaf_model.py` and `tests/test_racaf_c1_control_model.py` already do.
"""
import os
import shutil
import subprocess
import sys
import tempfile
import unittest

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "-1")

import numpy as np

import corn
import improved_training_data as itd
import improved_training_data_kappa as itdk
import joint_training_dataset as jtd
import joint_training_model as jtm
import local_feature_extraction_dataset as lfed
import racaf
import racaf_c3_kappa_fusion_model as c3m

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


class C3ModelStructureTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.model = c3m.build_c3_joint_model()
        cls.model.compile(optimizer="adam", loss=jtm.joint_corn_loss,
                          metrics=[corn.CORNQuadraticWeightedKappa()])

    def test_exact_parameter_and_tensor_counts(self):
        parameters = sum(int(np.prod(v.shape)) for v in self.model.trainable_variables)
        tensors = len(self.model.trainable_variables)
        self.assertEqual(parameters, 43_339_784)
        self.assertEqual(parameters, c3m.EXPECTED_TRAINABLE_PARAMETERS)
        self.assertEqual(tensors, c3m.EXPECTED_TRAINABLE_TENSORS)

    def test_parameter_budget_decomposes_exactly(self):
        parameters = sum(int(np.prod(v.shape)) for v in self.model.trainable_variables)
        self.assertEqual(
            parameters,
            c3m.NO_RACAF_TRAINABLE_PARAMETERS + c3m.GHAT_PARAMETERS + c3m.GAMMA_PARAMETERS)
        self.assertEqual(parameters - c3m.RACAF_REFERENCE_TRAINABLE_PARAMETERS, 1_278)
        self.assertEqual(parameters - c3m.NO_RACAF_TRAINABLE_PARAMETERS, 296_448)

    def test_gamma_layer_has_exactly_1280_parameters(self):
        gamma = self.model.get_layer(c3m.C3_FUSION_NAME).get_layer(c3m.GAMMA_LAYER_NAME)
        self.assertEqual(sum(int(np.prod(v.shape)) for v in gamma.trainable_variables), 1_280)
        self.assertEqual(c3m.GAMMA_PARAMETERS, 1_280)
        self.assertEqual(tuple(gamma.kernel.shape), (4, 256))
        self.assertEqual(tuple(gamma.bias.shape), (256,))
        self.assertTrue(gamma.use_bias)

    def test_gamma_is_zero_initialised_and_linear(self):
        gamma = self.model.get_layer(c3m.C3_FUSION_NAME).get_layer(c3m.GAMMA_LAYER_NAME)
        np.testing.assert_array_equal(np.asarray(gamma.kernel), np.zeros((4, 256), np.float32))
        np.testing.assert_array_equal(np.asarray(gamma.bias), np.zeros((256,), np.float32))
        import tensorflow as tf
        self.assertIs(gamma.activation, tf.keras.activations.linear)

    def test_gamma_is_exactly_zero_for_any_kappa_at_init(self):
        rng = np.random.default_rng(0)
        kappa = rng.random((16, 4), dtype=np.float32)
        gamma = c3m.gamma_from_kappa(self.model, kappa)
        self.assertEqual(gamma.shape, (16, 256))
        np.testing.assert_array_equal(gamma, np.zeros((16, 256), np.float32))

    def test_fusion_output_equals_E_exactly_at_init(self):
        fusion = self.model.get_layer(c3m.C3_FUSION_NAME)
        rng = np.random.default_rng(1)
        e = rng.random((3, 256), dtype=np.float32)
        g = rng.random((3, 64, 1152), dtype=np.float32)
        kappa = rng.random((3, 4), dtype=np.float32)
        fused = np.asarray(fusion.predict_on_batch([e, g, kappa]))
        np.testing.assert_array_equal(fused, e)

    def test_kappa_input_shape_is_exactly_four(self):
        kappa_inputs = [t for t in self.model.inputs
                        if len(t.shape) == 2 and t.shape[-1] == 4]
        self.assertEqual(len(kappa_inputs), 1)
        self.assertEqual(tuple(kappa_inputs[0].shape), (None, 4))
        self.assertEqual(c3m.KAPPA_DIM, 4)
        self.assertEqual(itdk.KAPPA_DIM, 4)

    def test_no_scalar_reliability_gate_remains(self):
        variable_paths = " ".join(v.path for v in self.model.trainable_variables).lower()
        self.assertNotIn("reliability_gate", variable_paths)
        self.assertIn("global_projection", variable_paths)
        self.assertIn("gamma_projection", variable_paths)

    def test_output_shape_is_corn_logits(self):
        self.assertEqual(tuple(self.model.outputs[0].shape), (None, corn.NUM_THRESHOLDS))

    def test_fusion_output_is_cast_to_float32(self):
        fusion = self.model.get_layer(c3m.C3_FUSION_NAME)
        self.assertEqual(fusion.get_layer("c3_output").dtype_policy.name, "float32")

    def test_verify_c3_model_passes_and_rejects_a_tampered_model(self):
        result = c3m.verify_c3_model(self.model)
        self.assertEqual(result["trainable_parameters"], 43_339_784)
        self.assertEqual(result["gamma_parameters"], 1_280)
        self.assertTrue(result["gamma_is_zero"])
        self.assertTrue(result["fusion_equals_e_at_init"])
        with self.assertRaises(RuntimeError):
            c3m.verify_c3_model(jtm.build_joint_model())   # the RACAF model is not a C3 model

    def test_weight_decay_applies_to_gamma_kernel_but_not_gamma_bias(self):
        import multiseed_runs as msr
        optimizer = msr.build_optimizer()
        gamma = self.model.get_layer(c3m.C3_FUSION_NAME).get_layer(c3m.GAMMA_LAYER_NAME)
        self.assertTrue(optimizer._use_weight_decay(gamma.kernel),
                        "gamma's kernel must receive the same AdamW decay as every other kernel")
        self.assertFalse(optimizer._use_weight_decay(gamma.bias),
                         "gamma's bias must be excluded from decay, like every other bias")

    def test_mechanism_diagnostic_reports_zero_engagement_at_init(self):
        rng = np.random.default_rng(2)
        stage5 = rng.random((2, *jtm.STAGE5_INPUT_SHAPE), dtype=np.float32)
        stage6 = rng.random((2, *jtm.STAGE6_INPUT_SHAPE), dtype=np.float32)
        kappa = rng.random((2, 4), dtype=np.float32)
        report = c3m.compute_mechanism_diagnostics(self.model, stage5, stage6, kappa, batch_size=2)
        self.assertEqual(report["n_samples"], 2)
        self.assertEqual(report["mean_abs_gamma"], 0.0)
        self.assertEqual(report["residual_ratio_median"], 0.0)
        self.assertFalse(report["mechanism_engaged"])
        self.assertEqual(report["mechanism_ratio_threshold"], 0.01)


def _write_cache_fixture(cache_dir, racaf_cache_dir, id_codes, image_size, seed=0):
    """Builds the four real cache artifacts per image, using the pipeline's OWN path builders."""
    os.makedirs(cache_dir, exist_ok=True)
    os.makedirs(racaf_cache_dir, exist_ok=True)
    rng = np.random.default_rng(seed)
    kappas = {}
    for index, id_code in enumerate(id_codes):
        np.save(lfed._cache_path(cache_dir, id_code, "rgb", image_size),
                rng.random((*image_size, 3)).astype(np.float32))
        np.save(lfed._cache_path(cache_dir, id_code, "vessel", image_size),
                rng.random((*image_size, 1)).astype(np.float32))
        np.save(lfed._cache_path(cache_dir, id_code, "lesion", image_size),
                rng.random((*image_size, 4)).astype(np.float32))
        # kappa[0] (MA) pinned to 1.0, mirroring the real cache's own degeneracy.
        kappa = np.array([1.0, 0.9 - 0.01 * index, 0.8 + 0.01 * index, 1.0], dtype=np.float32)
        np.savez(racaf.reliability_cache_path(racaf_cache_dir, id_code),
                 kappa=kappa, r=np.float32(float(kappa.mean())))
        kappas[id_code] = kappa
    return kappas


class KappaLoaderTests(unittest.TestCase):
    """Exercises the REAL `_build_joint_sample` path against a tiny on-disk cache."""

    IMAGE_SIZE = (32, 32)

    def setUp(self):
        self.tmp = tempfile.mkdtemp()
        self.cache_dir = os.path.join(self.tmp, "cache")
        self.racaf_cache_dir = os.path.join(self.tmp, "racaf")
        self.entries = [("aaa111", 0), ("bbb222", 2), ("ccc333", 4)]
        self.kappas = _write_cache_fixture(
            self.cache_dir, self.racaf_cache_dir, [e[0] for e in self.entries], self.IMAGE_SIZE)

    def tearDown(self):
        shutil.rmtree(self.tmp, ignore_errors=True)

    def _snapshot(self):
        snapshot = {}
        for root, _dirs, files in os.walk(self.tmp):
            for name in files:
                path = os.path.join(root, name)
                with open(path, "rb") as handle:
                    snapshot[path] = handle.read()
        return snapshot

    def test_load_kappa_returns_exact_cached_vector(self):
        for id_code, _grade in self.entries:
            kappa = itdk.load_kappa(id_code, self.racaf_cache_dir)
            self.assertEqual(kappa.shape, (4,))
            self.assertEqual(kappa.dtype, np.float32)
            np.testing.assert_array_equal(kappa, self.kappas[id_code])

    def test_load_kappa_raises_rather_than_recomputing(self):
        with self.assertRaises(itdk.MissingKappaError):
            itdk.load_kappa("not_cached_at_all", self.racaf_cache_dir)

    def test_sample_is_identical_to_the_existing_loader_plus_kappa(self):
        """The whole point of the wrapper: stage5/stage6/grade must come through byte-identical,
        proving augmentation and the cached-sample path are untouched."""
        for augment in (False, True):
            for id_code, grade in self.entries:
                rng_a = itd.per_image_augmentation_rng(42, 3, id_code) if augment else None
                rng_b = itd.per_image_augmentation_rng(42, 3, id_code) if augment else None
                baseline = itd.load_cached_sample(id_code, grade, self.cache_dir,
                                                  self.racaf_cache_dir, augment, rng_a,
                                                  image_size=self.IMAGE_SIZE)
                wrapped = itdk.load_cached_sample_kappa(id_code, grade, self.cache_dir,
                                                        self.racaf_cache_dir, augment, rng_b,
                                                        image_size=self.IMAGE_SIZE)
                np.testing.assert_array_equal(wrapped["stage5_input"], baseline["stage5_input"])
                np.testing.assert_array_equal(wrapped["stage6_input"], baseline["stage6_input"])
                self.assertEqual(wrapped["grade"], baseline["grade"])
                self.assertEqual(wrapped["image_id"], baseline["image_id"])
                np.testing.assert_array_equal(wrapped["kappa"], self.kappas[id_code])

    def test_loading_never_writes_or_regenerates_the_cache(self):
        before = self._snapshot()
        for id_code, grade in self.entries:
            itdk.load_cached_sample_kappa(id_code, grade, self.cache_dir, self.racaf_cache_dir,
                                          False, None, image_size=self.IMAGE_SIZE)
        after = self._snapshot()
        self.assertEqual(sorted(before), sorted(after), "cache file set changed")
        for path, payload in before.items():
            self.assertEqual(payload, after[path], f"{path} was modified")

    def test_epoch_dataset_signature_differs_only_in_the_third_input(self):
        ds_kappa = itdk.make_epoch_dataset_kappa(
            self.entries, epoch=0, run_seed=42, cache_dir=self.cache_dir,
            racaf_cache_dir=self.racaf_cache_dir, batch_size=2, augment=False,
            image_size=self.IMAGE_SIZE)
        ds_scalar = itd.make_epoch_dataset(
            self.entries, epoch=0, run_seed=42, cache_dir=self.cache_dir,
            racaf_cache_dir=self.racaf_cache_dir, batch_size=2, augment=False,
            image_size=self.IMAGE_SIZE)
        (k5, k6, kk), klabel = ds_kappa.element_spec
        (s5, s6, sr), slabel = ds_scalar.element_spec
        self.assertEqual(k5.shape, s5.shape)
        self.assertEqual(k6.shape, s6.shape)
        self.assertEqual(klabel.shape, slabel.shape)
        self.assertEqual(tuple(sr.shape), (None,))          # existing: scalar reliability
        self.assertEqual(tuple(kk.shape), (None, 4))        # C3: the (4,) kappa vector

    def test_epoch_dataset_yields_correct_kappa_in_the_existing_order(self):
        ds = itdk.make_epoch_dataset_kappa(
            self.entries, epoch=7, run_seed=123, cache_dir=self.cache_dir,
            racaf_cache_dir=self.racaf_cache_dir, batch_size=1, augment=True,
            image_size=self.IMAGE_SIZE)
        expected_order = itd.epoch_training_order(self.entries, 123, 7)
        seen = []
        for (_s5, _s6, kappa), grade in ds.as_numpy_iterator():
            seen.append((kappa[0].copy(), int(grade[0])))
        self.assertEqual(len(seen), len(expected_order))
        for (kappa, grade), (id_code, expected_grade) in zip(seen, expected_order):
            self.assertEqual(grade, expected_grade)
            np.testing.assert_array_equal(kappa, self.kappas[id_code])


_C3_INIT_SCRIPT = """
import os, sys
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "-1")
sys.path.insert(0, {repo_root!r})
import numpy as np
import tensorflow as tf
tf.keras.utils.set_random_seed({seed})
import racaf_c3_kappa_fusion_model as c3m
model = c3m.build_c3_joint_model_matched_init() if {matched} else c3m.build_c3_joint_model()
corn_layer = model.get_layer("corn")
np.save({kernel_path!r}, corn_layer.get_layer("corn_logits").kernel.numpy())
np.save({bias_path!r}, corn_layer.get_layer("corn_logits").bias.numpy())
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
np.save({kernel_path!r}, corn_layer.get_layer("corn_logits").kernel.numpy())
np.save({bias_path!r}, corn_layer.get_layer("corn_logits").bias.numpy())
"""


def _run_build_in_subprocess(script_template, tmp_dir, tag, seed, matched=None):
    kernel_path = os.path.join(tmp_dir, f"{tag}_kernel.npy")
    bias_path = os.path.join(tmp_dir, f"{tag}_bias.npy")
    script = script_template.format(repo_root=REPO_ROOT, seed=seed, matched=matched,
                                    kernel_path=kernel_path, bias_path=bias_path)
    completed = subprocess.run([sys.executable, "-c", script], capture_output=True, text=True,
                               cwd=REPO_ROOT, timeout=900)
    if completed.returncode != 0:
        raise RuntimeError(f"subprocess failed:\nSTDOUT:\n{completed.stdout}\nSTDERR:\n{completed.stderr}")
    return np.load(kernel_path), np.load(bias_path)


@unittest.skipUnless(os.environ.get("RUN_SLOW_TESTS", "1") != "0",
                     "set RUN_SLOW_TESTS=0 to skip the isolated-process init-parity test")
class CrossArmInitializationParityTests(unittest.TestCase):
    """C3's matched-init builder must leave Stage 05/06/07 and CORN at RACAF's own initial
    weights, so seed 42 means the same starting point in every arm. The plain builder must NOT
    have that parity, or the test would be vacuous -- gamma is zero-initialised and consumes no
    RNG, so the plain build would otherwise place CORN two draws early."""

    def test_matched_init_gives_identical_corn_initial_weights(self):
        with tempfile.TemporaryDirectory() as tmp:
            racaf_kernel, racaf_bias = _run_build_in_subprocess(
                _RACAF_INIT_SCRIPT, tmp, "racaf", seed=777)
            matched_kernel, matched_bias = _run_build_in_subprocess(
                _C3_INIT_SCRIPT, tmp, "matched", seed=777, matched=True)
        np.testing.assert_array_equal(racaf_kernel, matched_kernel)
        np.testing.assert_array_equal(racaf_bias, matched_bias)

    def test_unmatched_init_does_not_give_identical_corn_initial_weights(self):
        with tempfile.TemporaryDirectory() as tmp:
            racaf_kernel, _ = _run_build_in_subprocess(_RACAF_INIT_SCRIPT, tmp, "racaf2", seed=778)
            plain_kernel, _ = _run_build_in_subprocess(
                _C3_INIT_SCRIPT, tmp, "plain", seed=778, matched=False)
        self.assertFalse(np.array_equal(racaf_kernel, plain_kernel))


_MIXED_PRECISION_VERIFY_SCRIPT = """
import os, sys
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "-1")
sys.path.insert(0, {repo_root!r})
import tensorflow as tf
tf.keras.mixed_precision.set_global_policy("mixed_float16")
import racaf_c3_kappa_fusion_model as c3m
tf.keras.utils.set_random_seed(5)
model = c3m.build_c3_joint_model_matched_init()
result = c3m.verify_c3_model(model)
print("VERIFY_OK", result["fusion_equals_e_at_init"], result["trainable_parameters"])
"""


@unittest.skipUnless(os.environ.get("RUN_SLOW_TESTS", "1") != "0",
                     "set RUN_SLOW_TESTS=0 to skip the isolated-process mixed-precision test")
class MixedPrecisionTests(unittest.TestCase):
    """The training run builds C3 under `mixed_float16` on the T4 and calls `verify_c3_model`
    before epoch 1. Every other test here runs float32, so this is the only place the training
    policy's fp16 rounding of E inside the fusion is exercised. Isolated in a subprocess so the
    global dtype policy cannot leak into other tests."""

    def test_verify_c3_model_passes_under_mixed_float16(self):
        script = _MIXED_PRECISION_VERIFY_SCRIPT.format(repo_root=REPO_ROOT)
        completed = subprocess.run([sys.executable, "-c", script], capture_output=True, text=True,
                                   cwd=REPO_ROOT, timeout=900)
        self.assertEqual(completed.returncode, 0,
                         f"verify_c3_model failed under mixed_float16:\n{completed.stderr[-2000:]}")
        self.assertIn("VERIFY_OK True 43339784", completed.stdout)


if __name__ == "__main__":
    unittest.main()
