"""CPU unit tests for pl_convnext.py (the protocol-locked P/PL experiment). The real ImageNet weights
are verified on Colab by the notebook's pre-flight; here a randomly initialised Keras ConvNeXt-Tiny
with its built-in preprocessing stands in as the 'pretrained reference'."""
import os
import tempfile
import unittest

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "-1")

import keras
import numpy as np
import pandas as pd

import pl_convnext as pl
from lesion_segmentation_dataset import LESION_CLASSES
import local_feature_extraction_dataset as lfed

SIZE = 64          # structure and parameter counts do not depend on the input size


def _reference():
    keras.utils.set_random_seed(7)
    from keras.applications import ConvNeXtTiny
    ref = ConvNeXtTiny(include_top=False, weights=None, include_preprocessing=True, pooling="avg",
                       input_shape=(SIZE, SIZE, 3), name=pl.BACKBONE_NAME)
    # give LayerScale / norms non-trivial values so copying mistakes would be visible
    for v in ref.weights:
        if "prestem" not in v.path:
            v.assign(np.random.default_rng(len(v.path)).normal(0, 0.05, v.shape).astype("float32")
                     + (1.0 if v.path.endswith("gamma") else 0.0))
    return ref, [np.asarray(v.numpy()) for v in ref.weights if "prestem" not in v.path]


REF, REF_ARRAYS = _reference()


def _inputs(n=2, seed=0):
    rng = np.random.default_rng(seed)
    s5 = rng.uniform(0, 1, (n, SIZE, SIZE, 8)).astype("float32")
    s6 = rng.uniform(0, 1, (n, 256, 256, 3)).astype("float32")
    rel = rng.uniform(0, 1, (n, 1)).astype("float32")
    return s5, s6, rel


class StructureTests(unittest.TestCase):
    def test_parameter_counts(self):
        for arm in ("P", "PL"):
            model = pl.build_pl_model(arm, 42, REF_ARRAYS, image_size=SIZE)
            backbone = model.get_layer(pl.BACKBONE_NAME)
            self.assertEqual(pl.trainable_parameter_count(backbone),
                             pl.EXPECTED_BACKBONE_PARAMETERS[arm])
            self.assertEqual(pl.trainable_parameter_count(model),
                             pl.EXPECTED_BACKBONE_PARAMETERS[arm] + pl.EXPECTED_HEAD_PARAMETERS)
        self.assertEqual(pl.EXPECTED_BACKBONE_PARAMETERS["PL"] - pl.EXPECTED_BACKBONE_PARAMETERS["P"],
                         4 * 4 * 5 * 96)

    def test_channel_order_matches_the_repository(self):
        self.assertEqual(tuple(LESION_CLASSES), pl.EXPECTED_LESION_CLASSES)
        self.assertEqual(lfed.NUM_CHANNELS, 8)
        self.assertEqual(pl.CHANNEL_ORDER, ("R", "G", "B", "vessel", "MA", "HE", "EX", "SE"))


class PretrainedCopyTests(unittest.TestCase):
    def setUp(self):
        self.p = pl.build_pl_model("P", 42, REF_ARRAYS, image_size=SIZE)
        self.pl_ = pl.build_pl_model("PL", 42, REF_ARRAYS, image_size=SIZE)

    def test_every_p_weight_equals_the_reference(self):
        for v, a in zip(self.p.get_layer(pl.BACKBONE_NAME).weights, REF_ARRAYS):
            np.testing.assert_array_equal(v.numpy(), a)

    def test_pl_stem_rgb_slice_exact_extra_channels_zero_rest_equal(self):
        k = pl.stem_kernel(self.pl_.get_layer(pl.BACKBONE_NAME)).numpy()
        kp = pl.stem_kernel(self.p.get_layer(pl.BACKBONE_NAME)).numpy()
        self.assertEqual(k.shape, (4, 4, 8, 96))
        np.testing.assert_array_equal(k[:, :, :3, :], kp)
        self.assertFalse(k[:, :, 3:, :].any())
        for v, a in zip(self.pl_.get_layer(pl.BACKBONE_NAME).weights, REF_ARRAYS):
            if v.shape != a.shape:
                continue
            np.testing.assert_array_equal(v.numpy(), a)          # includes the first-layer bias
        self.assertEqual(len(self.pl_.pl_copy_report["expanded"]), 1)

    def test_p_matches_keras_preprocessed_reference(self):
        s5, _, _ = _inputs()
        adapter = self.p.get_layer("channel_adapter_p")
        ours = self.p.get_layer(pl.BACKBONE_NAME)(adapter(s5)).numpy()
        keras_path = REF(s5[..., :3] * 255.0).numpy()
        self.assertLess(float(np.max(np.abs(ours - keras_path))), pl.REFERENCE_PARITY_TOL)

    def test_initial_pl_equals_p(self):
        s5, s6, rel = _inputs()
        diff = np.max(np.abs(self.pl_.predict([s5, s6, rel], verbose=0)
                             - self.p.predict([s5, s6, rel], verbose=0)))
        self.assertLessEqual(float(diff), pl.PL_P_EQUIVALENCE_TOL)

    def test_auxiliary_inputs_are_inert(self):
        s5, s6, rel = _inputs()
        a = self.pl_.predict([s5, s6, rel], verbose=0)
        b = self.pl_.predict([s5, s6 * 0 + 9.0, rel * 0 - 3.0], verbose=0)
        np.testing.assert_array_equal(a, b)

    def test_p_ignores_the_prior_channels(self):
        s5, s6, rel = _inputs()
        s5b = s5.copy()
        s5b[..., 3:] = np.random.default_rng(3).uniform(0, 1, s5b[..., 3:].shape)
        np.testing.assert_array_equal(self.p.predict([s5, s6, rel], verbose=0),
                                      self.p.predict([s5b, s6, rel], verbose=0))


class HeadAndOptimizerTests(unittest.TestCase):
    def test_identical_head_initialisation_per_seed(self):
        for seed in pl.SEEDS:
            k, b = pl.corn_head_initial_weights(seed)
            for arm in ("P", "PL"):
                model = pl.build_pl_model(arm, seed, REF_ARRAYS, image_size=SIZE)
                kk, bb = model.get_layer("corn").get_layer("corn_logits").get_weights()
                np.testing.assert_array_equal(kk, k)
                np.testing.assert_array_equal(bb, b)
        self.assertFalse(np.array_equal(pl.corn_head_initial_weights(42)[0],
                                        pl.corn_head_initial_weights(123)[0]))

    def test_weight_decay_exempts_exactly_the_one_d_parameters(self):
        for arm in ("P", "PL"):
            model = pl.compile_pl_model(pl.build_pl_model(arm, 42, REF_ARRAYS, image_size=SIZE),
                                        [1.0] * 5, 1e-4, 0.05)
            report = pl.weight_decay_report(model)
            self.assertTrue(report["ok"], report)
            self.assertEqual(report["layer_scale"], pl.EXPECTED_LAYER_SCALE_VARIABLES)
            self.assertEqual(report["layer_scale_exempt"], pl.EXPECTED_LAYER_SCALE_VARIABLES)

    def test_one_training_step_runs_and_changes_prior_filters_only_through_gradients(self):
        model = pl.compile_pl_model(pl.build_pl_model("PL", 42, REF_ARRAYS, image_size=SIZE),
                                    [1.0] * 5, 1e-3, 0.05)
        s5, s6, rel = _inputs(4)
        model.fit([s5, s6, rel], np.array([0, 2, 3, 4]), epochs=1, batch_size=2, verbose=0)
        k = pl.stem_kernel(model.get_layer(pl.BACKBONE_NAME)).numpy()
        self.assertTrue(np.isfinite(k).all())
        self.assertTrue(k[:, :, 3:, :].any())              # prior filters learn from zero


class DecisionAndMetricTests(unittest.TestCase):
    OK = {"qwk": 0.0, "grade3_recall": 0.0, "false_urgent_rate": 0.0}

    def test_decision_rule(self):
        self.assertEqual(pl.decide([0.04, 0.03, 0.05], self.OK)["verdict"], "SUPPORTIVE")
        self.assertEqual(pl.decide([0.015, 0.001, 0.01], self.OK)["verdict"], "NOT_SUPPORTIVE")
        self.assertEqual(pl.decide([0.2, -0.01, 0.0], self.OK)["verdict"], "NOT_SUPPORTIVE")
        self.assertEqual(pl.decide([0.02, 0.01, 0.03], self.OK)["verdict"], "INCONCLUSIVE")
        bad = dict(self.OK, qwk=-0.021)                     # primary met, guardrail fails
        self.assertEqual(pl.decide([0.04, 0.03, 0.05], bad)["verdict"], "INCONCLUSIVE")
        edge = {"qwk": -0.02, "grade3_recall": -0.10, "false_urgent_rate": 0.02}
        self.assertEqual(pl.decide([0.03, 0.03, 0.03], edge)["verdict"], "SUPPORTIVE")

    def test_sensitivity_at_95_specificity(self):
        grades = np.array([0] * 100 + [4] * 10)
        scores = np.concatenate([np.linspace(0, 1, 100), np.linspace(0.9, 1.1, 10)])
        out = pl.sensitivity_at_specificity(grades, scores)
        self.assertGreaterEqual(out["specificity_achieved"], 0.95)
        self.assertEqual(out["sensitivity"], float(np.mean(scores[100:] > out["threshold"])))

    def test_run_metrics_from_a_per_sample_frame(self):
        rng = np.random.default_rng(0)
        g = np.array([0] * 40 + [1] * 10 + [2] * 20 + [3] * 8 + [4] * 12)
        frame = pd.DataFrame({"true_grade": g, "predicted_grade": np.clip(g + rng.integers(-1, 2, g.size), 0, 4),
                              "p_gt_2": rng.uniform(size=g.size) * 0.5 + (g >= 3) * 0.5,
                              "p_gt_3": rng.uniform(size=g.size) * 0.5 + (g == 4) * 0.5})
        m = pl.run_metrics(frame)
        for key in ("auroc_ge3_g4_vs_g012", "grade4_sensitivity_at_95_specificity", "qwk", "mae",
                    "grade3_recall", "false_urgent_rate"):
            self.assertIn(key, m)
        self.assertNotIn("persistent21_decoded_le2", m)


class PretrainedFileTests(unittest.TestCase):
    def test_drive_cache_is_reused_and_verified_without_network(self):
        with tempfile.TemporaryDirectory() as tmp:
            drive, local = os.path.join(tmp, "drive"), os.path.join(tmp, "local")
            os.makedirs(drive)
            fake = os.path.join(drive, pl.WEIGHTS_FILENAME)
            with open(fake, "wb") as fh:
                fh.write(b"pinned-weights")
            original_sha, original_net = pl.WEIGHTS_SHA256, pl.internet_reachable
            pl.WEIGHTS_SHA256 = pl.sha256_file(fake)
            pl.internet_reachable = lambda *a, **k: self.fail("must not touch the network")
            try:
                record = pl.ensure_pretrained_weights(drive, local)
                self.assertEqual(record["obtained_via"], "drive_cache")
                record = pl.ensure_pretrained_weights(drive, local)
                self.assertEqual(record["obtained_via"], "local_runtime_copy")
                with open(os.path.join(local, pl.WEIGHTS_FILENAME), "wb") as fh:
                    fh.write(b"tampered")
                with open(fake, "wb") as fh:
                    fh.write(b"tampered")
                pl.internet_reachable = lambda *a, **k: False
                with self.assertRaises(RuntimeError):
                    pl.ensure_pretrained_weights(drive, local)
            finally:
                pl.WEIGHTS_SHA256, pl.internet_reachable = original_sha, original_net

    def test_directory_fingerprint_detects_a_write(self):
        with tempfile.TemporaryDirectory() as tmp:
            with open(os.path.join(tmp, "a.bin"), "wb") as fh:
                fh.write(b"x")
            before = pl.directory_fingerprint([tmp])
            with open(os.path.join(tmp, "a.bin"), "ab") as fh:
                fh.write(b"yy")
            self.assertNotEqual(pl.directory_fingerprint([tmp]), before)


if __name__ == "__main__":
    unittest.main()
