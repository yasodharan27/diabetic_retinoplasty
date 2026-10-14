"""CPU tests for explain.py and the test-time-augmentation uncertainty in uncertainty_eval.py. A randomly
initialised P-graph grader (real architecture, no trained checkpoint) and synthetic inputs only: no dataset image,
no result. Checked: output shapes, the CORN score being explained, numerical stability, reproducibility, the
Integrated-Gradients completeness property, exact Kernel SHAP on additive functions, and the localisation measures."""
import os
import tempfile
import unittest

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "-1")

import numpy as np

import explain as ex
import uncertainty_eval as ue
from test_eyepacs_adaptation_train import REF_ARRAYS


class ScoreTests(unittest.TestCase):
    def test_the_explained_score_is_the_corn_expected_grade_or_one_cumulative_cut(self):
        import corn
        import tensorflow as tf

        import gradcam
        z = np.random.default_rng(0).normal(0, 3, (20, 4))
        p_cum = corn.decode_logits(z)["p_cum"]
        self.assertTrue(np.allclose(ex.score(z), p_cum.sum(axis=1)))
        for k in range(4):
            self.assertTrue(np.allclose(ex.score(z, cut=k), p_cum[:, k]))
        self.assertTrue(np.allclose(np.asarray(gradcam.score_from_logits(tf.constant(z, tf.float32))), ex.score(z), atol=1e-5))
        # the expected grade of the class distribution is the sum of the cumulative probabilities
        probs = corn.decode_logits(z)["class_probabilities"]
        self.assertTrue(np.allclose((probs * np.arange(5)).sum(axis=1), ex.score(z), atol=1e-5))

    def test_extreme_logits_stay_finite(self):
        z = np.array([[60.0, 60.0, 60.0, 60.0], [-60.0, -60.0, -60.0, -60.0], [60.0, -60.0, 60.0, -60.0]])
        s = ex.score(z)
        self.assertTrue(np.all(np.isfinite(s)))
        self.assertTrue(np.allclose(s, [4.0, 0.0, 1.0], atol=1e-6))


class ModelExplanationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        import keras

        import gradcam
        import pl_convnext as pl
        cls.tmp = tempfile.TemporaryDirectory(ignore_cleanup_errors=True)
        keras.mixed_precision.set_global_policy("float32")
        p = pl.build_pl_model("P", 42, REF_ARRAYS)
        cls.weights = os.path.join(cls.tmp.name, "model.weights.h5")
        p.save_weights(cls.weights)
        cls.grad_model, _ = gradcam.load_explainer(42, cls.weights, REF_ARRAYS)
        cls.fn = staticmethod(ex.graph_model_fn(cls.grad_model, batch_size=2))
        yy, xx = np.mgrid[:512, :512]
        disc = ((yy - 256) ** 2 + (xx - 256) ** 2 < 230 ** 2)[..., None]
        cls.rgb = (np.random.default_rng(1).random((512, 512, 3)) * 0.6 + 0.2).astype(np.float32) * disc

    @classmethod
    def tearDownClass(cls):
        cls.tmp.cleanup()

    def test_integrated_gradients_is_complete_and_reproducible(self):
        attribution, gap, heat = ex.integrated_gradients(self.grad_model, self.rgb, steps=32, batch=2)
        s1, s0 = ex.score(self.fn(self.rgb[None]))[0], ex.score(self.fn(np.zeros_like(self.rgb)[None]))[0]
        self.assertEqual((attribution.shape, heat.shape), ((512, 512, 3), (512, 512)))
        self.assertTrue(np.all(np.isfinite(attribution)))
        self.assertLess(gap, 0.05 * abs(s1 - s0) + 1e-3)                  # completeness: attributions sum to s(x) - s(baseline)
        again = ex.integrated_gradients(self.grad_model, self.rgb, steps=32, batch=2)[0]
        self.assertTrue(np.allclose(attribution, again))
        self.assertTrue(np.all(heat[0, :] == 0) or float(heat[:5, :5].max()) < 1e-6)   # the black corner carries no attribution
        with self.assertRaises(ValueError):
            ex.integrated_gradients(self.grad_model, self.rgb[None])

    def test_occlusion_shape_sign_and_reproducibility(self):
        raw, heat = ex.occlusion(self.fn, self.rgb, patch=128, stride=128)
        again, _ = ex.occlusion(self.fn, self.rgb, patch=128, stride=128)
        self.assertEqual((raw.shape, heat.shape), ((512, 512), (512, 512)))
        self.assertTrue(np.allclose(raw, again, atol=1e-5))                 # CPU inference is reproducible to rounding, not bit for bit
        self.assertTrue(np.all(np.isfinite(raw)) and 0.0 <= heat.min() and heat.max() <= 1.0)
        yy, xx = np.mgrid[:512, :512]
        small = self.rgb * (((yy - 256) ** 2 + (xx - 256) ** 2) < 100 ** 2)[..., None]      # a disc clear of the corner patches
        corner = ex.occlusion(lambda x: np.tile(x.reshape(len(x), -1).mean(axis=1, keepdims=True) * 40 - 5, (1, 4)), small, patch=128, stride=128)[0]
        self.assertLess(abs(float(corner[0, 0])), 1e-9)                    # occluding a black corner changes nothing
        self.assertGreater(float(corner[256, 256]), 0.0)                   # occluding the bright disc lowers a brightness score

    def test_tta_on_the_grader(self):
        logits = ue.tta_logits(self.fn, self.rgb[None])
        self.assertEqual(logits.shape, (8, 1, 4))
        summary = ue.tta_summary(logits)
        self.assertEqual((summary["pred"].shape, summary["expected_grade_sd"].shape), ((1,), (1,)))
        self.assertTrue(np.isfinite(summary["expected_grade_sd"][0]))


class TtaTests(unittest.TestCase):
    def test_the_eight_transformations(self):
        x = np.arange(2 * 3 * 3 * 1, dtype=np.float32).reshape(2, 3, 3, 1)
        outs = [ue.dihedral(x, k) for k in range(8)]
        self.assertTrue(np.array_equal(outs[0], x))
        self.assertEqual(len({o.tobytes() for o in outs}), 8)
        self.assertTrue(all(o.shape == x.shape and o.sum() == x.sum() for o in outs))

    def test_an_orientation_free_model_has_no_spread(self):
        fn = lambda rgb: np.tile(rgb.reshape(len(rgb), -1).mean(axis=1, keepdims=True), (1, 4))      # noqa: E731
        summary = ue.tta_summary(ue.tta_logits(fn, np.random.default_rng(0).random((3, 8, 8, 3))))
        self.assertTrue(np.allclose(summary["expected_grade_sd"], 0.0) and not summary["grade_changes"].any())


class ShapTests(unittest.TestCase):
    def test_kernel_shap_is_exact_on_an_additive_function(self):
        w = np.array([0.5, -1.0, 2.0, 0.0, 1.5, -0.25])
        phi, base = ex.kernel_shap(lambda masks: masks @ w + 3.0, len(w), n_samples=256, seed=0)
        self.assertTrue(np.allclose(phi, w, atol=1e-8))
        self.assertAlmostEqual(base, 3.0)
        self.assertAlmostEqual(float(phi.sum()), float(w.sum()))          # efficiency

    def test_kernel_shap_splits_an_interaction_fairly_and_is_reproducible(self):
        value = lambda m: m[:, 0] * m[:, 1] * 2.0                             # noqa: E731
        phi, _ = ex.kernel_shap(value, 4, n_samples=400, seed=3)          # 14 coalitions: enumerated, exact
        self.assertTrue(np.allclose(phi, [1.0, 1.0, 0.0, 0.0], atol=1e-8))
        big = lambda m: m[:, 0] * m[:, 1] * 2.0 + m @ np.linspace(-1, 1, 12)   # noqa: E731
        phi12, _ = ex.kernel_shap(big, 12, n_samples=2000, seed=3)        # sampled (4,094 coalitions), paired
        self.assertTrue(np.allclose(phi12[:2] - np.linspace(-1, 1, 12)[:2], [1.0, 1.0], atol=0.1))
        self.assertTrue(np.allclose(phi12[2:], np.linspace(-1, 1, 12)[2:], atol=0.1))
        self.assertTrue(np.array_equal(phi12, ex.kernel_shap(big, 12, n_samples=2000, seed=3)[0]))

    def test_superpixel_shap_on_a_toy_model(self):
        rgb = np.zeros((64, 64, 3), np.float32)
        rgb[:, 32:] = 0.9
        fn = lambda x: np.tile(x[:, :, 32:].reshape(len(x), -1).mean(axis=1, keepdims=True) * 20 - 5, (1, 4))   # noqa: E731
        heat, segments, phi, base = ex.superpixel_shap(fn, rgb, n_segments=8, n_samples=128, batch=64)
        self.assertEqual((heat.shape, segments.shape), ((64, 64), (64, 64)))
        self.assertGreater(float(heat[:, 40:].mean()), float(heat[:, :20].mean()))         # credit goes to the bright half
        self.assertAlmostEqual(float(phi.sum() + base), float(ex.score(fn(rgb[None]))[0]), places=6)


class LocalisationTests(unittest.TestCase):
    def test_the_measures_and_their_uniform_baseline(self):
        mask = np.zeros((32, 32), bool)
        mask[4:8, 4:8] = True
        perfect = mask.astype(float)
        out = ex.localisation(perfect, mask, top_fraction=16 / 1024)
        self.assertEqual((out["pointing_hit"], out["top_precision"], out["auroc"]), (True, 1.0, 1.0))
        self.assertAlmostEqual(out["lesion_fraction"], 16 / 1024)
        off = ex.localisation(np.roll(perfect, 16, axis=0), mask)
        self.assertFalse(off["pointing_hit"])
        with self.assertRaises(ValueError):
            ex.localisation(perfect, np.zeros((8, 8), bool))

    def test_the_mask_follows_the_frame_geometry(self):
        native = np.zeros((300, 450), np.uint8)
        native[100:120, 200:220] = 255
        direct, cropped = ex.frame_mask(native, size=64), ex.frame_mask(native, box=(0, 300, 75, 375), size=64)
        self.assertEqual((direct.shape, cropped.shape, direct.dtype), ((64, 64), (64, 64), bool))
        cols = lambda m: np.flatnonzero(m.any(axis=0))                      # noqa: E731
        self.assertLessEqual(abs(int(cols(direct)[0]) - round(200 / 450 * 64)), 1)          # the full frame scales columns by 64 / 450
        self.assertLessEqual(abs(int(cols(cropped)[0]) - round((200 - 75) / 300 * 64)), 1)  # the cut shifts them by 75, then 64 / 300
        self.assertGreater(cropped.sum(), direct.sum())                    # and the lesion is larger in the cropped frame


if __name__ == "__main__":
    unittest.main()
