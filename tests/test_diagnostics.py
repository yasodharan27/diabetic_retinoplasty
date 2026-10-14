"""CPU tests for the two inference-only diagnostics (records 89-92): the crop rule and frame pair of
geometry_diagnostic, and the decoding, calibration and prior-shift helpers of operating_point_diagnostic. Synthetic
images and logits only; no model and no dataset image."""
import os
import tempfile
import unittest

import numpy as np

import geometry_diagnostic as gd
import operating_point_diagnostic as op


def synthetic_fundus(path, h=300, w=450, margin=60):
    import cv2
    image = np.zeros((h, w, 3), np.uint8)
    yy, xx = np.mgrid[:h, :w]
    disc = (yy - h / 2) ** 2 + (xx - w / 2) ** 2 < (h / 2 - 5) ** 2
    image[disc] = (40, 90, 160)
    cv2.imwrite(path, image)
    return disc


class GeometryTests(unittest.TestCase):
    def test_the_rule_is_the_recorded_one(self):
        self.assertEqual((gd.GRAY_THRESHOLD, gd.LINE_FRACTION, gd.MIN_SIDE_FRACTION), (12, 0.02, 0.10))
        self.assertEqual((gd.MIN_EFFECT, gd.APTOS_GUARDRAIL, gd.PRIMARY_SET), (0.02, -0.01, "eyepacs_val"))

    def test_the_box_cuts_the_dark_margins_and_leaves_a_full_frame_alone(self):
        import cv2
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "a.png")
            disc = synthetic_fundus(path)
            top, bottom, left, right = gd.fundus_box(cv2.imread(path))
            cols = np.flatnonzero(disc.any(axis=0))
            self.assertLessEqual(abs(left - cols[0]), 3)
            self.assertLessEqual(abs(right - 1 - cols[-1]), 3)
            full = np.full((100, 150, 3), 120, np.uint8)
            self.assertEqual(gd.fundus_box(full), (0, 100, 0, 150))
            self.assertEqual(gd.fundus_box(np.zeros((100, 150, 3), np.uint8)), (0, 100, 0, 150))     # no fundus: no cut
            sliver = np.zeros((100, 150, 3), np.uint8)
            sliver[:, 70:75] = 200                                                                     # narrower than 10 %: no cut
            self.assertEqual(gd.fundus_box(sliver), (0, 100, 0, 150))

    def test_two_frames_share_stage_2_and_differ_only_in_the_cut(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "a.png")
            synthetic_fundus(path)
            direct, crop, box, native = gd.two_frames(path)
            self.assertEqual((direct.shape, crop.shape, direct.dtype), ((512, 512, 3), (512, 512, 3), np.float32))
            self.assertEqual(native, (300, 450))
            self.assertLess(box[3] - box[2], 450)                       # the sides were cut
            self.assertGreater(float((crop.max(axis=2) > 0.05).mean()), float((direct.max(axis=2) > 0.05).mean()))   # less dark area


class OperatingPointTests(unittest.TestCase):
    def setUp(self):
        rng = np.random.default_rng(0)
        self.grades = rng.integers(0, 5, 600)
        self.logits = (self.grades[:, None] - np.arange(4)[None, :] - 0.5) * 2.0 + rng.normal(0, 1.0, (600, 4))

    def test_rule_a_is_the_corn_decode(self):
        import corn
        p = op.cumulative(self.logits)
        self.assertTrue(np.array_equal(op.decode(p), corn.decode_logits(self.logits)["predicted_grade"]))
        cp = op.class_probabilities(p)
        self.assertTrue(np.allclose(cp.sum(axis=1), 1.0, atol=1e-6))

    def test_temperature_recovers_a_known_scale(self):
        t = op.fit_temperature(self.logits, self.grades)
        self.assertAlmostEqual(op.fit_temperature(self.logits * 3.0, self.grades) / t, 3.0, delta=0.06)     # scaling the logits scales the temperature
        self.assertAlmostEqual(op.fit_temperature(self.logits / 3.0, self.grades) / t, 1 / 3, delta=0.01)

    def test_thresholds_are_on_the_grid_and_do_not_lower_qwk(self):
        import arch1_posthoc as ph
        p = op.cumulative(self.logits - 1.0)                            # shifted scores: 0.5 is a poor threshold
        tau = op.fit_thresholds(p, self.grades)
        self.assertTrue(all(any(abs(t - g) < 1e-9 for g in list(op.GRID) + [0.5]) for t in tau))
        self.assertGreaterEqual(ph._fast_qwk(self.grades, op.decode(p, tau)), ph._fast_qwk(self.grades, op.decode(p)))

    def test_em_recovers_a_prior_shift(self):
        rng = np.random.default_rng(1)
        source = np.full(5, 0.2)
        posterior = rng.dirichlet(np.ones(5) * 0.5, size=4000)
        target_prior, adjusted = op.em_priors(posterior, source)
        self.assertTrue(np.isclose(target_prior.sum(), 1.0) and np.allclose(adjusted.sum(axis=1), 1.0))
        self.assertTrue(np.allclose(op.em_priors(posterior, posterior.mean(axis=0))[0], posterior.mean(axis=0), atol=0.05))

    def test_folds_are_stratified_and_keep_patients_together(self):
        folds = op.stratified_folds(self.grades)
        self.assertEqual(sorted(set(folds.tolist())), list(range(op.FOLDS)))
        for g in range(5):
            counts = np.bincount(folds[self.grades == g], minlength=op.FOLDS)
            self.assertLessEqual(counts.max() - counts.min(), 1)
        patients = np.repeat(np.arange(300), 2)
        grouped = op.stratified_folds(self.grades, patients)
        self.assertTrue(all(len(set(grouped[patients == p].tolist())) == 1 for p in range(300)))

    def test_describe_reports_every_quantity(self):
        p = op.cumulative(self.logits)
        out = op.describe(self.grades, op.decode(p), op.class_probabilities(p))
        self.assertEqual(sorted(out), sorted(["qwk", "recall", "sensitivity", "specificity", "ece", "decoded_counts"]))
        self.assertEqual(sum(out["decoded_counts"]), 600)


if __name__ == "__main__":
    unittest.main()
