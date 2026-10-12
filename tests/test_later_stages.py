"""Light CPU tests for the later-stage files that need no model: the uncertainty analysis on synthetic tables, the
Grad-CAM helpers that are pure arithmetic, and the fixed settings of the external re-test. No TensorFlow model is
built here; the model-dependent paths (Grad-CAM on a grader, the demo, the re-test runs) are exercised on Colab."""
import os
import tempfile
import unittest

import numpy as np
import pandas as pd

import uncertainty_eval as ue


def seed_tables(tmp, n=400, noise=(0.8, 0.8, 0.8), seed=0):
    rng = np.random.default_rng(seed)
    grades = rng.integers(0, 5, n)
    hard = rng.random(n) < 0.3                                         # images every seed finds harder
    paths = {}
    for k, s in enumerate((42, 123, 2026)):
        logits = (grades[:, None] - np.arange(4)[None, :] - 0.5) * 3.0 + rng.normal(0, noise[k], (n, 4)) * np.where(hard, 4.0, 1.0)[:, None]
        frame = pd.DataFrame({"image_id": [f"i{j}" for j in range(n)], "true_grade": grades, **{f"logit_{c}": logits[:, c] for c in range(4)}})
        paths[s] = os.path.join(tmp, f"seed{s}.csv")
        frame.to_csv(paths[s], index=False)
    return paths, grades


class UncertaintyTests(unittest.TestCase):
    def test_the_ensemble_averages_probabilities_and_decodes_at_one_half(self):
        with tempfile.TemporaryDirectory() as tmp:
            paths, grades = seed_tables(tmp)
            ids, g, logits = ue.read_seed_tables(paths)
            e = ue.ensemble(logits)
            mean = np.mean([ue.cumulative(z) for z in logits], axis=0)
            self.assertTrue(np.array_equal(e["pred"], (mean > 0.5).sum(axis=1)))
            self.assertTrue(np.allclose(e["class_prob"].sum(axis=1), 1.0))
            self.assertEqual(e["seed_pred"].shape, (3, len(grades)))
            self.assertTrue(np.all(e["entropy"] >= 0) and np.all(e["seed_disagreement"] >= 0))
            self.assertTrue(np.array_equal(g, grades))

    def test_an_informative_score_improves_the_retained_set_and_beats_random_referral(self):
        with tempfile.TemporaryDirectory() as tmp:
            paths, _ = seed_tables(tmp)
            result, table = ue.evaluate(paths, "toy")
            curve = result["scores"]["entropy"]["selective"]
            self.assertEqual([r["coverage"] for r in curve], list(ue.COVERAGES))
            self.assertGreater(curve[-1]["accuracy"], curve[0]["accuracy"])          # half coverage is more accurate than full
            self.assertGreater(result["scores"]["entropy"]["error_detection_auroc"], 0.6)
            random_half = [r for r in result["random_referral"] if r["coverage"] == 0.5][0]
            self.assertGreater(curve[-1]["qwk"], random_half["qwk_mean"])
            self.assertEqual(curve[0]["n"], 400)
            self.assertEqual(sorted(result["scores"]), sorted(ue.SCORES))
            self.assertEqual(len(table), 400)
            self.assertTrue(0.0 <= result["seeds_disagree_on_grade"] <= 1.0)

    def test_tables_of_different_images_are_refused(self):
        with tempfile.TemporaryDirectory() as tmp:
            paths, _ = seed_tables(tmp)
            frame = pd.read_csv(paths[123])
            frame.iloc[::-1].to_csv(paths[123], index=False)
            with self.assertRaises(ValueError):
                ue.read_seed_tables(paths)


class GradCamHelperTests(unittest.TestCase):
    def test_lesion_agreement_and_overlay_are_plain_arithmetic(self):
        import gradcam
        cam = np.zeros((8, 8), np.float32)
        cam[2, 3] = 1.0
        cam[5, 5] = 0.5
        mask = np.zeros((8, 8), bool)
        mask[2, 3] = True
        hit = gradcam.lesion_agreement(cam, mask, top_fraction=1 / 64)
        self.assertEqual((hit["pointing_hit"], hit["top_fraction_precision"], hit["has_lesion"]), (True, 1.0, True))
        self.assertAlmostEqual(hit["lesion_fraction"], 1 / 64)
        self.assertFalse(gradcam.lesion_agreement(cam, np.roll(mask, 3, axis=0))["pointing_hit"])
        with self.assertRaises(ValueError):
            gradcam.lesion_agreement(cam, np.zeros((4, 4), bool))
        picture = gradcam.overlay(np.full((8, 8, 3), 0.5, np.float32), cam)
        self.assertEqual((picture.shape, picture.dtype), ((8, 8, 3), np.uint8))
        self.assertGreater(int(picture[2, 3, 0]), int(picture[0, 0, 0]))               # red where the map is high


if __name__ == "__main__":
    unittest.main()
