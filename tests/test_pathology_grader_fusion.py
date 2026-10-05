"""CPU tests for pathology_grader_fusion.py: the fixed fusion arithmetic, CORN decoding of cumulative
probabilities, alignment with the stored P predictions, the P + P control, and the seed / summary reports."""
import os
import unittest

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "-1")

import numpy as np
import pandas as pd

import arch1_posthoc as ph
import arch1_train as at
import corn
import pathology_grader_fusion as pf

REPO = os.path.dirname(os.path.abspath(pf.__file__))
REAL_RESULTS = os.path.join(REPO, "results")
GRADES = np.array([0] * 60 + [1] * 14 + [2] * 34 + [3] * 8 + [4] * 12)
IDS = [f"{k:012x}" for k in range(len(GRADES))]


def table(seed, quality=2.0, ids=IDS, grades=GRADES):
    rng = np.random.default_rng(seed)
    logits = (grades[:, None] - np.arange(4)[None, :] - 0.5) * quality + rng.normal(0, 1.0, (len(grades), 4))
    return ph.table_arrays(pd.DataFrame(at.metrics_from_logits(list(ids), grades, logits)[1]))


class RuleTests(unittest.TestCase):
    def test_the_rule_is_fixed_and_untuned(self):
        self.assertEqual(pf.FUSION["weights"], [0.5, 0.5])
        self.assertEqual(pf.FUSION["decode_threshold"], 0.5)
        self.assertIs(pf.FUSION["tuned_on_validation"], False)
        self.assertEqual(len(pf.FUSION["members"]), 2)
        self.assertEqual([pf.control_seed(s) for s in (42, 123, 2026)], [123, 2026, 42])

    def test_decode_is_corns_rule_on_cumulative_probabilities(self):
        logits = np.random.default_rng(0).normal(0, 3, (500, 4))
        decoded = corn.decode_logits(logits)
        np.testing.assert_array_equal(pf.decode(decoded["p_cum"].astype(np.float64)), decoded["predicted_grade"])
        np.testing.assert_array_equal(pf.decode([[0.9, 0.8, 0.7, 0.6], [0.4, 0.3, 0.2, 0.1], [0.9, 0.6, 0.5, 0.2]]), [4, 0, 2])

    def test_fusion_arithmetic(self):
        ids, g = np.array(["a", "b"]), np.array([1, 3])
        a = {"ids": ids, "grade": g, "pred": np.array([2, 4]), "p_gt": np.array([[0.9, 0.6, 0.2, 0.1], [1.0, 0.9, 0.8, 0.7]])}
        b = {"ids": ids, "grade": g, "pred": np.array([1, 2]), "p_gt": np.array([[0.7, 0.3, 0.1, 0.0], [0.9, 0.7, 0.3, 0.1]])}
        fused = pf.fuse(a, b)
        np.testing.assert_allclose(fused["p_gt"], [[0.8, 0.45, 0.15, 0.05], [0.95, 0.8, 0.55, 0.4]])
        np.testing.assert_array_equal(fused["pred"], [1, 3])                      # grade = #{k: fused_k > 0.5}
        np.testing.assert_array_equal(pf.fuse(b, a)["p_gt"], fused["p_gt"])       # symmetric
        same = pf.fuse(a, a)
        np.testing.assert_array_equal(same["p_gt"], a["p_gt"])
        self.assertTrue(np.all(np.diff(fused["p_gt"], axis=1) <= 1e-12))          # stays a valid cumulative vector
        x, y = table(1), table(2)
        f = pf.fuse(x, y)
        np.testing.assert_allclose(f["p_gt"], (x["p_gt"] + y["p_gt"]) / 2.0)
        np.testing.assert_array_equal(f["pred"], (f["p_gt"] > 0.5).sum(1))
        self.assertTrue(np.all(np.diff(f["p_gt"], axis=1) <= 1e-12))

    def test_alignment_with_p_is_enforced(self):
        a = table(1)
        reordered = {k: (v[::-1] if isinstance(v, np.ndarray) else v) for k, v in table(2).items()}
        with self.assertRaises(RuntimeError):
            pf.fuse(a, reordered)
        other_grades = dict(table(2), grade=np.roll(GRADES, 1))
        with self.assertRaises(RuntimeError):
            pf.fuse(a, other_grades)
        shorter = {k: v[:-1] for k, v in table(2).items()}
        with self.assertRaises(RuntimeError):
            pf.fuse(a, shorter)


class ReportTests(unittest.TestCase):
    def setUp(self):
        self.p = {s: table(s) for s in pf.SEEDS}
        self.path = {s: table(1000 + s, quality=1.2) for s in pf.SEEDS}
        self.shuffled = {s: {"lesions": table(2000 + s, quality=0.2), "vessel": self.path[s],
                             "MA": table(3000 + s, quality=1.0)} for s in pf.SEEDS}
        self.idx = ph.bootstrap_indices(GRADES, n_boot=30)

    def _report(self, s):
        return pf.seed_report(s, self.path[s], self.p, self.shuffled[s], self.idx)

    def test_seed_report(self):
        r = self._report(42)
        self.assertEqual((r["seed"], r["control_seed"], r["n"]), (42, 123, len(GRADES)))
        self.assertEqual(set(r["models"]), {"pathology", "p", "fused", "control"})
        fused = ph.metrics(pf.fuse(self.p[42], self.path[42]))
        control = ph.metrics(pf.fuse(self.p[42], self.p[123]))
        self.assertAlmostEqual(r["models"]["fused"]["qwk"], fused["qwk"], places=12)
        self.assertAlmostEqual(r["models"]["control"]["qwk"], control["qwk"], places=12)
        self.assertAlmostEqual(r["models"]["p"]["qwk"], ph.metrics(self.p[42])["qwk"], places=12)
        for key in ("qwk", "auroc_ge1", "auroc_ge2", "auroc_ge3", "auroc_ge4", "grade3_recall", "grade4_recall",
                    "false_urgent_rate"):
            self.assertIn(key, r["models"]["pathology"])
        self.assertAlmostEqual(r["models"]["pathology"]["grade4_recall"], ph.metrics(self.path[42])["recall_per_grade"][4])
        self.assertAlmostEqual(r["delta_vs_p"]["fused"]["qwk"]["delta"], fused["qwk"] - r["models"]["p"]["qwk"], places=12)
        self.assertIn("ci_low", r["delta_vs_p"]["fused"]["qwk"])
        self.assertNotIn("_draws", r["delta_vs_p"]["fused"]["qwk"])
        self.assertAlmostEqual(r["delta_fused_vs_control"]["qwk"]["delta"], fused["qwk"] - control["qwk"], places=12)
        self.assertEqual(r["pathology_shuffle"]["vessel"]["dqwk"], 0.0)            # the "shuffled" table is the intact one
        self.assertLess(r["pathology_shuffle"]["lesions"]["dqwk"], 0.0)
        self.assertEqual(set(r["fused_shuffle"]), {"lesions", "vessel"})           # per-class shuffles are not fused
        self.assertEqual(r["fused_shuffle"]["vessel"]["dqwk"], 0.0)
        expected = ph.metrics(pf.fuse(self.p[42], self.shuffled[42]["lesions"]))["qwk"] - fused["qwk"]
        self.assertAlmostEqual(r["matched_seed"]["lesion_shuffle_dqwk"], expected, places=12)
        self.assertEqual(r["matched_seed"]["lesion_dependence"], bool(expected <= -0.01 + 1e-12))
        self.assertEqual(r["matched_seed"]["reference"], "P-42")
        self.assertEqual(set(r["matched_seed"]["checks"]), set(ph.TOLERANCE))       # only the recorded tolerances

    def test_summary_mean_sd_and_markdown(self):
        reports = {s: self._report(s) for s in pf.SEEDS}
        summary = pf.summarise(reports)
        for name in ("pathology", "p", "fused", "control"):
            x = [reports[s]["models"][name]["qwk"] for s in pf.SEEDS]
            self.assertAlmostEqual(summary["models"][name]["qwk"]["mean"], np.mean(x), places=12)
            self.assertAlmostEqual(summary["models"][name]["qwk"]["sd"], np.std(x, ddof=1), places=12)
        x = [reports[s]["delta_vs_p"]["fused"]["qwk"]["delta"] for s in pf.SEEDS]
        self.assertAlmostEqual(summary["delta_vs_p"]["fused"]["qwk"]["mean"], np.mean(x), places=12)
        self.assertEqual(set(summary["pathology_shuffle_dqwk"]), {"MA", "lesions", "vessel"})
        self.assertEqual(summary["interpretation_fused_vs_p"]["category"], ph.interpret(reports)["category"])
        self.assertEqual(summary["fusion"], pf.FUSION)
        text = pf.summary_markdown(summary, reports)
        for needle in ("fixed before training and is not tuned", "P + P control", "pathology grader alone",
                       "AUROC ≥4", "grade-4 recall", "No new success threshold", "§54 category for the fused model"):
            self.assertIn(needle, text)
        with self.assertRaises(RuntimeError):
            pf.summarise({s: reports[s] for s in (42, 123)})

    @unittest.skipUnless(os.path.isdir(os.path.join(REAL_RESULTS, "PL_ConvNeXtPriors")), "stored P tables not copied locally")
    def test_stored_p_tables_are_found_aligned_and_fusable(self):
        tables = {s: ph._read_table(pf.p_prediction_path(REAL_RESULTS, s)) for s in pf.SEEDS}
        self.assertEqual(len(tables[42]["ids"]), 730)
        for s in pf.SEEDS:
            pf.assert_aligned(tables[42], tables[s], f"P-{s}")
            np.testing.assert_array_equal(pf.decode(tables[s]["p_gt"]), tables[s]["pred"])   # stored grades = the rule
        control = pf.fuse(tables[42], tables[123])
        self.assertTrue(0.85 < ph.metrics(control)["qwk"] < 1.0)
        with self.assertRaises(RuntimeError):
            pf.p_prediction_path(os.path.join(REAL_RESULTS, "nothing"), 42)


if __name__ == "__main__":
    unittest.main()
