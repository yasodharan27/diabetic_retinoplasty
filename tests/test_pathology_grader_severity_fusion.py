"""CPU tests for pathology_grader_severity_fusion.py: the locked rule's arithmetic, its ordinal validity, the
table checks that gate the run, and the evaluation on synthetic tables."""
import os
import unittest

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "-1")

import numpy as np
import pandas as pd

import arch1_posthoc as ph
import arch1_train as at
import pathology_grader_fusion as pf
import pathology_grader_severity_fusion as sf

GRADES = np.array([0] * 361 + [1] * 74 + [2] * 198 + [3] * 39 + [4] * 58)
IDS = [f"{k:012x}" for k in range(len(GRADES))]


def table(seed, quality=2.0):
    rng = np.random.default_rng(seed)
    logits = (GRADES[:, None] - np.arange(4)[None, :] - 0.5) * quality + rng.normal(0, 1.0, (len(GRADES), 4))
    return ph.table_arrays(pd.DataFrame(at.metrics_from_logits(IDS, GRADES, logits)[1]))


class RuleTests(unittest.TestCase):
    def test_locked_formula(self):
        self.assertEqual((sf.RULE["fused_0"], sf.RULE["fused_1"], sf.RULE["fused_2"], sf.RULE["fused_3"]),
                         ("0.5 * P_0 + 0.5 * Path_0", "0.5 * P_1 + 0.5 * Path_1", "0.5 * P_2 + 0.5 * Path_2", "min(P_3, fused_2)"))
        self.assertIs(sf.RULE["tuned_on_validation"], False)
        self.assertIs(sf.RULE["learned"], False)
        ids, g = np.array(["a", "b", "c", "d"]), np.array([4, 4, 2, 4])
        p = {"ids": ids, "grade": g, "pred": np.zeros(4, int),
             "p_gt": np.array([[1.0, 0.9, 0.8, 0.7], [1.0, 0.9, 0.4, 0.35], [0.9, 0.8, 0.2, 0.1], [1.0, 1.0, 0.9, 0.9]])}
        q = dict(p, p_gt=np.array([[0.9, 0.7, 0.6, 0.1], [0.8, 0.5, 0.2, 0.0], [0.7, 0.6, 0.6, 0.5], [1.0, 0.9, 0.5, 0.0]]))
        fused = sf.fuse_severity(p, q)
        np.testing.assert_allclose(fused["p_gt"], [[0.95, 0.8, 0.7, 0.7], [0.9, 0.7, 0.3, 0.3], [0.8, 0.7, 0.4, 0.1],
                                                   [1.0, 0.95, 0.7, 0.7]])
        np.testing.assert_array_equal(fused["pred"], [4, 2, 2, 4])
        # Row 0 and row 3: the pathology branch says "not >=4" (0.1, 0.0) and the RGB decision stands.
        np.testing.assert_array_equal(pf.fuse(p, q)["pred"][[0, 3]], [3, 3])       # the recorded 50 / 50 loses them
        # Row 3: the cap, not P_3, sets fused_3 (0.9 -> 0.7) -- the vector stays ordinal.
        self.assertEqual(fused["p_gt"][3, 3], fused["p_gt"][3, 2])

    def test_properties_on_realistic_tables(self):
        p, q = table(1), table(2, quality=1.2)
        fused, old = sf.fuse_severity(p, q), pf.fuse(p, q)
        np.testing.assert_allclose(fused["p_gt"][:, :3], old["p_gt"][:, :3])       # >=1, >=2, >=3 are the 50 / 50 fusion
        np.testing.assert_allclose(fused["p_gt"][:, 3], np.minimum(p["p_gt"][:, 3], fused["p_gt"][:, 2]))
        self.assertTrue(np.all(np.diff(fused["p_gt"], axis=1) <= 1e-12))           # a valid cumulative vector
        np.testing.assert_array_equal(fused["pred"], (fused["p_gt"] > 0.5).sum(1))
        self.assertTrue(np.all(fused["p_gt"][:, 3] <= p["p_gt"][:, 3] + 1e-12))    # never above the RGB probability
        shuffled = dict(q, p_gt=q["p_gt"].copy())
        shuffled["p_gt"][:, 3] = 0.0                                              # whatever the branch says about >=4 ...
        np.testing.assert_array_equal(sf.fuse_severity(p, shuffled)["p_gt"], fused["p_gt"])   # ... is not used
        same = sf.fuse_severity(p, p)
        np.testing.assert_allclose(same["p_gt"], p["p_gt"])                        # fusing a model with itself is itself
        with self.assertRaises(RuntimeError):
            sf.fuse_severity(p, {k: v[::-1] for k, v in q.items()})

    def test_table_checks(self):
        t = table(3)
        self.assertTrue(sf.check_table(t, "t", IDS))
        with self.assertRaises(RuntimeError):
            sf.check_table(t, "t", IDS[::-1])                                     # wrong order
        with self.assertRaises(RuntimeError):
            sf.check_table(dict(t, pred=np.roll(t["pred"], 1)), "t", IDS)         # grades not the decode of p_gt
        bad = dict(t, p_gt=t["p_gt"][:, ::-1].copy())
        with self.assertRaises(RuntimeError):
            sf.check_table(bad, "t", IDS)                                         # not cumulative
        with self.assertRaises(RuntimeError):
            sf.check_table({k: v[:-1] for k, v in t.items()}, "t", IDS[:-1])      # not 730 images
        with self.assertRaises(RuntimeError):
            sf.check_table(t, "t", IDS, logits=np.zeros((len(IDS), 4)))           # p_gt is not cumprod(sigmoid(logits))


class EvaluationTests(unittest.TestCase):
    def test_evaluation_on_synthetic_tables(self):
        tables = {"p": {s: table(s) for s in sf.SEEDS}, "pathology": {s: table(100 + s, 1.2) for s in sf.SEEDS},
                  "shuffled": {s: {"lesions": table(200 + s, 0.2), "vessel": table(100 + s, 1.2)} for s in sf.SEEDS}}
        r = sf.evaluate(tables, GRADES, n_boot=20)
        for s in sf.SEEDS:
            ps = r["per_seed"][s]
            self.assertEqual(set(ps["models"]), {"p", "pathology", "fusion_5050", "severity", "control_severity"})
            want = ph.metrics(sf.fuse_severity(tables["p"][s], tables["pathology"][s]))
            self.assertAlmostEqual(ps["models"]["severity"]["qwk"], want["qwk"], places=12)
            control = ph.metrics(sf.fuse_severity(tables["p"][s], tables["p"][pf.control_seed(s)]))
            self.assertAlmostEqual(ps["models"]["control_severity"]["qwk"], control["qwk"], places=12)
            self.assertAlmostEqual(ps["models"]["fusion_5050"]["qwk"], ph.metrics(pf.fuse(tables["p"][s], tables["pathology"][s]))["qwk"], places=12)
            self.assertAlmostEqual(ps["delta"]["severity_minus_p"]["qwk"]["delta"], want["qwk"] - ps["models"]["p"]["qwk"], places=12)
            self.assertIn("ci_low", ps["delta"]["severity_minus_5050"]["qwk"])
            self.assertEqual(ps["shuffle"]["vessel"]["severity_dqwk"], 0.0)        # the "shuffled" table is the intact one
            self.assertLess(ps["shuffle"]["lesions"]["severity_dqwk"], 0.0)
            self.assertEqual(set(ps["matched_seed"]["checks"]), set(ph.TOLERANCE))
        s = r["summary"]
        x = [r["per_seed"][k]["models"]["severity"]["qwk"] for k in sf.SEEDS]
        self.assertAlmostEqual(s["models"]["severity"]["qwk"]["mean"], np.mean(x), places=12)
        self.assertAlmostEqual(s["models"]["severity"]["qwk"]["sd"], np.std(x, ddof=1), places=12)
        self.assertEqual(set(s["delta"]), {"severity_minus_p", "severity_minus_5050", "severity_minus_control"})
        self.assertIn("ci_low", s["delta"]["severity_minus_p"]["qwk"])
        text = sf.markdown(r)
        for needle in ("fused_3 = min(P_3, fused_2)", "Descriptive and post hoc", "not independent", "No other fusion rule was evaluated",
                       "IDRiD grading test is untouched", "severity − recorded 50 / 50", "descriptive only"):
            self.assertIn(needle, text)


if __name__ == "__main__":
    unittest.main()
