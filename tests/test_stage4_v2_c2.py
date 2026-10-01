"""CPU tests for stage4_v2_c2.py on small synthetic data (mechanics only -- no C2 result is produced)."""
import json
import os
import tempfile
import unittest

import numpy as np

import pipeline_v2_config as v2cfg
import stage34_cache_v2 as cache
import stage4_v2_aptos_cache as ac
import stage4_v2_c2 as c2

CHANNELS = cache.channel_names(v2cfg.STAGE4_V2A_CLASSES)
Q_NAMES = cache.pyramid_feature_names(CHANNELS)


def synthetic(n=150, seed=0):
    rng = np.random.default_rng(seed)
    grades = np.repeat(np.arange(5), n // 5)
    R = rng.normal(size=(len(grades), 80)) + 0.4 * grades[:, None] * (rng.random(80) > 0.7)
    Q = rng.normal(size=(len(grades), len(Q_NAMES))) + 0.4 * grades[:, None] * (rng.random(len(Q_NAMES)) > 0.8)
    V = rng.normal(size=(len(grades), 42))
    return R, Q, V, grades


class DesignTests(unittest.TestCase):
    def test_feature_dimensions_and_capacity(self):
        self.assertEqual(len(Q_NAMES), 8 * 2 * 21)              # Q-pyr = 2K x {mean,max} x 21 cells
        self.assertEqual(len(cache.pyramid_feature_names(("vessel",))), 42)
        dims = {"R": 768, "Q": 336, "V": 42}
        for arm in c2.ARMS:
            keys, blocks = c2.arm_design(arm, dims)
            self.assertEqual(sum(n for _, n in blocks), v2cfg.C2_K_TOTAL)      # capacity matched
        self.assertEqual([n for _, n in c2.arm_design("RQ", dims)[1]], [32, 32])
        R, Q, V, g = synthetic()
        keys, blocks = c2.arm_design("RQ", {"R": 80, "Q": Q.shape[1], "V": 42})
        x = np.concatenate([R, Q], 1)
        probe = c2.make_probe(blocks, 0).fit(x, (g >= 2).astype(int))
        self.assertEqual(probe.named_steps["red"].transform(x).shape[1], 64)

    def test_standardisation_fitted_on_training_rows_only(self):
        R, Q, V, g = synthetic()
        keys, blocks = c2.arm_design("R", {"R": 80, "Q": 1, "V": 1})
        train = np.arange(0, len(g), 2)
        probe = c2.make_probe(blocks, 0).fit(R[train], (g[train] >= 2).astype(int))
        np.testing.assert_allclose(probe.named_steps["red"].named_steps["s"].mean_, R[train].mean(0))
        self.assertFalse(np.allclose(probe.named_steps["red"].named_steps["s"].mean_, R.mean(0)))

    def test_loco_drops_both_channels_of_a_class(self):
        cols = c2.loco_columns(Q_NAMES, "HE")
        self.assertEqual(len(cols), len(Q_NAMES) * 3 // 4)
        self.assertFalse(any(Q_NAMES[k].startswith("HE:") for k in cols))


class StatisticsTests(unittest.TestCase):
    def test_bootstrap_deterministic_and_grade_stratified(self):
        g = np.repeat(np.arange(5), [50, 10, 30, 6, 8])
        a = c2.stratified_bootstrap_indices(g, 20, seed=1)
        b = c2.stratified_bootstrap_indices(g, 20, seed=1)
        for x, y in zip(a, b):
            np.testing.assert_array_equal(x, y)
            self.assertEqual(np.bincount(g[x], minlength=5).tolist(), [50, 10, 30, 6, 8])
        self.assertFalse(np.array_equal(a[0], c2.stratified_bootstrap_indices(g, 20, seed=2)[0]))

    def test_decision_rule_is_the_preregistered_one(self):
        self.assertEqual(v2cfg.C2_MIN_DELTA, 0.005)
        self.assertTrue(c2.decide(0.006, 0.001, 0.80, 0.79)["PASS"])
        self.assertFalse(c2.decide(0.004, 0.001, 0.80, 0.79)["PASS"])     # delta too small
        self.assertFalse(c2.decide(0.010, -0.001, 0.80, 0.79)["PASS"])    # CI includes 0
        self.assertFalse(c2.decide(0.010, 0.002, 0.80, 0.81)["PASS"])     # substitution (R+Q < Q)

    def test_run_probe_deterministic_with_per_cut_results(self):
        R, Q, V, g = synthetic()
        kw = dict(loco=True, n_repeats=2, n_boot=50, log=lambda *a: None)
        a = c2.run_probe(R, Q, V, g, Q_NAMES, **kw)
        b = c2.run_probe(R, Q, V, g, Q_NAMES, **kw)
        self.assertEqual(json.dumps(a["arms"], sort_keys=True), json.dumps(b["arms"], sort_keys=True))
        self.assertEqual(set(a["arms"]["RQ"]["per_cut"]), {">=1", ">=2", ">=3", ">=4"})
        self.assertEqual(set(a["loco_RQ_minus_RQc"]), set(v2cfg.STAGE4_V2A_CLASSES))
        self.assertIn("PASS", a["decision"])
        self.assertGreater(a["arms"]["R"]["mean_auroc"], 0.6)          # planted signal is found
        md = c2.report_markdown(a, {"stage4_sha256": "b" * 64, "stage4_generation": "s4v2-bbbbbbbbbbbb-K4",
                                    "stage3_sha256": v2cfg.STAGE3_LWNET_SHA256, "n_repeats": 2, "n_boot": 50})
        self.assertIn("C2 PASS" if a["decision"]["PASS"] else "C2 FAIL", md)


class IsolationTests(unittest.TestCase):
    def test_validation_and_missing_ids_refused(self):
        ids = ["a", "b", "c"]
        x = np.arange(9.0).reshape(3, 3)
        np.testing.assert_array_equal(c2.align(["c", "a"], ids, x, "Q"), x[[2, 0]])
        with self.assertRaises(c2.C2Error):
            c2.align(["a", "b"], ids, x, "Q", val_ids=["b"])
        with self.assertRaises(c2.C2Error):
            c2.align(["a", "z"], ids, x, "Q")

    def test_r_sha_and_pyramid_provenance_enforced(self):
        with tempfile.TemporaryDirectory() as tmp:
            r = os.path.join(tmp, "r.npz")
            np.savez(r, train_ids=np.array(["a"]), y_train=np.array([1]), f512=np.zeros((1, 4)))
            with self.assertRaises(c2.C2Error):
                c2.load_r(r)                                     # not the pinned R file
            self.assertEqual(c2.load_r(r, expected_sha256=None)[2].shape, (1, 4))
            q = os.path.join(tmp, "q.npz")
            feats = {"00000000000a": np.zeros(len(Q_NAMES), np.float32)}
            ac.save_pyramid_features(q, feats, ["00000000000a"], CHANNELS, {"stage4_sha256": "b" * 64})
            self.assertEqual(c2.load_pyramid(q, {"stage4_sha256": "b" * 64})[1].shape, (1, len(Q_NAMES)))
            with self.assertRaises(ac.AptosCacheError):
                c2.load_pyramid(q, {"stage4_sha256": "c" * 64})   # features of another Stage-4 model


if __name__ == "__main__":
    unittest.main()
