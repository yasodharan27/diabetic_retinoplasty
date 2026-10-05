"""CPU tests for arch1_posthoc.py on a synthetic experiment tree written with the project's own per-sample
schema and verdict code (arch1_train). If the real P / PL evaluation files have been copied to
results/PL_ConvNeXtPriors, the metric code is also checked against what those runs stored."""
import json
import os
import tempfile
import unittest

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "-1")

import numpy as np
import pandas as pd

import arch1_posthoc as ph
import arch1_train as at

REPO = os.path.dirname(os.path.abspath(at.__file__)) if os.path.exists(
    os.path.join(os.path.dirname(os.path.abspath(at.__file__)), "corn.py")) else os.path.abspath(
    os.path.join(os.path.dirname(os.path.abspath(at.__file__)), "..", "..", ".."))
REAL_BASELINE = os.path.join(REPO, "results", "PL_ConvNeXtPriors", "2026-09-28_05-22-44")
GRADES = np.array([0] * 60 + [1] * 14 + [2] * 34 + [3] * 8 + [4] * 12)
IDS = [f"{k:012x}" for k in range(len(GRADES))]


def _text(path):
    with open(path, encoding="utf-8") as fh:
        return fh.read()


def _logits(seed, quality):
    """CORN logits whose ordinal quality grows with `quality` (deterministic)."""
    rng = np.random.default_rng(seed)
    return (GRADES[:, None] - np.arange(4)[None, :] - 0.5) * quality + rng.normal(0, 1.0, (len(GRADES), 4))


def _write_tables(directory, seed, quality, prefix="per_sample"):
    os.makedirs(directory, exist_ok=True)
    out = {}
    for which, s in (("best", seed), ("last", seed + 1000)):
        m, rows = at.metrics_from_logits(IDS, GRADES, _logits(s, quality))
        pd.DataFrame(rows).to_csv(os.path.join(directory, f"{prefix}_{which}.csv"), index=False)
        out[which] = m
    return out


def _history(directory, best_index, epochs):
    os.makedirs(directory, exist_ok=True)
    for e in range(1, epochs + 1):
        with open(os.path.join(directory, f"epoch_{e:04d}.json"), "w") as fh:
            json.dump({"epoch": e, "val_QWK": 0.90, "QWK": 0.97, "learning_rate": 1e-4}, fh)


def build_tree(root, qualities, dqwk=-0.03, label=None):
    """A three-seed experiment + P / PL baselines in the on-Drive layout. Returns (runs_root, baseline_root)."""
    runs_root, baseline = os.path.join(root, "Architecture1"), os.path.join(root, "PL_ConvNeXtPriors", "run")
    stored = {}
    for arm, folder in ph.ARMS.items():
        for seed in ph.SEEDS:
            base = os.path.join(baseline, folder, f"seed_{seed}")
            m = _write_tables(os.path.join(base, "evaluation"), seed + (7 if arm == "PL" else 0), 2.0)
            with open(os.path.join(base, "evaluation", "metrics_best.json"), "w") as fh:
                json.dump(dict({k: v for k, v in m["best"].items() if not isinstance(v, dict)},
                               best_epoch_index_0based=4), fh, default=float)
            _history(os.path.join(base, "history"), 4, 9)
            stored[(arm, seed)] = m["best"]
    p42 = stored[("P", 42)]
    verdicts = {}
    dqwks = dqwk if isinstance(dqwk, (tuple, list)) else (dqwk,) * len(ph.SEEDS)
    for seed, quality, dqwk in zip(ph.SEEDS, qualities, dqwks):
        run_dir = os.path.join(runs_root, f"arch1_{'b' * 12}_seed{seed}")
        m = _write_tables(os.path.join(run_dir, "metrics"), seed + 50, quality)
        config = {"seed": seed, "git_commit": "abc", "config_hash": f"h{seed}", "stage4_sha256": "b" * 64,
                  "stage3_sha256": "c" * 64, "stage4_generation": "g4", "bundle_id": "bundle", "bundle_fingerprint": "f" * 64,
                  "split_sha256": "d" * 64, "population_sha256": "e" * 64, "ema": at.EMA}
        if label:
            config.update(experiment="PLv2", label=label)
        results = {}
        for which in ("best", "last"):
            res = {"metrics": m[which],
                   "permutation": {k: {"qwk": m[which]["qwk"] + dqwk, "dqwk": dqwk, "auroc_ge3_g4_vs_g012": 0.7}
                                   for k in ("pathology", "vessel", "both")},
                   "checkpoint": {"which": which.upper(), "weights_sha256": "0" * 64, "best_epoch": 5, "completed_epoch": 12}}
            if which == "best":
                res["one_seed_checks"] = at.one_seed_checks(res, p42)
            with open(os.path.join(run_dir, "metrics", f"metrics_{which}.json"), "w") as fh:
                json.dump(res, fh, default=float)
            results[which] = res
        with open(os.path.join(run_dir, "config.json"), "w") as fh:
            json.dump(config, fh)
        _history(os.path.join(run_dir, "history"), 5, 12)
        verdicts[seed] = at.final_verdict(config, results["best"], p42, c2_pass=False, last=results["last"])
        at._write_json_last(os.path.join(run_dir, "verdict.json"), verdicts[seed])
    at.write_summary(os.path.join(runs_root, f"arch1_{'b' * 12}_3seed_summary"), verdicts)
    return runs_root, baseline, verdicts, stored


class MetricTests(unittest.TestCase):
    def test_metrics_match_the_project_metric_code(self):
        m, rows = at.metrics_from_logits(IDS, GRADES, _logits(3, 1.5))
        got = ph.metrics(ph.table_arrays(pd.DataFrame(rows)))
        for key in ("qwk", "auroc_ge3_g4_vs_g012", "grade3_recall", "false_urgent_rate", "mae"):
            self.assertAlmostEqual(got[key], m[key], places=9, msg=key)
        self.assertAlmostEqual(got["mean_cut_auroc"], np.mean([m[f"auroc_cut_ge{k}"] for k in (1, 2, 3, 4)]), places=9)
        for g in range(5):
            self.assertAlmostEqual(got["recall_per_grade"][g], m["recall_per_grade"][g], places=9)
        self.assertEqual(int(np.sum(got["confusion_matrix"])), len(GRADES))

    def test_bootstrap_metrics_equal_the_full_metrics(self):
        for seed, quality in ((3, 1.5), (4, 0.3)):                                # 0.3: many tied / wrong predictions
            arrays = ph.table_arrays(pd.DataFrame(at.metrics_from_logits(IDS, GRADES, _logits(seed, quality))[1]))
            arrays["p_gt"] = np.round(arrays["p_gt"], 2)                           # force tied scores
            for idx in [np.arange(len(GRADES))] + ph.bootstrap_indices(GRADES, n_boot=5):
                full, fast = ph.metrics(arrays, idx), ph.bootstrap_metrics(arrays, idx)
                for key in ph.BOOT_KEYS:
                    self.assertAlmostEqual(fast[key], full[key], places=10, msg=key)

    def test_bootstrap_is_grade_stratified_and_deterministic(self):
        idx = ph.bootstrap_indices(GRADES, n_boot=20)
        self.assertEqual(len(idx), 20)
        for i in idx:
            np.testing.assert_array_equal(np.bincount(GRADES[i], minlength=5), np.bincount(GRADES, minlength=5))
        again = ph.bootstrap_indices(GRADES, n_boot=20)
        np.testing.assert_array_equal(idx[7], again[7])

    def test_paired_delta_requires_the_same_images(self):
        a = ph.table_arrays(pd.DataFrame(at.metrics_from_logits(IDS, GRADES, _logits(1, 2.0))[1]))
        b = ph.table_arrays(pd.DataFrame(at.metrics_from_logits(IDS, GRADES, _logits(2, 2.0))[1]))
        d = ph.paired_delta(a, b, indices=ph.bootstrap_indices(GRADES, n_boot=50))
        self.assertLessEqual(d["qwk"]["ci_low"], d["qwk"]["delta"] + 0.05)
        self.assertLess(d["qwk"]["ci_low"], d["qwk"]["ci_high"])
        self.assertEqual(ph.paired_delta(a, a)["qwk"]["delta"], 0.0)
        shuffled = dict(b, ids=b["ids"][::-1])
        with self.assertRaises(RuntimeError):
            ph.paired_delta(a, shuffled)

    def test_threshold_check_uses_only_the_recorded_tolerances(self):
        self.assertEqual(ph.TOLERANCE, {"qwk": ("min", -0.02), "auroc_ge3_g4_vs_g012": ("min", -0.01),
                                        "grade3_recall": ("min", -0.10), "false_urgent_rate": ("max", 0.02)})
        self.assertEqual(ph.SHUFFLE_MIN_DROP, 0.01)
        ref = {"qwk": 0.90, "auroc_ge3_g4_vs_g012": 0.95, "grade3_recall": 0.60, "false_urgent_rate": 0.04}
        ok = ph.threshold_check({"qwk": 0.88, "auroc_ge3_g4_vs_g012": 0.94, "grade3_recall": 0.50, "false_urgent_rate": 0.06}, ref)
        self.assertTrue(all(c["passed"] for c in ok.values()))
        bad = ph.threshold_check({"qwk": 0.879, "auroc_ge3_g4_vs_g012": 0.939, "grade3_recall": 0.49, "false_urgent_rate": 0.061}, ref)
        self.assertFalse(any(c["passed"] for c in bad.values()))

    def test_interpretation_categories(self):
        def seeds(dependence, maintained):
            return {s: {"matched_seed": {"lesion_dependence": d, "performance_maintained": m}}
                    for s, d, m in zip(ph.SEEDS, dependence, maintained)}
        T, F = True, False
        cases = {"A": [((T, T, T), (T, T, T))],
                 "B": [((T, T, T), (F, F, F))],
                 "C": [((F, F, F), (T, T, T)), ((F, F, F), (F, F, F)), ((F, F, F), (T, F, T))],
                 "D": [((T, F, T), (T, T, T)), ((T, T, T), (T, F, T)), ((T, T, T), (F, F, T)), ((F, T, F), (F, F, F))]}
        for category, examples in cases.items():
            for dependence, maintained in examples:
                got = ph.interpret(seeds(dependence, maintained), "X")
                self.assertEqual(got["category"], category, (dependence, maintained))
                self.assertEqual(got["lesion_dependence_seeds"], [s for s, d in zip(ph.SEEDS, dependence) if d])
        self.assertIn("worth pursuing", ph.interpret(seeds((T,) * 3, (T,) * 3), "X")["statement"])
        self.assertIn("nothing follows automatically", ph.interpret(seeds((T,) * 3, (T,) * 3), "X")["statement"])
        self.assertIn("before adding any mechanism", ph.interpret(seeds((T,) * 3, (F,) * 3), "X")["statement"])
        self.assertIn("X is closed", ph.interpret(seeds((F,) * 3, (T,) * 3), "X")["statement"])
        self.assertIn("No seed is singled out", ph.interpret(seeds((T, F, T), (T,) * 3), "X")["statement"])
        self.assertEqual(set(ph.CATEGORIES), set("ABCD"))

    @unittest.skipUnless(os.path.isdir(REAL_BASELINE), "real P / PL evaluation files not copied locally")
    def test_reproduces_the_stored_p_and_pl_metrics(self):
        for arm in ph.ARMS:
            for seed in ph.SEEDS:
                run = ph.load_baseline(REAL_BASELINE, arm, seed)
                self.assertEqual(len(run["best"]["grade"]), 730)
                ph.check_stored(ph.metrics(run["best"]), run["stored"], f"{arm}-{seed}")       # raises on mismatch
                self.assertAlmostEqual(ph.metrics(run["best"])["mae"], run["stored"]["mae"], places=9)
        p42 = ph.metrics(ph.load_baseline(REAL_BASELINE, "P", 42)["best"])
        self.assertAlmostEqual(p42["qwk"], 0.9184106157644645, places=9)
        # The finding the report exists to surface: P's own other seeds do not pass the P-42 thresholds.
        p123 = ph.threshold_check(ph.metrics(ph.load_baseline(REAL_BASELINE, "P", 123)["best"]), p42)
        p2026 = ph.threshold_check(ph.metrics(ph.load_baseline(REAL_BASELINE, "P", 2026)["best"]), p42)
        self.assertFalse(p123["grade3_recall"]["passed"])
        self.assertFalse(p2026["auroc_ge3_g4_vs_g012"]["passed"])


class ReportTests(unittest.TestCase):
    def test_full_report_on_a_synthetic_experiment(self):
        with tempfile.TemporaryDirectory() as root:
            runs_root, baseline, verdicts, stored = build_tree(root, qualities=(2.0, 2.0, 2.0))
            out = os.path.join(root, "out")
            r = ph.main(["--runs", runs_root, "--baseline", baseline, "--out", out, "--n-boot", "40", "--section", "55"])
            summary = json.loads(_text(os.path.join(runs_root, f"arch1_{'b' * 12}_3seed_summary", "summary.json")))
            absolute = r["absolute_reference_bounds"]
            self.assertEqual(absolute["tally_recorded_by_the_run"], summary["route"])      # copied, never recomputed
            self.assertEqual(absolute["label"], "P-42-derived absolute reference bounds; not a multi-seed success criterion")
            deltas = {k: [] for k in ph.MATCHED_KEYS}
            for seed in ph.SEEDS:
                p = r["per_seed"][seed]
                self.assertEqual(p["absolute"]["status"], verdicts[seed]["status"])
                run_m, p_m = verdicts[seed]["criteria"], stored[("P", seed)]
                self.assertAlmostEqual(p["versus"]["P"]["qwk"]["delta"], run_m["qwk"]["arch1"] - p_m["qwk"], places=9)
                self.assertEqual(p["matched_seed"]["reference"], f"P-{seed}")               # same seed, not P-42
                self.assertEqual(p["matched_seed"]["lesion_shuffle_dqwk"], -0.03)
                self.assertTrue(p["matched_seed"]["lesion_dependence"])
                within = all(c["passed"] for c in ph.threshold_check(p["best"], ph.metrics(
                    ph.load_baseline(baseline, "P", seed)["best"])).values())
                self.assertEqual(p["matched_seed"]["performance_maintained"], within)
                self.assertEqual(p["history"]["epochs_run"], 12)
                self.assertAlmostEqual(p["history"]["train_minus_val_qwk_at_best"], 0.07)
                for k in deltas:
                    deltas[k].append(p["versus"]["P"][k]["delta"])
            for k, values in deltas.items():                                             # mean +/- SD over the seeds
                self.assertAlmostEqual(r["matched"]["delta"][k]["mean"], np.mean(values), places=12)
                self.assertAlmostEqual(r["matched"]["delta"][k]["sd"], np.std(values, ddof=1), places=12)
            self.assertAlmostEqual(r["matched"]["lesion_shuffle_dqwk"]["mean"], -0.03)
            self.assertLessEqual(r["matched"]["delta"]["qwk"]["ci_low"], r["matched"]["delta"]["qwk"]["ci_high"])
            self.assertEqual(r["interpretation"]["category"], ph.interpret(r["per_seed"])["category"])
            self.assertIn(r["interpretation"]["category"], "AD")                          # dependence is 3/3 here
            self.assertEqual(set(r["reference_context"]), {f"{a}-{s}" for a in ph.ARMS for s in ph.SEEDS})
            self.assertTrue(r["reference_context"]["P-42"]["all_metric_checks_pass"])       # P-42 against itself
            for name in ("results.json", "report.md", "record_section.md"):
                self.assertTrue(os.path.exists(os.path.join(out, name)))
            report = _text(os.path.join(out, "report.md"))
            for needle in ("## 1. Matched-seed comparison with P (primary analysis)", "lesion-shuffle ΔQWK",
                           "## 2. Interpretation: ", "before seeds 123 and 2026 finished",
                           "## 3. P-42-derived absolute reference bounds; not a multi-seed success criterion",
                           "Tally recorded by the run", "## 6. Not concluded here", "No superiority over P is claimed",
                           "stay dormant", "Everything in this report is descriptive."):
                self.assertIn(needle, report)
            for banned in ("next step", "Consequence", "VIABLE →", "is the next"):
                self.assertNotIn(banned, report)                                          # a reading, never a launch
            section = _text(os.path.join(out, "record_section.md"))
            self.assertTrue(section.startswith("## 55. Architecture 1 — three seeds (42, 123, 2026): category **"))
            for needle in ("Matched-seed comparison with P (primary analysis", "Interpretation (rules of §54)", "EMA: none",
                           "FAILED", "P-42-derived absolute reference bounds; not a multi-seed success criterion",
                           "**Not concluded.**"):
                self.assertIn(needle, section)
            json.loads(_text(os.path.join(out, "results.json")))                         # valid JSON

    def test_categories_and_labels_end_to_end(self):
        with tempfile.TemporaryDirectory() as root:                                      # maps ignored in every seed
            runs_root, baseline, _, _ = build_tree(root, qualities=(2.0, 2.0, 2.0), dqwk=-0.001)
            r = ph.analyse(runs_root, baseline, n_boot=10, log=lambda *a: None)
            self.assertEqual((r["interpretation"]["category"], r["label"]), ("C", "Architecture 1"))
            self.assertEqual(r["absolute_reference_bounds"]["tally_recorded_by_the_run"], "CLOSED")
            self.assertIn("Architecture 1 is closed as a way of using them", ph.report_markdown(r))
            self.assertIn("Architecture 1 — three seeds", ph.record_section(r, 56))
        with tempfile.TemporaryDirectory() as root:                                      # maps used in two seeds of three
            runs_root, baseline, _, _ = build_tree(root, qualities=(2.0, 2.0, 2.0), dqwk=(-0.03, -0.001, -0.03))
            r = ph.analyse(runs_root, baseline, n_boot=10, log=lambda *a: None)
            self.assertEqual(r["interpretation"]["category"], "D")
            self.assertEqual(r["interpretation"]["lesion_dependence_seeds"], [42, 2026])
            self.assertIn("No seed is singled out", ph.report_markdown(r))

    def test_refuses_an_incomplete_experiment(self):
        with tempfile.TemporaryDirectory() as root:
            runs_root, baseline, _, _ = build_tree(root, qualities=(2.0, 2.0, 2.0))
            os.remove(os.path.join(runs_root, f"arch1_{'b' * 12}_seed123", "verdict.json"))
            with self.assertRaises(RuntimeError):
                ph.analyse(runs_root, baseline, n_boot=5, log=lambda *a: None)
            with self.assertRaises(RuntimeError):
                ph.find_runs(os.path.join(root, "nothing"))


if __name__ == "__main__":
    unittest.main()
