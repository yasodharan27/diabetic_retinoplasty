"""Fast, CPU-only unit tests for icdr_two_route_head.py and the pure parts of
icdr_two_route_experiment.py (the pre-registered C1 two-route head experiment)."""
import os
import unittest

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "-1")

import numpy as np
import tensorflow as tf
from keras import Input, Model, layers

import corn
import icdr_two_route_experiment as exp
import icdr_two_route_head as th


def _synthetic_training_set(seed=0, per_grade=(60, 25, 45, 20, 25), d=th.D_MODEL):
    """Grades with a weak ordinal signal plus noise, so every task is fittable but not separable."""
    rng = np.random.default_rng(seed)
    grades = np.concatenate([np.full(n, g) for g, n in enumerate(per_grade)])
    direction = rng.normal(size=d)
    e = rng.normal(size=(grades.size, d)) + 0.4 * grades[:, None] * direction[None, :]
    return e.astype(np.float32), grades


class ParameterCountTests(unittest.TestCase):  # A
    def test_both_heads_have_exactly_1028_parameters(self):
        report = th.parameter_parity_report()
        self.assertEqual(report["original_corn"]["trainable_parameters"], 1028)
        self.assertEqual(report["H1_corn_refit"]["trainable_parameters"], 1028)
        self.assertEqual(report["H2_two_route"]["trainable_parameters"], 1028)
        self.assertEqual(report["difference_H2_minus_H1"], 0)
        self.assertEqual(report["difference_H2_minus_original"], 0)
        self.assertEqual(report["H1_corn_refit"]["trainable_tensors"], 2)
        self.assertEqual(report["H2_two_route"]["trainable_tensors"], 4)

    def test_fitted_heads_have_1028_parameters(self):
        e, g = _synthetic_training_set()
        self.assertEqual(th.fit_h1_corn_refit(e, g).parameter_count(), 1028)
        self.assertEqual(th.fit_h2_two_route(e, g).parameter_count(), 1028)


class ProbabilityConstructionTests(unittest.TestCase):  # B, C, D
    def setUp(self):
        rng = np.random.default_rng(1)
        self.z_pdr = rng.normal(0, 3, size=500)
        self.z_npdr = rng.normal(0, 3, size=(500, 3))
        self.out = th.two_route_outputs(self.z_pdr, self.z_npdr)

    def test_probabilities_sum_to_one_and_are_valid(self):
        p = self.out["probabilities"]
        np.testing.assert_allclose(p.sum(axis=1), 1.0, atol=1e-12)
        self.assertTrue(np.all(p >= -1e-15))

    def test_q_equals_p4(self):
        np.testing.assert_allclose(self.out["probabilities"][:, 4], self.out["q"], atol=0)
        np.testing.assert_allclose(self.out["q"], 1 / (1 + np.exp(-self.z_pdr)), rtol=1e-12)
        np.testing.assert_allclose(self.out["p_grade4"], self.out["q"], atol=0)

    def test_npdr_probabilities_are_one_minus_q_times_corn_over_grades_0_to_3(self):
        p_cond = 1 / (1 + np.exp(-self.z_npdr))
        p_cum = np.cumprod(p_cond, axis=1)
        p_npdr = np.stack([1 - p_cum[:, 0], p_cum[:, 0] - p_cum[:, 1], p_cum[:, 1] - p_cum[:, 2],
                           p_cum[:, 2]], axis=1)
        np.testing.assert_allclose(self.out["probabilities"][:, :4],
                                   (1 - self.out["q"])[:, None] * p_npdr, atol=1e-12)
        np.testing.assert_allclose(p_npdr.sum(axis=1), 1.0, atol=1e-12)

    def test_p_ge3_is_p3_plus_p4(self):
        p = self.out["probabilities"]
        np.testing.assert_allclose(self.out["p_ge3"], p[:, 3] + p[:, 4], atol=1e-15)

    def test_h1_outputs_match_the_projects_corn_decode(self):
        z = np.random.default_rng(2).normal(0, 3, size=(300, 4))
        out = th.corn_outputs(z)
        ref = corn.decode_logits(z)
        np.testing.assert_array_equal(out["predicted_grade"], ref["predicted_grade"])
        np.testing.assert_allclose(out["probabilities"], ref["class_probabilities"], atol=1e-6)
        np.testing.assert_allclose(out["p_ge3"], ref["p_cum"][:, 2], atol=1e-6)
        np.testing.assert_allclose(out["p_grade4"], ref["p_cum"][:, 3], atol=1e-6)
        np.testing.assert_allclose(out["probabilities"].sum(axis=1), 1.0, atol=1e-12)


class DecodeTests(unittest.TestCase):  # E
    def test_q_above_half_decodes_to_4_regardless_of_npdr_chain(self):
        out = th.two_route_outputs(np.array([0.01, 5.0]), np.array([[-9, -9, -9], [-9, -9, -9]]))
        np.testing.assert_array_equal(out["predicted_grade"], [4, 4])

    def test_q_at_or_below_half_uses_corn_decode_over_grades_0_to_3(self):
        z_npdr = np.array([[-9, -9, -9], [9, -9, -9], [9, 9, -9], [9, 9, 9]], dtype=float)
        for z_pdr in (0.0, -0.01, -5.0):  # q = 0.5 exactly is NOT > 0.5
            out = th.two_route_outputs(np.full(4, z_pdr), z_npdr)
            np.testing.assert_array_equal(out["predicted_grade"], [0, 1, 2, 3])

    def test_threshold_is_fixed_at_one_half(self):
        self.assertEqual(th.PDR_DECISION_THRESHOLD, 0.5)


class Grade4ExclusionTests(unittest.TestCase):  # F
    def test_task_definitions(self):
        g = np.array([0, 1, 2, 3, 4, 4])
        h2 = dict((n, (m, t)) for n, m, t in th.h2_tasks(g))
        for k in range(3):
            mask, _ = h2[f"npdr_task_{k}"]
            self.assertFalse(mask[g == 4].any(), f"grade 4 enters NPDR task {k}")
        np.testing.assert_array_equal(h2["pdr_route"][0], np.ones(6, bool))
        np.testing.assert_array_equal(h2["pdr_route"][1], (g == 4).astype(float))
        h1 = dict((n, (m, t)) for n, m, t in th.h1_tasks(g))
        self.assertTrue(h1["corn_task_2"][0][g == 4].all())  # CORN does use grade 4 in task 2

    def test_grade4_rows_cannot_influence_the_npdr_route(self):
        """With the (training-only) standardised features held fixed, overwriting every grade-4
        row with arbitrary values leaves each NPDR task's fitted parameters bit-for-bit
        identical: grade-4 samples provide no NPDR supervision. (The shared standardiser itself
        is computed over all training rows, identically for H1 and H2.)"""
        e, g = _synthetic_training_set(seed=3)
        base = th.fit_h2_two_route(e, g)
        idx4 = np.flatnonzero(g == 4)
        rng = np.random.default_rng(0)
        mean, sd = th.standardiser(e)
        x = (e.astype(np.float64) - mean) / sd
        w = np.asarray(th.CLASS_WEIGHTS)[g]
        for name, mask, target in th.h2_tasks(g):
            if not name.startswith("npdr"):
                continue
            beta_a, b_a, _ = th.fit_weighted_logistic(x[mask], target[mask], w[mask])
            x_mod = x.copy()
            x_mod[idx4] = rng.normal(size=(idx4.size, x.shape[1])) * 50.0
            beta_b, b_b, _ = th.fit_weighted_logistic(x_mod[mask], target[mask], w[mask])
            np.testing.assert_array_equal(beta_a, beta_b)
            self.assertEqual(b_a, b_b)
        self.assertEqual(base.task_info["pdr_route"]["n"], g.size)
        self.assertEqual(sum(base.task_info["npdr_task_0"]["grades_in_task"].values()),
                         int((g <= 3).sum()))
        self.assertNotIn(4, base.task_info["npdr_task_0"]["grades_in_task"])

    def test_route_population_counts_on_the_split_grade_distribution(self):
        g = np.concatenate([np.full(n, k) for k, n in enumerate((1444, 296, 799, 154, 236))])
        tasks = dict((n, m) for n, m, _ in th.h2_tasks(g))
        self.assertEqual(int(tasks["pdr_route"].sum()), 2929)
        self.assertEqual(int(tasks["npdr_task_0"].sum()), 2693)


class InputParityAndFitterTests(unittest.TestCase):  # G + fitter correctness
    def test_h1_and_h2_share_one_fitter_config_and_standardiser(self):
        e, g = _synthetic_training_set(seed=4)
        h1, h2 = th.fit_h1_corn_refit(e, g), th.fit_h2_two_route(e, g)
        self.assertEqual(h1.l2, h2.l2)
        # identical standardisation: the PDR route of H2 and CORN task 3 of H1 would differ only by
        # task definition; check the fitter is deterministic and input-order sensitive only to data
        np.testing.assert_array_equal(th.fit_h2_two_route(e, g).kernel, h2.kernel)
        np.testing.assert_array_equal(th.fit_h1_corn_refit(e, g).kernel, h1.kernel)
        for info in list(h1.task_info.values()) + list(h2.task_info.values()):
            self.assertTrue(info["converged"], info)

    def test_fitter_matches_a_scipy_free_reference_optimum(self):
        """At the optimum the gradient of the stated objective must vanish."""
        rng = np.random.default_rng(5)
        x = rng.normal(size=(300, 8))
        y = (x[:, 0] + rng.normal(size=300) > 0).astype(float)
        w = rng.uniform(0.5, 2.0, size=300)
        beta, b, info = th.fit_weighted_logistic(x, y, w, l2=1e-2)
        z = x @ beta + b
        r = w * (1 / (1 + np.exp(-z)) - y) / 300
        np.testing.assert_allclose(x.T @ r + 1e-2 * beta, 0, atol=1e-7)
        self.assertAlmostEqual(float(r.sum()), 0.0, places=7)
        self.assertTrue(info["converged"])

    def test_standardisation_is_folded_into_the_raw_E_head(self):
        e, g = _synthetic_training_set(seed=6)
        h2 = th.fit_h2_two_route(e, g)
        mean, sd = th.standardiser(e)
        pdr_beta, pdr_b, _ = th.fit_weighted_logistic(
            (e.astype(np.float64) - mean) / sd, (g == 4).astype(float), np.asarray(th.CLASS_WEIGHTS)[g])
        z_std = ((e.astype(np.float64) - mean) / sd) @ pdr_beta + pdr_b
        np.testing.assert_allclose(h2.logits(e)[:, 0], z_std, atol=1e-8)

    def test_keras_export_reproduces_numpy_logits(self):
        e, g = _synthetic_training_set(seed=7)
        for fitted in (th.fit_h1_corn_refit(e, g), th.fit_h2_two_route(e, g)):
            keras_logits = th.to_keras(fitted).predict(e, verbose=0)
            np.testing.assert_allclose(keras_logits, fitted.logits(e), atol=1e-3)

    def test_convergence_record_reports_scipy_status_and_gradient(self):
        rng = np.random.default_rng(9)
        x = rng.normal(size=(200, 5))
        y = (x[:, 0] > 0).astype(float)
        _, _, info = th.fit_weighted_logistic(x, y, np.ones(200))
        self.assertIn("scipy_success", info)
        self.assertTrue(info["converged"])
        self.assertLessEqual(info["final_grad_max_abs"], th.CONVERGED_GRAD_TOL)

    def test_tasks_need_both_classes(self):
        with self.assertRaises(ValueError):
            th.fit_weighted_logistic(np.zeros((5, 2)), np.zeros(5), np.ones(5))


class FrozenBackboneTests(unittest.TestCase):  # H
    def _tiny_backbone(self):
        inp = Input(shape=(8,))
        h = layers.Dense(6, name="stage_like_a")(inp)
        h = layers.LayerNormalization(name="stage_like_b")(h)
        out = layers.Dense(4, name="stage_like_c")(h)
        return Model(inp, out)

    def test_freeze_removes_trainable_variables_and_inference_does_not_mutate_weights(self):
        tf.keras.utils.set_random_seed(0)
        model = th.freeze(self._tiny_backbone())
        self.assertEqual(len(model.trainable_variables), 0)
        before = th.weights_fingerprint(model)
        model.predict(np.random.default_rng(0).normal(size=(10, 8)), verbose=0)
        model(np.ones((2, 8)), training=False)
        self.assertEqual(th.weights_fingerprint(model), before)

    def test_fingerprint_detects_a_change(self):
        model = self._tiny_backbone()
        before = th.weights_fingerprint(model)
        kernel, bias = model.get_layer("stage_like_a").get_weights()
        model.get_layer("stage_like_a").set_weights([kernel + 1e-6, bias])
        self.assertNotEqual(th.weights_fingerprint(model), before)


class RealBackboneIntegrationTests(unittest.TestCase):  # G + H on the real NO_RACAF graph
    def test_E_is_the_corn_head_input_and_the_frozen_backbone_is_not_mutated(self):
        import no_racaf_model
        tf.keras.utils.set_random_seed(123)
        model = th.freeze(no_racaf_model.build_no_racaf_joint_model())
        self.assertEqual(len(model.trainable_variables), 0)
        before = {n: th.weights_fingerprint(model.get_layer(n)) for n in exp.STAGE_LAYER_NAMES}
        rng = np.random.default_rng(0)
        s5 = rng.uniform(0, 1, size=(1, *model.inputs[0].shape[1:])).astype(np.float32)
        s6 = rng.uniform(0, 1, size=(1, *model.inputs[1].shape[1:])).astype(np.float32)
        rel = np.array([[0.9]], np.float32)
        logits, logits_eager, e = exp.forward_with_embedding(model, s5, s6, rel)
        self.assertEqual(e.shape, (1, th.D_MODEL))
        kernel, bias = [np.asarray(w, np.float64) for w in model.get_layer("corn").get_weights()]
        np.testing.assert_allclose(e @ kernel + bias, logits, atol=1e-4)
        np.testing.assert_allclose(logits_eager, logits, atol=1e-4)
        after = {n: th.weights_fingerprint(model.get_layer(n)) for n in exp.STAGE_LAYER_NAMES}
        self.assertEqual(before, after)


class DecisionRuleTests(unittest.TestCase):
    OK = {"grade3_recall": 0.0, "auroc_ge3_g3_vs_g012": 0.0, "qwk": 0.0, "false_urgent_rate": 0.0}

    def test_supportive(self):
        self.assertEqual(exp.decide([0.04, 0.03, 0.05], self.OK)["verdict"], "SUPPORTIVE")

    def test_not_supportive_small_mean(self):
        self.assertEqual(exp.decide([0.015, 0.001, 0.01], self.OK)["verdict"], "NOT_SUPPORTIVE")

    def test_mean_exactly_at_the_not_supportive_bar_is_not_not_supportive(self):
        # mean 0.01 is not < 0.01, and only one seed is non-positive
        self.assertEqual(exp.decide([0.02, 0.0, 0.01], self.OK)["verdict"], "INCONCLUSIVE")

    def test_not_supportive_two_non_positive_seeds(self):
        self.assertEqual(exp.decide([0.2, -0.01, 0.0], self.OK)["verdict"], "NOT_SUPPORTIVE")

    def test_trade_off_when_a_guardrail_fails(self):
        bad = dict(self.OK, qwk=-0.021)
        self.assertEqual(exp.decide([0.04, 0.03, 0.05], bad)["verdict"], "TRADE_OFF")
        bad = dict(self.OK, false_urgent_rate=0.021)
        self.assertEqual(exp.decide([0.04, 0.03, 0.05], bad)["verdict"], "TRADE_OFF")

    def test_inconclusive(self):
        self.assertEqual(exp.decide([0.02, 0.01, 0.03], self.OK)["verdict"], "INCONCLUSIVE")
        self.assertEqual(exp.decide([0.06, -0.01, 0.06], self.OK)["verdict"], "INCONCLUSIVE")

    def test_guardrail_boundaries_are_inclusive(self):
        edge = {"grade3_recall": -0.10, "auroc_ge3_g3_vs_g012": -0.03, "qwk": -0.02,
                "false_urgent_rate": 0.02}
        self.assertEqual(exp.decide([0.03, 0.03, 0.03], edge)["verdict"], "SUPPORTIVE")


class OfflineDryRunTests(unittest.TestCase):
    """Runs the whole post-extraction pipeline (smoke checks, fits, scoring, bootstrap, verdict,
    report) on synthetic embeddings, so a Colab run cannot fail after hours of E extraction."""

    def test_run_analysis_end_to_end_on_synthetic_embeddings(self):
        rng = np.random.default_rng(11)
        train_counts, val_counts = (150, 40, 90, 30, 40), (60, 15, 40, 10, 20)
        y_train = np.concatenate([np.full(n, g) for g, n in enumerate(train_counts)])
        y_val = np.concatenate([np.full(n, g) for g, n in enumerate(val_counts)])
        direction = rng.normal(size=th.D_MODEL)
        embeddings, h0 = {}, {}
        for seed in exp.BACKBONE_SEEDS:
            def draw(y):
                return (rng.normal(size=(y.size, th.D_MODEL))
                        + 0.05 * y[:, None] * direction[None, :]).astype(np.float32)
            embeddings[seed] = (draw(y_train), draw(y_val))
            h0[seed] = {"predicted_grade": np.clip(y_val + rng.integers(-1, 2, y_val.size), 0, 4),
                        "p_ge3": rng.uniform(size=y_val.size), "p_grade4": rng.uniform(size=y_val.size)}
        persistent = (y_val == 4) & (np.arange(y_val.size) % 3 == 0)
        results = {"experiment_version": exp.EXPERIMENT_VERSION, "out_dir": "(dry run)",
                   "repo_commit": "dry-run", "fitting_configuration": th.fitting_configuration(),
                   "parameter_parity": th.parameter_parity_report(),
                   "population": {"n_train": int(y_train.size), "n_val": int(y_val.size)}}
        val_ids = [f"v{i}" for i in range(y_val.size)]
        import tempfile
        with tempfile.TemporaryDirectory() as out_dir:
            out, table, samples = exp.run_analysis(results, embeddings, y_train, y_val, h0,
                                                   persistent, val_ids, out_dir=out_dir,
                                                   strict=False)
            for name in ("REPORT.md", "results.json", "per_seed_results.csv", "h2_minus_h1.csv",
                         "per_sample_predictions.csv", "heads_seed_42.npz"):
                self.assertTrue(os.path.exists(os.path.join(out_dir, name)), name)
            self.assertFalse(os.path.exists(os.path.join(out_dir, "results.json.tmp")))
            report = open(os.path.join(out_dir, "REPORT.md"), encoding="utf-8").read()
        for section in ("Pre-registered verdict", "Detailed analysis", "B. Fit diagnostics",
                        "C. Every metric", "D. Confusion matrices", "E. What changed",
                        "F. The persistent-21", "G. Score distributions"):
            self.assertIn(section, report)
        self.assertIn(out["decision"]["verdict"],
                      {"SUPPORTIVE", "NOT_SUPPORTIVE", "TRADE_OFF", "INCONCLUSIVE"})
        self.assertEqual(len(out["decision"]["per_seed_delta"]), 3)
        lo, hi = out["decision"]["bootstrap"]["mean_delta_ci"]
        self.assertLessEqual(lo, hi)
        self.assertEqual(len(table), 9)                      # 3 seeds x H0/H1/H2
        self.assertEqual(len(samples), 3 * y_val.size)
        np.testing.assert_allclose(samples[[f"H2_P{k}" for k in range(5)]].sum(axis=1), 1.0,
                                   atol=1e-9)
        self.assertTrue(out["smoke"]["grade4_excluded_from_npdr_route"])
        for seed in exp.BACKBONE_SEEDS:
            for head in ("H1", "H2"):
                self.assertEqual(out["fitted_heads"][seed][head]["parameters"], 1028)
            self.assertNotIn(4, out["fitted_heads"][seed]["H2"]["tasks"]["npdr_task_0"]
                             ["grades_in_task"])


class ResumeAndPersistenceTests(unittest.TestCase):
    REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

    def test_experiment_directory_is_created_then_resumed_then_marked_completed(self):
        import json
        import tempfile
        with tempfile.TemporaryDirectory() as root:
            group = os.path.join(root, exp.OUTPUT_GROUP).replace("\\", "/")
            out_dir, completed = exp.resolve_experiment_dir(group)
            self.assertFalse(completed)
            exp.freeze_or_verify_preregistration(out_dir, self.REPO_ROOT, "abc")
            again, completed = exp.resolve_experiment_dir(group)
            self.assertEqual(again, out_dir)                 # resumed, not a new directory
            self.assertFalse(completed)
            with open(os.path.join(out_dir, exp.RESULTS_FILENAME), "w") as fh:
                json.dump({}, fh)
            self.assertTrue(exp.resolve_experiment_dir(group)[1])
            second = os.path.join(group, "2099-01-01_00-00-00")
            os.makedirs(second)
            with open(os.path.join(second, exp.PREREGISTRATION_FILENAME), "w") as fh:
                json.dump({}, fh)
            with self.assertRaises(RuntimeError):
                exp.resolve_experiment_dir(group)

    def test_preregistration_is_written_once_then_verified_and_tampering_stops(self):
        import json
        import tempfile
        with tempfile.TemporaryDirectory() as out_dir:
            _, how = exp.freeze_or_verify_preregistration(out_dir, self.REPO_ROOT, "c1")
            self.assertEqual(how, "written")
            _, how = exp.freeze_or_verify_preregistration(out_dir, self.REPO_ROOT, "c2")
            self.assertEqual(how, "verified")              # a new commit alone is allowed
            path = os.path.join(out_dir, exp.PREREGISTRATION_FILENAME)
            stored = json.load(open(path))
            self.assertIn("source_sha256", stored)
            stored["fitting"]["l2"] = 1e-3
            json.dump(stored, open(path, "w"))
            open(os.path.join(out_dir, "E_seed_42.npz"), "wb").close()   # data already touched
            with self.assertRaises(RuntimeError):
                exp.freeze_or_verify_preregistration(out_dir, self.REPO_ROOT, "c3")

    def test_preregistration_is_refrozen_only_before_any_data_contact(self):
        import json
        import tempfile
        with tempfile.TemporaryDirectory() as out_dir:
            exp.freeze_or_verify_preregistration(out_dir, self.REPO_ROOT, "c1")
            path = os.path.join(out_dir, exp.PREREGISTRATION_FILENAME)
            stored = json.load(open(path))
            stored["source_sha256"]["icdr_two_route_head.py"] = "0" * 64   # an older code version
            json.dump(stored, open(path, "w"))
            open(os.path.join(out_dir, "run_status.json"), "w").close()     # not a data artifact
            new, how = exp.freeze_or_verify_preregistration(out_dir, self.REPO_ROOT, "c2")
            self.assertIn("re-frozen", how)
            self.assertEqual(new["supersedes"][0]["changed_keys"], ["source_sha256"])
            self.assertTrue(os.path.exists(os.path.join(out_dir, new["supersedes"][0]["file"])))
            _, how = exp.freeze_or_verify_preregistration(out_dir, self.REPO_ROOT, "c3")
            self.assertEqual(how, "verified")
            self.assertEqual(exp.data_artifacts(out_dir), [])

    def test_population_pin_uses_the_six_run_manifest_including_empty_fov_exclusions(self):
        """The cached population is the split minus the pinned empty-FOV ids (8 train + 3
        validation on the real data) -- it must pass, and any other mismatch must stop."""
        import json
        import tempfile
        import types
        counts = exp.SPLIT_TRAIN_COUNTS
        train = [(f"t{g}_{i}", g) for g, n in enumerate(counts) for i in range(n)]
        val = [(f"v{i}", i % 5) for i in range(733)]
        empty = [train[i][0] for i in (0, 1, 2, 3, 1500, 1501, 1800, 1900)] + ["v0", "v1", "v2"]
        fake_msr = types.SimpleNamespace(EXPECTED_SPLIT_SHA256="abc",
                                         verify_split=lambda: (train, val, "abc"))
        fake_itd = types.SimpleNamespace(
            locally_cached_entries=lambda entries, *a: [e for e in entries if e[0] not in empty])
        original = exp.ensure_local_cache
        exp.ensure_local_cache = lambda config: "already_extracted"
        try:
            with tempfile.TemporaryDirectory() as six:
                manifest = os.path.join(six, "experiment_manifest.json")
                json.dump({"n_train_yielded": 2921, "n_val_yielded": 730, "empty_fov_ids": empty},
                          open(manifest, "w"))
                data = exp.prepare_population(None, fake_itd, fake_msr, six)
                self.assertEqual(data["population"]["n_train"], 2921)
                self.assertEqual(len(data["population"]["excluded_empty_fov_train"]), 8)
                self.assertEqual(data["population"]["n_val"], 730)
                json.dump({"n_train_yielded": 2929, "n_val_yielded": 730, "empty_fov_ids": empty},
                          open(manifest, "w"))
                with self.assertRaises(RuntimeError):
                    exp.prepare_population(None, fake_itd, fake_msr, six)
                json.dump({"n_train_yielded": 2921, "n_val_yielded": 730,
                           "empty_fov_ids": empty[1:]}, open(manifest, "w"))
                with self.assertRaises(RuntimeError):                 # unexplained exclusion
                    exp.prepare_population(None, fake_itd, fake_msr, six)
        finally:
            exp.ensure_local_cache = original

    def test_embedding_status_from_disk(self):
        import tempfile
        with tempfile.TemporaryDirectory() as out_dir:
            self.assertEqual(exp.embedding_status(out_dir, 42), "NOT_STARTED")
            npz, record = exp._embedding_paths(out_dir, 42)
            open(npz, "wb").close()
            self.assertEqual(exp.embedding_status(out_dir, 42), "PARTIAL")
            open(record, "w").close()
            self.assertEqual(exp.embedding_status(out_dir, 42), "EXTRACTED")


class MetricsTests(unittest.TestCase):
    def test_head_metrics_on_a_hand_built_case(self):
        grades = np.array([0, 1, 2, 2, 3, 3, 4, 4])
        pred = np.array([0, 1, 3, 2, 3, 2, 4, 1])
        p_ge3 = np.array([.1, .2, .6, .3, .7, .4, .9, .5])
        p4 = np.array([.0, .1, .2, .1, .3, .2, .8, .4])
        m = exp.head_metrics(grades, pred, p_ge3, p4, persistent_mask=np.array(
            [0, 0, 0, 0, 0, 0, 0, 1], bool))
        self.assertAlmostEqual(m["grade4_recall"], 0.5)
        self.assertAlmostEqual(m["grade4_precision"], 1.0)
        self.assertAlmostEqual(m["grade3_recall"], 0.5)
        self.assertAlmostEqual(m["false_urgent_rate"], 0.25)
        self.assertEqual(m["persistent21_decoded_le2"], 1)
        self.assertEqual(m["grade4_decoded_counts"], [0, 1, 0, 0, 1])
        self.assertAlmostEqual(m["mae"], (1 + 1 + 3) / 8)
        # grade 4 vs 0-2 on P(>=3): positives .9,.5 ; negatives .1,.2,.6,.3 -> 7/8
        self.assertAlmostEqual(m["auroc_ge3_g4_vs_g012"], 7 / 8)


if __name__ == "__main__":
    unittest.main()
