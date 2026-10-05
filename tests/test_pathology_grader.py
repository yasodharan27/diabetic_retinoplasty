"""CPU tests for the pathology-only grader (pathology_grader_model.py, pathology_grader_train.py). The sequence
is tested with recorders; one slow test runs the REAL loop for two epochs on the 5-image synthetic bundle with
the RGB cache deleted -- that is not an experiment and produces no result."""
import json
import os
import shutil
import tempfile
import unittest

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "-1")

import numpy as np
import pandas as pd

import arch1_data as ad
import arch1_posthoc as ph
import arch1_train as at
import pathology_grader_fusion as pf
import pathology_grader_model as pm
import pathology_grader_train as pt
import pipeline_v2_config as v2cfg
import stage34_cache_v2 as cache
import weighted_corn

try:
    from tests import v2_bundle_fixture as fx
except ImportError:
    import v2_bundle_fixture as fx

CW = list(weighted_corn.PREREGISTERED_CLASS_WEIGHTS)
CHANNELS = cache.channel_names(v2cfg.STAGE4_V2A_CLASSES)
SLOW = bool(os.environ.get("STAGED_SKIP_SLOW"))


def _bundle(tmp):
    b = fx.build(os.path.join(tmp, "data"))
    bundle = ad.Arch1Bundle(expected_bundle_id=b["bundle"]["bundle_id"], expected_split_sha256=v2cfg.SPLIT_SHA256,
                            expected_stage4_sha256=fx.MODEL_SHA, roots=b["roots"], expected_population=5)
    return bundle, b


def _inputs(size, n=2, seed=1):
    rng = np.random.default_rng(seed)
    return {"vessel": rng.uniform(0, 1, (n, size, size, 1)).astype("float32"),
            "pathology": rng.uniform(0, 1, (n, size, size, 8)).astype("float32")}


class ModelTests(unittest.TestCase):
    def test_inputs_are_vessel_and_lesion_maps_only(self):
        self.assertEqual(pm.input_channel_names(CHANNELS),
                         ("vessel", "MA:mean", "MA:max", "HE:mean", "HE:max", "EX:mean", "EX:max", "SE:mean", "SE:max"))
        model = pm.build_pathology_grader(CHANNELS, 42, image_size=64)
        self.assertEqual([i.name for i in model.inputs], ["vessel", "pathology"])
        self.assertEqual([tuple(i.shape[1:]) for i in model.inputs], [(64, 64, 1), (64, 64, 8)])
        names = [l.name for l in model.layers]
        self.assertFalse([n for n in names if "convnext" in n or "rgb" in n or "injection" in n], names)
        self.assertEqual(tuple(model.output.shape[1:]), (4,))
        report = pm.parameter_report(model)
        self.assertEqual(report["head"], 256 * 4 + 4)
        self.assertEqual(report["total"], report["encoder"] + report["head_norm"] + report["head"])
        self.assertLess(report["total"], 2_000_000)                               # no backbone hidden anywhere
        self.assertEqual(pm.INPUT_NAMES, ("vessel", "pathology"))

    def test_both_inputs_and_every_lesion_class_reach_the_output(self):
        model = pm.build_pathology_grader(CHANNELS, 42, image_size=64)
        x = _inputs(64)
        base = model.predict_on_batch(x)
        self.assertTrue(np.all(np.isfinite(base)))
        self.assertGreater(np.abs(model.predict_on_batch(dict(x, vessel=1 - x["vessel"])) - base).max(), 1e-6)
        for c in range(4):
            changed = x["pathology"].copy()
            changed[..., 2 * c:2 * c + 2] = 1 - changed[..., 2 * c:2 * c + 2]
            self.assertGreater(np.abs(model.predict_on_batch(dict(x, pathology=changed)) - base).max(), 1e-6, c)

    def test_seed_handling(self):
        a = at.weights_sha256(pm.build_pathology_grader(CHANNELS, 42, image_size=64))
        self.assertEqual(a, at.weights_sha256(pm.build_pathology_grader(CHANNELS, 42, image_size=64)))
        self.assertNotEqual(a, at.weights_sha256(pm.build_pathology_grader(CHANNELS, 123, image_size=64)))
        model = pm.build_pathology_grader(CHANNELS, 2026, image_size=64)
        kernel, bias = model.get_layer("corn").get_layer("corn_logits").get_weights()
        want_kernel, want_bias = pm.head_initial_weights(2026)
        np.testing.assert_array_equal(kernel, want_kernel)
        np.testing.assert_array_equal(bias, want_bias)
        m42 = pt.build_compiled_model(CHANNELS, 42, CW, mixed_precision=False, image_size=64)
        start = at.weights_sha256(m42)
        pristine_123 = at.weights_sha256(pt.build_compiled_model(CHANNELS, 123, CW, mixed_precision=False, image_size=64))
        m42.train_on_batch(_inputs(64), np.array([1, 3], "int32"))
        self.assertNotEqual(at.weights_sha256(m42), start)
        self.assertEqual(at.weights_sha256(pt.build_compiled_model(CHANNELS, 123, CW, mixed_precision=False, image_size=64)),
                         pristine_123)                                            # nothing leaks between seeds

    def test_compiled_like_p(self):
        model = pt.build_compiled_model(CHANNELS, 42, CW, mixed_precision=False, image_size=64)
        logits = np.random.default_rng(0).normal(size=(6, 4)).astype("float32")
        grades = np.array([0, 1, 2, 3, 4, 2], "int32")
        self.assertAlmostEqual(float(model.loss(grades, logits)),
                               float(weighted_corn.weighted_corn_loss_value(logits, grades, CW)), places=6)
        import pl_convnext as pl
        self.assertEqual(pl.weight_decay_report(model)["decayed_one_d"], [])
        self.assertEqual(type(pl.inner_optimizer(model)).__name__, "AdamW")


class DataTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls._tmp = tempfile.TemporaryDirectory()
        cls.bundle, cls.built = _bundle(cls._tmp.name)
        shutil.rmtree(cls.bundle.dirs["stage2_rgb_v2"])                          # the RGB cache does not exist here

    @classmethod
    def tearDownClass(cls):
        cls._tmp.cleanup()

    def test_reads_vessel_and_pathology_without_any_rgb_file(self):
        self.assertFalse(os.path.exists(self.bundle.dirs["stage2_rgb_v2"]))
        image_id = fx.TRAIN[0][0]
        s = pt.load_inputs(self.bundle, image_id)
        self.assertEqual(set(s), {"vessel", "pathology"})
        self.assertEqual((s["vessel"].shape, s["pathology"].shape), ((512, 512, 1), (512, 512, 8)))
        self.assertEqual((s["vessel"].dtype, s["pathology"].dtype), (np.float32, np.float32))
        self.assertTrue(0.0 <= s["pathology"].min() and s["pathology"].max() <= 1.0)
        np.testing.assert_allclose(s["vessel"][..., 0], self.built["vessels"][image_id], atol=1e-7)
        self.assertTrue(pt.require_inputs(self.bundle, self.bundle.population))
        with self.assertRaises(Exception):                                        # the Architecture-1 reader needs RGB
            self.bundle.load_sample(image_id)
        self.assertEqual(pt.KINDS, ("stage3_cache_v2", "stage4_cache_v2"))

    def test_batches_follow_p_order_and_carry_no_rgb(self):
        seq = pt.make_epoch_sequence(self.bundle, fx.TRAIN, 3, 42, 2, augment=True)
        self.assertEqual(len(seq), 2)
        inputs, grades = seq[0]
        self.assertEqual(set(inputs), {"vessel", "pathology"})
        self.assertEqual((inputs["vessel"].shape, inputs["pathology"].shape), ((2, 512, 512, 1), (2, 512, 512, 8)))
        order = ad.epoch_entries([(str(i), int(g)) for i, g in fx.TRAIN], 42, 3, True)
        np.testing.assert_array_equal(grades, [g for _, g in order[:2]])
        np.testing.assert_array_equal(seq[1][1], [order[2][1]])
        val = pt.make_epoch_sequence(self.bundle, fx.VAL, 0, 42, 2, augment=False)
        np.testing.assert_array_equal(val[0][1], [g for _, g in fx.VAL])           # validation: given order, unaugmented
        own = pt.load_inputs(self.bundle, fx.VAL[0][0])
        np.testing.assert_array_equal(val[0][0]["pathology"][0], own["pathology"])

    def test_augmentation_is_deterministic_and_is_ps_spatial_transform(self):
        import improved_training_data as itd
        image_id = fx.TRAIN[1][0]
        s = pt.load_inputs(self.bundle, image_id)
        a = pt.augment_inputs(s, itd.per_image_augmentation_rng(42, 5, image_id))
        b = pt.augment_inputs(s, itd.per_image_augmentation_rng(42, 5, image_id))
        np.testing.assert_array_equal(a["pathology"], b["pathology"])
        np.testing.assert_array_equal(a["vessel"], b["vessel"])
        full = ad.augment_sample(dict(s, rgb=np.zeros((512, 512, 3), "float32")),
                                 itd.per_image_augmentation_rng(42, 5, image_id))    # what Architecture 1 applied
        np.testing.assert_array_equal(a["vessel"], full["vessel"])
        np.testing.assert_array_equal(a["pathology"], full["pathology"])
        np.testing.assert_array_equal(np.sort(a["pathology"].ravel()), np.sort(s["pathology"].ravel()))   # values untouched
        seq1 = pt.make_epoch_sequence(self.bundle, fx.TRAIN, 5, 42, 2, augment=True)[0][0]
        seq2 = pt.make_epoch_sequence(self.bundle, fx.TRAIN, 5, 42, 2, augment=True)[0][0]
        np.testing.assert_array_equal(seq1["pathology"], seq2["pathology"])

    def test_shuffle_groups_and_integrity(self):
        groups = pt.shuffle_groups(CHANNELS)
        self.assertEqual(list(groups), ["vessel", "MA", "HE", "EX", "SE", "lesions", "all"])
        self.assertEqual(groups["vessel"], (True, ()))
        self.assertEqual((groups["MA"], groups["HE"], groups["EX"], groups["SE"]),
                         ((False, (0, 1)), (False, (2, 3)), (False, (4, 5)), (False, (6, 7))))
        self.assertEqual((groups["lesions"], groups["all"]), ((False, tuple(range(8))), (True, tuple(range(8)))))
        for n in (2, 5, 730):
            perm = pt.derangement(n)
            self.assertFalse(np.any(perm == np.arange(n)))
            np.testing.assert_array_equal(np.sort(perm), np.arange(n))
            np.testing.assert_array_equal(perm, pt.derangement(n))               # fixed

        class Probe:                                                              # one output per input part
            def predict_on_batch(self, x):
                p = x["pathology"]
                return np.stack([x["vessel"].mean((1, 2, 3))] + [p[..., 2 * c:2 * c + 2].mean((1, 2, 3)) for c in range(4)], 1)

        entries = fx.TRAIN + fx.VAL
        base = pt.predict_logits(Probe(), self.bundle, entries, batch_size=2)
        column = {"vessel": [0], "MA": [1], "HE": [2], "EX": [3], "SE": [4], "lesions": [1, 2, 3, 4], "all": [0, 1, 2, 3, 4]}
        for name, changed in column.items():
            got = pt.predict_logits(Probe(), self.bundle, entries, batch_size=2, permute=name)
            for c in range(5):
                if c in changed:
                    self.assertFalse(np.any(got[:, c] == base[:, c]), (name, c))                 # every image got another's
                    np.testing.assert_array_equal(np.sort(got[:, c]), np.sort(base[:, c]))       # ... a permutation of them
                else:
                    np.testing.assert_array_equal(got[:, c], base[:, c], err_msg=f"{name} leaked into column {c}")


class StubBundle:
    """Just the identity the sequence needs (training and evaluation are replaced by recorders)."""
    stage4_sha256, stage3_sha256, stage4_generation, fingerprint = "b" * 64, "c" * 64, "s4v2-bbbbbbbbbbbb-K4", "f" * 64
    channels = CHANNELS
    bundle = {"split_sha256": v2cfg.SPLIT_SHA256, "population_sha256": "e" * 64, "bundle_id": "bundle"}
    grades = np.array([0] * 60 + [1] * 14 + [2] * 34 + [3] * 8 + [4] * 12)
    val_ids = tuple(f"{k:012x}" for k in range(len(grades)))
    train_ids = ("f" * 12,)


def _table_rows(seed, quality, ids=StubBundle.val_ids, grades=StubBundle.grades):
    rng = np.random.default_rng(seed)
    logits = (grades[:, None] - np.arange(4)[None, :] - 0.5) * quality + rng.normal(0, 1.0, (len(grades), 4))
    return at.metrics_from_logits(list(ids), grades, logits)


class SequenceTests(unittest.TestCase):
    def setUp(self):
        self.bundle = StubBundle()
        self.p_tables = {s: ph.table_arrays(pd.DataFrame(_table_rows(s, 2.0)[1])) for s in pt.SEEDS}

    def _fakes(self, calls, fail_at=None):
        import multiseed_runs as msr

        def train(run_dir, bundle, seed, class_weights, *, repo_dir, staging_dir, log):
            calls.append(("train", seed, run_dir, staging_dir, pt.seed_state(run_dir)))
            at.ensure_run_config(run_dir, pt.run_mapping(bundle, seed, class_weights, repo_dir=repo_dir))
            if seed == fail_at:
                raise OSError("simulated Drive failure")
            msr.write_stop_decision(run_dir, 3, "early_stopping")

        def evaluate(run_dir, bundle, seed, class_weights):
            calls.append(("evaluate", seed))
            out = os.path.join(run_dir, "metrics")
            os.makedirs(out, exist_ok=True)
            metrics, rows = _table_rows(100 + seed, 1.2)
            pd.DataFrame(rows).to_csv(os.path.join(out, "per_sample_best.csv"), index=False)
            checkpoint = {"which": "BEST", "weights_sha256": "0" * 64, "best_epoch": 2, "completed_epoch": 3}
            shuffle = {}
            for k, name in enumerate(pt.shuffle_groups(bundle.channels)):
                m, r = _table_rows(200 + seed + k, 1.2 if name == "vessel" else 0.4)
                pd.DataFrame(r).to_csv(os.path.join(out, f"per_sample_best_shuffle_{name}.csv"), index=False)
                shuffle[name] = {"qwk": m["qwk"], "dqwk": m["qwk"] - metrics["qwk"]}
            return {"best": {"metrics": metrics, "checkpoint": checkpoint, "shuffle": shuffle},
                    "last": {"metrics": metrics, "checkpoint": dict(checkpoint, which="LAST")}}

        return train, evaluate

    def _run(self, root, calls, **kw):
        train, evaluate = self._fakes(calls, kw.pop("fail_at", None))
        return pt.run_sequence(root, self.bundle, CW, repo_dir=os.getcwd(), staging_root=os.path.join(root, "staging"),
                               log=lambda *a: None, train_fn=train, evaluate_fn=evaluate,
                               p_tables=kw.pop("p_tables", self.p_tables))

    def test_config_declares_the_fusion_before_training(self):
        mapping = pt.run_mapping(self.bundle, 42, CW)
        self.assertEqual(mapping["fusion"], pf.FUSION)
        self.assertEqual((mapping["experiment"], mapping["rgb_input"], mapping["ema"], mapping["seed"]),
                         ("PathologyGrader", False, "none", 42))
        self.assertEqual(mapping["input_channels"], ["vessel"] + list(CHANNELS))
        for key in ("split_sha256", "bundle_fingerprint", "stage3_sha256", "stage4_sha256", "stage4_generation",
                    "class_weights", "learning_rate", "batch_size", "shuffle_seed"):
            self.assertIn(key, mapping)
        self.assertEqual({k: mapping[k] for k in at.P_PROTOCOL}, at.P_PROTOCOL)   # the P protocol, unchanged
        self.assertNotEqual(at.config_hash(mapping), at.config_hash(pt.run_mapping(self.bundle, 123, CW)))
        self.assertEqual(pt.run_dir_for("/e", "b" * 64, 2026), f"/e/PathologyGrader/pathgrader_{'b' * 12}_seed2026")
        self.assertEqual(pt.summary_dir_for("/e", "b" * 64), f"/e/PathologyGrader/pathgrader_{'b' * 12}_3seed_summary")
        with self.assertRaises(ValueError):
            pt.run_dir_for("/e", "b" * 64, 7)

    def test_three_seeds_in_order_with_results_and_summary(self):
        with tempfile.TemporaryDirectory() as root:
            calls = []
            reports, summary = self._run(root, calls)
            self.assertEqual([c[:2] for c in calls], [("train", 42), ("evaluate", 42), ("train", 123), ("evaluate", 123),
                                                       ("train", 2026), ("evaluate", 2026)])
            self.assertEqual(len({c[3] for c in calls if c[0] == "train"}), 3)
            for seed in pt.SEEDS:
                d = pt.run_dir_for(root, self.bundle.stage4_sha256, seed)
                with open(os.path.join(d, "config.json")) as fh:
                    cfg = json.load(fh)
                self.assertEqual((cfg["seed"], cfg["fusion"]["weights"], cfg["rgb_input"]), (seed, [0.5, 0.5], False))
                with open(os.path.join(d, "result.json")) as fh:
                    stored = json.load(fh)
                self.assertEqual((stored["seed"], stored["control_seed"]), (seed, pf.control_seed(seed)))
                self.assertEqual(set(stored["models"]), {"pathology", "p", "fused", "control"})
                self.assertEqual(set(stored["pathology_shuffle"]), {"vessel", "MA", "HE", "EX", "SE", "lesions", "all"})
                self.assertEqual(set(stored["fused_shuffle"]), {"lesions", "vessel"})
                self.assertEqual(stored["fusion"]["tuned_on_validation"], False)
                self.assertTrue(os.path.exists(os.path.join(d, "result.md")))
                self.assertAlmostEqual(reports[seed]["models"]["p"]["qwk"], ph.metrics(self.p_tables[seed])["qwk"], places=12)
            sdir = pt.summary_dir_for(root, self.bundle.stage4_sha256)
            for name in ("summary.json", "summary.md"):
                self.assertTrue(os.path.exists(os.path.join(sdir, name)))
            self.assertEqual(summary["seeds"], [42, 123, 2026])
            calls.clear()
            self._run(root, calls)                                                # rerun: nothing is repeated
            self.assertEqual(calls, [])

    def test_failure_stops_the_sequence_and_a_rerun_resumes_that_seed(self):
        with tempfile.TemporaryDirectory() as root:
            calls = []
            with self.assertRaises(OSError):
                self._run(root, calls, fail_at=123)
            self.assertEqual([c[:2] for c in calls], [("train", 42), ("evaluate", 42), ("train", 123)])
            sha = self.bundle.stage4_sha256
            self.assertFalse(os.path.exists(pt.run_dir_for(root, sha, 2026)))
            self.assertFalse(os.path.exists(pt.summary_dir_for(root, sha)))
            self.assertEqual([pt.seed_state(pt.run_dir_for(root, sha, s)) for s in pt.SEEDS], ["complete", "in_progress", "new"])
            calls.clear()
            self._run(root, calls)
            self.assertEqual([c[:2] for c in calls], [("train", 123), ("evaluate", 123), ("train", 2026), ("evaluate", 2026)])
            self.assertEqual((calls[0][4], calls[2][4]), ("in_progress", "new"))

    def test_misaligned_p_predictions_are_refused_before_any_training(self):
        with tempfile.TemporaryDirectory() as root:
            calls = []
            wrong = dict(self.p_tables)
            wrong[123] = {k: v[::-1] for k, v in self.p_tables[123].items()}
            with self.assertRaises(RuntimeError):
                self._run(root, calls, p_tables=wrong)
            self.assertEqual(calls, [])
            self.assertEqual(os.listdir(root), [])

    def test_run_directory_is_bound_to_its_seed(self):
        with tempfile.TemporaryDirectory() as root:
            d = pt.run_dir_for(root, self.bundle.stage4_sha256, 42)
            at.ensure_run_config(d, pt.run_mapping(self.bundle, 42, CW))
            with self.assertRaises(RuntimeError):
                at.ensure_run_config(d, pt.run_mapping(self.bundle, 123, CW))


@unittest.skipIf(SLOW, "slow: real two-epoch loop")
class RealLoopTests(unittest.TestCase):
    def test_trains_resumes_evaluates_and_reports_without_the_rgb_cache(self):
        with tempfile.TemporaryDirectory() as tmp:
            bundle, _ = _bundle(tmp)
            shutil.rmtree(bundle.dirs["stage2_rgb_v2"])                           # this branch must never need it
            grade_of = dict(fx.TRAIN + fx.VAL)
            run_dir = os.path.join(tmp, "PathologyGrader", "run_seed42")
            kw = dict(repo_dir=os.getcwd(), staging_dir=os.path.join(tmp, "stage"), grade_of=grade_of, mixed_precision=False)
            pt.train_seed(run_dir, bundle, 42, CW, max_epochs=1, log=lambda *a: None, **kw)
            with open(os.path.join(run_dir, "config.json")) as fh:
                cfg = json.load(fh)
            self.assertEqual((cfg["experiment"], cfg["fusion"], cfg["stage4_sha256"]), ("PathologyGrader", pf.FUSION, fx.MODEL_SHA))
            with open(os.path.join(run_dir, "initialization.json")) as fh:
                init = json.load(fh)
            self.assertEqual((init["seed"], init["initial_epoch"], init["pretrained_weights"]), (42, 0, None))
            import multiseed_runs as msr
            self.assertIsNotNone(msr.read_stop_decision(run_dir))
            os.remove(os.path.join(run_dir, "checkpoints", msr.STOP_DECISION_FILENAME))   # as if cut off mid-run
            messages = []
            pt.train_seed(run_dir, bundle, 42, CW, max_epochs=2, log=lambda *a: messages.append(" ".join(map(str, a))), **kw)
            self.assertTrue(any("resumed at epoch 1" in m for m in messages), messages)
            with open(os.path.join(run_dir, "initialization.json")) as fh:
                self.assertEqual(json.load(fh), init)                             # not re-initialised
            self.assertEqual([h["epoch"] for h in msr.read_history(run_dir)], [1, 2])
            results = pt.evaluate_run(run_dir, bundle, 42, CW, grade_of=grade_of, mixed_precision=False)
            self.assertEqual(set(results), {"best", "last"})
            self.assertEqual(set(results["best"]["shuffle"]), {"vessel", "MA", "HE", "EX", "SE", "lesions", "all"})
            self.assertEqual(len(results["best"]["checkpoint"]["weights_sha256"]), 64)
            best = pd.read_csv(os.path.join(run_dir, "metrics", "per_sample_best.csv"), dtype={"image_id": str})
            self.assertEqual(list(best["image_id"]), [i for i, _ in fx.VAL])
            p = best[[f"p_gt_{k}" for k in range(4)]].to_numpy()
            self.assertTrue(np.all(np.diff(p, axis=1) <= 1e-9) and np.all((p >= 0) & (p <= 1)))   # cumulative CORN probabilities
            np.testing.assert_array_equal(pf.decode(p), best["predicted_grade"].to_numpy())
            for name in results["best"]["shuffle"]:
                self.assertTrue(os.path.exists(os.path.join(run_dir, "metrics", f"per_sample_best_shuffle_{name}.csv")))
            ids, grades = [i for i, _ in fx.VAL], np.array([g for _, g in fx.VAL])
            p_tables = {s: ph.table_arrays(pd.DataFrame(at.metrics_from_logits(
                ids, grades, np.random.default_rng(s).normal(size=(len(ids), 4)))[1])) for s in pt.SEEDS}
            report = pt.seed_result(run_dir, None, results, p_tables=p_tables, indices=[np.arange(len(ids))])
            self.assertEqual((report["seed"], report["status"]), (42, "RECORDED"))
            self.assertEqual(pt.seed_state(run_dir), "complete")
            self.assertEqual(report["identity"]["input_channels"], ["vessel"] + list(CHANNELS))
            with self.assertRaises(RuntimeError):                                 # same directory, another seed
                pt.train_seed(run_dir, bundle, 123, CW, max_epochs=1, **kw)


if __name__ == "__main__":
    unittest.main()
