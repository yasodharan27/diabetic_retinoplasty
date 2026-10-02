"""CPU tests for arch1_train.py. The training loop is exercised for ONE epoch on the 5-image synthetic bundle
(random ConvNeXt reference weights, float32) to prove checkpoint/metadata/resume mechanics -- this is not an
Architecture-1 experiment and produces no result."""
import json
import os
import tempfile
import unittest

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "-1")

import keras
import numpy as np

import arch1_data as ad
import arch1_model as a1
import arch1_train as at
import pipeline_v2_config as v2cfg
import pl_convnext as pl
import stage34_cache_v2 as cache
import weighted_corn

try:
    from tests import v2_bundle_fixture as fx
except ImportError:
    import v2_bundle_fixture as fx

CW = list(weighted_corn.PREREGISTERED_CLASS_WEIGHTS)


def _reference(size):
    keras.utils.set_random_seed(7)
    from keras.applications import ConvNeXtTiny
    ref = ConvNeXtTiny(include_top=False, weights=None, include_preprocessing=True, pooling="avg",
                       input_shape=(size, size, 3), name=pl.BACKBONE_NAME)
    for v in ref.weights:
        if "prestem" not in v.path:
            v.assign(np.random.default_rng(len(v.path)).normal(0, 0.05, v.shape).astype("float32")
                     + (1.0 if v.path.endswith("gamma") else 0.0))
    return ref


class ProtocolTests(unittest.TestCase):
    def test_locked_p_protocol(self):
        prereg = {"batch_size": 2, "max_epochs": 50, "learning_rate": 1e-4, "weight_decay": 0.05,
                  "primary_metric": "val_QWK", "early_stopping": {"patience": 12},
                  "reduce_lr_on_plateau": {"patience": 4, "factor": 0.5, "min_lr": 1e-6}}
        self.assertTrue(at.assert_protocol(prereg, CW))
        with self.assertRaises(RuntimeError):
            at.assert_protocol(dict(prereg, learning_rate=3e-4), CW)
        with self.assertRaises(RuntimeError):
            at.assert_protocol(prereg, [1.0] * 5)

    def test_one_seed_checks(self):
        p42 = {"qwk": 0.90, "auroc_ge3_g4_vs_g012": 0.80, "grade3_recall": 0.50, "false_urgent_rate": 0.05}
        good = {"metrics": {"qwk": 0.89, "auroc_ge3_g4_vs_g012": 0.795, "grade3_recall": 0.45, "false_urgent_rate": 0.06},
                "permutation": {"pathology": {"dqwk": -0.02}}}
        self.assertTrue(at.one_seed_checks(good, p42)["all_pass"])
        no_use = dict(good, permutation={"pathology": {"dqwk": -0.005}})          # Q not used
        self.assertFalse(at.one_seed_checks(no_use, p42)["checks"]["q_permutation_contributes"])
        worse = dict(good, metrics=dict(good["metrics"], qwk=0.87))
        self.assertFalse(at.one_seed_checks(worse, p42)["checks"]["noninferior_qwk"])

    def test_final_verdict_opens_only_when_every_preset_criterion_passes(self):
        p42 = {"qwk": 0.90, "auroc_ge3_g4_vs_g012": 0.80, "grade3_recall": 0.50, "false_urgent_rate": 0.05}
        cfg = {"seed": 42, "git_commit": "abc", "stage4_sha256": "s4", "stage3_sha256": "s3", "ema": at.EMA}

        def best(dqwk, qwk=0.89):
            res = {"metrics": {"qwk": qwk, "auroc_ge3_g4_vs_g012": 0.795, "grade3_recall": 0.45,
                               "false_urgent_rate": 0.06},
                   "permutation": {k: {"qwk": qwk + dqwk, "dqwk": dqwk, "auroc_ge3_g4_vs_g012": 0.7}
                                   for k in ("pathology", "vessel", "both")},
                   "checkpoint": {"which": "BEST", "weights_sha256": "0" * 64, "best_epoch": 3}}
            res["one_seed_checks"] = at.one_seed_checks(res, p42)
            return res

        opened = at.final_verdict(cfg, best(-0.02), p42, c2_pass=False)
        self.assertEqual(opened["status"], "OPEN")
        self.assertAlmostEqual(opened["criteria"]["lesion_shuffle"]["qwk_drop"], 0.02)
        self.assertIn("FAILED", opened["statements"]["c2"])
        self.assertIn("No EMA", opened["statements"]["ema"])
        self.assertEqual(opened["identity"]["ema"], "none")
        ignored = at.final_verdict(cfg, best(-0.005), p42, c2_pass=False)       # maps not used -> closed
        self.assertEqual((ignored["status"], ignored["failed_criteria"]), ("CLOSED", ["lesion_shuffle"]))
        worse = at.final_verdict(cfg, best(-0.02, qwk=0.87), p42, c2_pass=False)
        self.assertEqual((worse["status"], worse["failed_criteria"]), ("CLOSED", ["qwk"]))
        text = at.verdict_markdown(opened)
        for needle in ("OPEN", "exploratory", "No EMA", "lesion-shuffle QWK drop", "not statistically conclusive"):
            self.assertIn(needle, text)

    def test_metrics_from_logits(self):
        rng = np.random.default_rng(0)
        grades = np.array([0, 1, 2, 3, 4] * 4)
        logits = rng.normal(size=(20, 4)) + (grades[:, None] - np.arange(4)) * 2
        m, rows = at.metrics_from_logits([f"{k:012x}" for k in range(20)], grades, logits)
        for key in ("qwk", "mae", "auroc_cut_ge1", "auroc_cut_ge4", "auroc_ge3_g4_vs_g012", "grade3_recall",
                    "recall_per_grade", "false_urgent_rate"):
            self.assertIn(key, m)
        self.assertEqual(len(rows), 20)
        self.assertGreater(m["qwk"], 0.5)


class ModelInputTests(unittest.TestCase):
    """Small (64 px) model: both priors influence the output once the zero-initialised gates move."""

    def test_vessel_and_pathology_each_affect_output_after_gates_open(self):
        size = 64
        channels = cache.channel_names(v2cfg.STAGE4_V2A_CLASSES)
        model = at.build_compiled_model(channels, 42, _reference(size), CW, mixed_precision=False, image_size=size)
        rng = np.random.default_rng(1)
        x = {"rgb": rng.uniform(0, 1, (2, size, size, 3)).astype("float32"),
             "vessel": rng.uniform(0, 1, (2, size, size, 1)).astype("float32"),
             "pathology": rng.uniform(0, 1, (2, size, size, 8)).astype("float32")}
        base = model.predict_on_batch(x)
        np.testing.assert_array_equal(base, model.predict_on_batch(dict(x, vessel=1 - x["vessel"])))  # zero gates
        for layer in a1.injection_layers(model):
            layer.gamma.assign(np.full(layer.gamma.shape, 0.5, "float32"))
        base = model.predict_on_batch(x)
        self.assertGreater(np.abs(model.predict_on_batch(dict(x, vessel=1 - x["vessel"])) - base).max(), 1e-5)
        self.assertGreater(np.abs(model.predict_on_batch(dict(x, pathology=1 - x["pathology"])) - base).max(), 1e-5)

    def test_permutation_is_a_derangement_and_changes_only_the_prior(self):
        class Probe:
            def predict_on_batch(self, x):
                return np.stack([x["rgb"].mean((1, 2, 3)), x["vessel"].mean((1, 2, 3)),
                                 x["pathology"].mean((1, 2, 3)), np.zeros(len(x["rgb"]))], 1)

        with tempfile.TemporaryDirectory() as tmp:
            b = fx.build(tmp)
            bundle = ad.Arch1Bundle(expected_bundle_id=b["bundle"]["bundle_id"], expected_split_sha256=v2cfg.SPLIT_SHA256,
                                    expected_stage4_sha256=fx.MODEL_SHA, roots=b["roots"], expected_population=5)
            entries = fx.TRAIN
            base = at.predict_logits(Probe(), bundle, entries, batch_size=2)
            perm = at.predict_logits(Probe(), bundle, entries, batch_size=2, permute="pathology")
            np.testing.assert_array_equal(base[:, :2], perm[:, :2])         # RGB and vessel untouched
            self.assertFalse(np.any(base[:, 2] == perm[:, 2]))              # every image got another's maps
            np.testing.assert_array_equal(np.sort(base[:, 2]), np.sort(perm[:, 2]))
            both = at.predict_logits(Probe(), bundle, entries, batch_size=2, permute="both")
            np.testing.assert_array_equal(base[:, 0], both[:, 0])


class SeedIsolationTests(unittest.TestCase):
    def test_every_seed_starts_from_a_fresh_initialisation(self):
        size = 64
        channels = cache.channel_names(v2cfg.STAGE4_V2A_CLASSES)
        ref = _reference(size)
        ref_before = [w.numpy().copy() for w in ref.weights]

        def build(seed):
            return at.build_compiled_model(channels, seed, ref, CW, mixed_precision=False, image_size=size)

        pristine_123 = at.weights_sha256(build(123))
        m42 = build(42)
        start_42 = at.weights_sha256(m42)
        backbone_42 = [w.numpy().copy() for layer in a1.backbone_layers(m42) for w in layer.weights]
        rng = np.random.default_rng(1)
        x = {"rgb": rng.uniform(0, 1, (2, size, size, 3)).astype("float32"),
             "vessel": rng.uniform(0, 1, (2, size, size, 1)).astype("float32"),
             "pathology": rng.uniform(0, 1, (2, size, size, 8)).astype("float32")}
        for _ in range(2):
            m42.train_on_batch(x, np.array([1, 3], "int32"))
        self.assertNotEqual(at.weights_sha256(m42), start_42)                 # seed 42 has learned something
        m123 = build(123)                                                     # built AFTER seed 42 trained
        self.assertEqual(at.weights_sha256(m123), pristine_123)               # ...and inherits none of it
        self.assertEqual(at.weights_sha256(build(42)), start_42)              # seed 42's start is reproducible
        self.assertNotEqual(pristine_123, start_42)                           # the seeds differ (priors, head)
        backbone_123 = [w.numpy() for layer in a1.backbone_layers(m123) for w in layer.weights]
        for a, b in zip(backbone_42, backbone_123):                           # same frozen pretrained start
            np.testing.assert_array_equal(a, b)
        for before, w in zip(ref_before, ref.weights):                        # the reference is never trained
            np.testing.assert_array_equal(before, w.numpy())
        self.assertTrue(all(float(np.abs(l.gamma.numpy()).max()) == 0.0 for l in a1.injection_layers(m123)))


class ThreeSeedSequenceTests(unittest.TestCase):
    """Orchestration only: training and evaluation are replaced by recorders, so nothing is trained."""
    P42 = {"qwk": 0.90, "auroc_ge3_g4_vs_g012": 0.80, "grade3_recall": 0.50, "false_urgent_rate": 0.05}

    @classmethod
    def setUpClass(cls):
        cls._tmp = tempfile.TemporaryDirectory()
        b = fx.build(os.path.join(cls._tmp.name, "data"))
        cls.bundle = ad.Arch1Bundle(expected_bundle_id=b["bundle"]["bundle_id"], expected_split_sha256=v2cfg.SPLIT_SHA256,
                                    expected_stage4_sha256=fx.MODEL_SHA, roots=b["roots"], expected_population=5)

    @classmethod
    def tearDownClass(cls):
        cls._tmp.cleanup()

    def _fakes(self, calls, fail_at=None, unfinished_at=None, qwk=None):
        import multiseed_runs as msr
        qwk = qwk or {42: 0.91, 123: 0.89, 2026: 0.90}

        def train(run_dir, bundle, seed, reference, class_weights, *, repo_dir, staging_dir, log):
            calls.append(("train", seed, run_dir, staging_dir, at.seed_state(run_dir)))
            at.ensure_run_config(run_dir, at.run_mapping(bundle, seed, class_weights, repo_dir=repo_dir))
            if seed == fail_at:
                raise OSError("simulated Drive failure")
            if seed != unfinished_at:
                msr.write_stop_decision(run_dir, 3, "early_stopping")

        def evaluate(run_dir, bundle, seed, reference, class_weights, p42_metrics):
            calls.append(("evaluate", seed, run_dir))
            res = {"metrics": {"qwk": qwk[seed], "auroc_ge3_g4_vs_g012": 0.80, "grade3_recall": 0.50,
                               "false_urgent_rate": 0.05},
                   "permutation": {k: {"qwk": qwk[seed] - 0.03, "dqwk": -0.03, "auroc_ge3_g4_vs_g012": 0.7}
                                   for k in ("pathology", "vessel", "both")},
                   "checkpoint": {"which": "BEST", "weights_sha256": "0" * 64, "best_epoch": 2}}
            res["one_seed_checks"] = at.one_seed_checks(res, p42_metrics)
            return {"best": res, "last": res}

        return train, evaluate

    def _run(self, root, calls, **kw):
        train, evaluate = self._fakes(calls, **kw)
        return at.run_sequence(root, self.bundle, None, CW, self.P42, False, repo_dir=os.getcwd(),
                               staging_root=os.path.join(root, "staging"), log=lambda *a: None,
                               train_fn=train, evaluate_fn=evaluate)

    def test_seeds_run_in_order_each_in_its_own_directory_then_the_summary(self):
        self.assertEqual(at.SEEDS, (42, 123, 2026))
        with tempfile.TemporaryDirectory() as root:
            calls = []
            verdicts, summary = self._run(root, calls)
            self.assertEqual([(c[0], c[1]) for c in calls], [("train", 42), ("evaluate", 42), ("train", 123),
                                                              ("evaluate", 123), ("train", 2026), ("evaluate", 2026)])
            dirs = [at.run_dir_for(root, fx.MODEL_SHA, s) for s in at.SEEDS]
            self.assertEqual([os.path.basename(d) for d in dirs],
                             [f"arch1_{'b' * 12}_seed{s}" for s in at.SEEDS])
            self.assertEqual(len({c[3] for c in calls if c[0] == "train"}), 3)          # separate staging too
            hashes = set()
            for seed, d in zip(at.SEEDS, dirs):
                with open(os.path.join(d, "config.json")) as fh:
                    cfg = json.load(fh)
                self.assertEqual((cfg["seed"], cfg["ema"], cfg["stage4_sha256"]), (seed, "none", fx.MODEL_SHA))
                hashes.add(cfg["config_hash"])
                for name in ("verdict.json", "verdict.md"):
                    self.assertTrue(os.path.exists(os.path.join(d, name)))
                self.assertEqual(verdicts[seed]["seed"], seed)
                self.assertIn(f"Seed {seed} of the sequence 42, 123, 2026", verdicts[seed]["statements"]["scope"])
            self.assertEqual(len(hashes), 3)
            sdir = at.summary_dir_for(root, fx.MODEL_SHA)
            self.assertEqual(os.path.basename(sdir), f"arch1_{'b' * 12}_3seed_summary")
            for name in ("summary.json", "summary.md"):
                self.assertTrue(os.path.exists(os.path.join(sdir, name)))
            self.assertEqual(summary["route"], "VIABLE")
            calls.clear()                                                    # a rerun keeps completed seeds as recorded
            again, _ = self._run(root, calls)
            self.assertEqual(calls, [])
            self.assertEqual({s: v["status"] for s, v in again.items()}, {s: "OPEN" for s in at.SEEDS})

    def test_a_failed_seed_stops_the_sequence_and_a_rerun_resumes_that_seed(self):
        with tempfile.TemporaryDirectory() as root:
            calls = []
            with self.assertRaises(OSError):
                self._run(root, calls, fail_at=123)
            self.assertEqual([(c[0], c[1]) for c in calls], [("train", 42), ("evaluate", 42), ("train", 123)])
            self.assertFalse(os.path.exists(at.run_dir_for(root, fx.MODEL_SHA, 2026)))   # never started
            self.assertFalse(os.path.exists(at.summary_dir_for(root, fx.MODEL_SHA)))     # no partial summary
            self.assertEqual([at.seed_state(at.run_dir_for(root, fx.MODEL_SHA, s)) for s in at.SEEDS],
                             ["complete", "in_progress", "new"])
            calls.clear()
            _, summary = self._run(root, calls)
            self.assertEqual([(c[0], c[1]) for c in calls], [("train", 123), ("evaluate", 123), ("train", 2026),
                                                              ("evaluate", 2026)])              # 42 is not retrained
            self.assertEqual(calls[0][2], at.run_dir_for(root, fx.MODEL_SHA, 123))               # same directory
            self.assertEqual((calls[0][4], calls[2][4]), ("in_progress", "new"))                 # resume vs fresh
            self.assertEqual(summary["seeds"], [42, 123, 2026])

    def test_a_seed_that_stops_without_a_stop_decision_is_fatal_not_skipped(self):
        with tempfile.TemporaryDirectory() as root:
            calls = []
            with self.assertRaises(RuntimeError):
                self._run(root, calls, unfinished_at=42)
            self.assertEqual([(c[0], c[1]) for c in calls], [("train", 42)])                     # no evaluation, no seed 123
            self.assertFalse(os.path.exists(at.summary_dir_for(root, fx.MODEL_SHA)))

    def test_run_directories_are_bound_to_their_seed(self):
        with tempfile.TemporaryDirectory() as root:
            with self.assertRaises(ValueError):
                at.run_dir_for(root, fx.MODEL_SHA, 7)
            d42 = at.run_dir_for(root, fx.MODEL_SHA, 42)
            at.ensure_run_config(d42, at.run_mapping(self.bundle, 42, CW))
            at.ensure_run_config(d42, at.run_mapping(self.bundle, 42, CW))                       # same seed: fine
            with self.assertRaises(RuntimeError):
                at.ensure_run_config(d42, at.run_mapping(self.bundle, 123, CW))
            calls = []
            self._run(root, calls)
            with open(os.path.join(d42, "verdict.json")) as fh:                                  # a misfiled verdict
                wrong = dict(json.load(fh), seed=123)
            with open(os.path.join(d42, "verdict.json"), "w") as fh:
                json.dump(wrong, fh)
            with self.assertRaises(RuntimeError):
                self._run(root, calls)


class AggregateTests(unittest.TestCase):
    P42 = {"qwk": 0.90, "auroc_ge3_g4_vs_g012": 0.80, "grade3_recall": 0.50, "false_urgent_rate": 0.05}

    def _verdict(self, seed, qwk, dqwk=-0.03, **identity):
        cfg = dict({"seed": seed, "git_commit": "abc", "config_hash": f"h{seed}", "stage4_sha256": "s4",
                    "stage3_sha256": "s3", "stage4_generation": "g4", "bundle_id": "b", "bundle_fingerprint": "f",
                    "split_sha256": "sp", "population_sha256": "pop", "ema": at.EMA}, **identity)
        res = {"metrics": {"qwk": qwk, "auroc_ge3_g4_vs_g012": 0.80, "grade3_recall": 0.50, "false_urgent_rate": 0.05},
               "permutation": {k: {"qwk": qwk + dqwk, "dqwk": dqwk, "auroc_ge3_g4_vs_g012": 0.7}
                               for k in ("pathology", "vessel", "both")},
               "checkpoint": {"which": "BEST", "weights_sha256": "0" * 64, "best_epoch": 4}}
        res["one_seed_checks"] = at.one_seed_checks(res, self.P42)
        return at.final_verdict(cfg, res, self.P42, c2_pass=False)

    def test_mean_sd_and_pass_counts(self):
        v = {42: self._verdict(42, 0.91), 123: self._verdict(123, 0.89), 2026: self._verdict(2026, 0.93, dqwk=-0.02)}
        a = at.aggregate(v)
        self.assertAlmostEqual(a["stats"]["qwk"]["mean"], 0.91)
        self.assertAlmostEqual(a["stats"]["qwk"]["sd"], 0.02)                                    # n-1
        self.assertAlmostEqual(a["stats"]["lesion_shuffle"]["mean"], (0.03 + 0.03 + 0.02) / 3)
        self.assertEqual(a["seeds_passing_each_criterion"], {k: 3 for k in at.CRITERIA})
        self.assertEqual((a["route"], a["seeds_passing_all"]), ("VIABLE", [42, 123, 2026]))
        self.assertEqual(a["per_seed"][123]["values"]["qwk"], 0.89)
        self.assertIn("not a pre-registered superiority experiment", a["statements"]["scope"])
        self.assertNotIn("superior to", a["statements"]["reading"])
        text = at.summary_markdown(a)
        for needle in ("VIABLE", "seed 2026", "3/3", "No EMA", "FAILED", "No superiority"):
            self.assertIn(needle, text)

    def test_mixed_and_closed_routes(self):
        mixed = at.aggregate({42: self._verdict(42, 0.91), 123: self._verdict(123, 0.87),        # QWK below 0.88
                              2026: self._verdict(2026, 0.92, dqwk=-0.004)})                     # maps not used
        self.assertEqual((mixed["route"], mixed["seeds_passing_all"]), ("MIXED", [42]))
        self.assertEqual(mixed["seeds_passing_each_criterion"]["qwk"], 2)
        self.assertEqual(mixed["seeds_passing_each_criterion"]["lesion_shuffle"], 2)
        self.assertIn("inconsistent", mixed["statements"]["reading"])
        self.assertIn("Nothing is tuned", mixed["statements"]["reading"])
        closed = at.aggregate({s: self._verdict(s, 0.92, dqwk=-0.001) for s in at.SEEDS})
        self.assertEqual((closed["route"], closed["seeds_passing_all"]), ("CLOSED", []))
        self.assertIn("closed", closed["statements"]["reading"])

    def test_refuses_incomplete_or_inconsistent_sets(self):
        v = {s: self._verdict(s, 0.91) for s in at.SEEDS}
        with self.assertRaises(RuntimeError):                                                    # only two seeds
            at.aggregate({s: v[s] for s in (42, 123)})
        with self.assertRaises(RuntimeError):                                                    # another Stage-4 model
            at.aggregate(v | {2026: self._verdict(2026, 0.91, stage4_sha256="other")})
        with self.assertRaises(RuntimeError):                                                    # the same run twice
            at.aggregate(v | {123: self._verdict(123, 0.91, config_hash="h42")})
        with tempfile.TemporaryDirectory() as tmp:
            with self.assertRaises(RuntimeError):
                at.write_summary(os.path.join(tmp, "s"), {42: v[42]})
            self.assertFalse(os.path.exists(os.path.join(tmp, "s")))


class TrainingLoopTests(unittest.TestCase):
    def test_one_epoch_run_writes_fingerprinted_checkpoints_and_refuses_changed_config(self):
        with tempfile.TemporaryDirectory() as tmp:
            b = fx.build(os.path.join(tmp, "data"))
            bundle = ad.Arch1Bundle(expected_bundle_id=b["bundle"]["bundle_id"], expected_split_sha256=v2cfg.SPLIT_SHA256,
                                    expected_stage4_sha256=fx.MODEL_SHA, roots=b["roots"], expected_population=5)
            grade_of = dict(fx.TRAIN + fx.VAL)
            run_dir = os.path.join(tmp, "Architecture1", "run_seed42")
            ref = _reference(512)
            at.train_seed(run_dir, bundle, 42, ref, CW, repo_dir=os.getcwd(), staging_dir=os.path.join(tmp, "stage"),
                          max_epochs=1, grade_of=grade_of, mixed_precision=False, log=lambda *a: None)
            with open(os.path.join(run_dir, "config.json")) as fh:
                cfg = json.load(fh)
            for key in ("stage4_sha256", "stage3_sha256", "bundle_fingerprint", "split_sha256", "population_sha256",
                        "seed", "learning_rate", "batch_size", "class_weights", "config_hash", "stage4_generation"):
                self.assertIn(key, cfg)
            self.assertEqual(cfg["stage4_sha256"], fx.MODEL_SHA)
            self.assertTrue(os.path.isdir(os.path.join(run_dir, "checkpoints")))
            import multiseed_runs as msr
            self.assertIsNotNone(msr.read_stop_decision(run_dir))            # epoch cap reached
            p42 = {"qwk": 0.90, "auroc_ge3_g4_vs_g012": 0.80, "grade3_recall": 0.50, "false_urgent_rate": 0.05}
            res = at.evaluate_run(run_dir, bundle, 42, ref, CW, p42_metrics=p42, grade_of=grade_of,
                                  mixed_precision=False)
            self.assertEqual(set(res), {"best", "last"})
            self.assertIn("pathology", res["best"]["permutation"])
            self.assertEqual(len(res["best"]["checkpoint"]["weights_sha256"]), 64)
            self.assertEqual(cfg["ema"], "none")
            self.assertEqual(res["last"]["checkpoint"]["completed_epoch"], 1)
            verdict = at.write_verdict(run_dir, res, p42, c2_pass=False, sources={"p42": "fixture"})
            self.assertIn(verdict["status"], ("OPEN", "CLOSED"))
            self.assertEqual(len(verdict["history"]), 1)
            self.assertEqual(verdict["identity"]["stage4_sha256"], fx.MODEL_SHA)
            for name in ("verdict.json", "verdict.md"):
                self.assertTrue(os.path.exists(os.path.join(run_dir, name)))
            self.assertTrue(os.path.exists(os.path.join(run_dir, "metrics", "per_sample_best.csv")))
            # Resume: an interrupted seed continues from ITS checkpoint; it is not re-initialised.
            with open(os.path.join(run_dir, "initialization.json")) as fh:
                init = json.load(fh)
            self.assertEqual((init["seed"], init["initial_epoch"], len(init["weights_sha256"])), (42, 0, 64))
            self.assertEqual(verdict["initialization"], init)
            os.remove(os.path.join(run_dir, "checkpoints", msr.STOP_DECISION_FILENAME))   # as if cut off mid-run
            self.assertEqual(at.seed_state(run_dir), "complete")             # the verdict still marks it done
            messages = []
            at.train_seed(run_dir, bundle, 42, ref, CW, repo_dir=os.getcwd(), staging_dir=os.path.join(tmp, "stage"),
                          max_epochs=2, grade_of=grade_of, mixed_precision=False,
                          log=lambda *a: messages.append(" ".join(map(str, a))))
            self.assertTrue(any("resumed at epoch 1" in m for m in messages), messages)
            with open(os.path.join(run_dir, "initialization.json")) as fh:
                self.assertEqual(json.load(fh), init)                        # not rewritten on resume
            self.assertEqual([h["epoch"] for h in msr.read_history(run_dir)], [1, 2])
            with self.assertRaises(RuntimeError):                            # same dir, another seed
                at.train_seed(run_dir, bundle, 123, ref, CW, repo_dir=os.getcwd(), staging_dir=os.path.join(tmp, "s"),
                              max_epochs=1, grade_of=grade_of, mixed_precision=False)
            with self.assertRaises(RuntimeError):                            # same dir, different protocol
                at.train_seed(run_dir, bundle, 42, ref, CW, repo_dir=os.getcwd(), staging_dir=os.path.join(tmp, "s"),
                              protocol=dict(at.P_PROTOCOL, learning_rate=3e-4), max_epochs=1, grade_of=grade_of,
                              mixed_precision=False)


if __name__ == "__main__":
    unittest.main()
