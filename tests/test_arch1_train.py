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
            res = at.evaluate_run(run_dir, bundle, 42, ref, CW, grade_of=grade_of, mixed_precision=False)
            self.assertEqual(set(res), {"best", "last"})
            self.assertIn("pathology", res["best"]["permutation"])
            self.assertEqual(len(res["best"]["checkpoint"]["weights_sha256"]), 64)
            self.assertTrue(os.path.exists(os.path.join(run_dir, "metrics", "per_sample_best.csv")))
            with self.assertRaises(RuntimeError):                            # same dir, different protocol
                at.train_seed(run_dir, bundle, 42, ref, CW, repo_dir=os.getcwd(), staging_dir=os.path.join(tmp, "s"),
                              protocol=dict(at.P_PROTOCOL, learning_rate=3e-4), max_epochs=1, grade_of=grade_of,
                              mixed_precision=False)


if __name__ == "__main__":
    unittest.main()
