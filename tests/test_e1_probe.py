"""CPU tests for the E1 mechanism probe (e1_probe). Synthetic features and the 5-image synthetic bundle only;
nothing here is a probe result."""
import os
import tempfile
import unittest

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "-1")

import keras
import numpy as np

import arch1_data as ad
import e1_model as em
import e1_probe as ep
import e1_train as et
import pipeline_v2_config as v2cfg
import pl_convnext as pl

try:
    from tests import v2_bundle_fixture as fx
except ImportError:
    import v2_bundle_fixture as fx

PRIOR = (0.07, 0.11, 0.11, 0.05)


class MetricTests(unittest.TestCase):
    def test_cell_auroc_equals_sklearn(self):
        from sklearn.metrics import roc_auc_score
        rng = np.random.default_rng(0)
        for _ in range(5):
            y = rng.random(5000) < 0.1
            s = rng.normal(0, 1, 5000) + 0.8 * y
            self.assertAlmostEqual(ep.cell_auroc(y, s), roc_auc_score(y, s), places=12)
        self.assertTrue(np.isnan(ep.cell_auroc(np.zeros(10, bool), rng.normal(size=10))))

    def test_probe_scores_use_target_ge_half_per_class(self):
        rng = np.random.default_rng(1)
        t = rng.random((6, 16, 16, 4)).astype("float32")
        z = rng.normal(size=t.shape).astype("float32")
        scores = ep.probe_scores(t, z)
        self.assertEqual(set(scores), {"MA", "HE", "EX", "SE", "mean"})
        self.assertAlmostEqual(scores["HE"], ep.cell_auroc(t[..., 1] >= 0.5, z[..., 1]), places=12)
        self.assertAlmostEqual(scores["mean"], np.mean([scores[c] for c in em.LESION_CLASSES]), places=12)

    def test_protocol_is_the_preregistered_one(self):
        self.assertEqual((ep.PROBE["learning_rate"], ep.PROBE["batch_size"], ep.PROBE["epochs"]), (1e-3, 16, 5))
        self.assertEqual((ep.PROBE["optimizer"], ep.PROBE["augmentation"], ep.PROBE["checkpoint"]), ("Adam", "none", "final epoch"))
        self.assertEqual((ep.N_BOOT, ep.BOOT_SEED), (2000, 20260927))


class ProbeTests(unittest.TestCase):
    def test_fresh_probe_is_identical_for_both_encoders_and_learns(self):
        rng = np.random.default_rng(2)
        w = rng.normal(0, 1, (32, 4)).astype("float32")
        features = rng.normal(0, 1, (48, 4, 4, 32)).astype("float16")
        targets = (1 / (1 + np.exp(-(features.astype("float32") @ w)))).astype("float32")
        a = ep.build_probe(42, PRIOR, features.shape[1:])
        b = ep.build_probe(42, PRIOR, features.shape[1:])
        for x, y in zip(a.get_weights(), b.get_weights()):
            np.testing.assert_array_equal(x, y)                            # same fresh probe for P and E1
        np.testing.assert_allclose(a.get_weights()[1], em.prior_logits(PRIOR), rtol=1e-6)
        self.assertEqual(a.count_params(), 32 * 4 + 4)
        logits, curve, final = ep.train_probe(42, PRIOR, features[:32], targets[:32], features[32:], targets[32:])
        with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as tmp:   # the same from memory-mapped features
            # (Windows keeps a mapped file open until the map is collected; the leftover temp file is harmless)
            np.save(os.path.join(tmp, "tr.npy"), features[:32])
            np.save(os.path.join(tmp, "va.npy"), features[32:])
            tr, va = np.load(os.path.join(tmp, "tr.npy"), mmap_mode="r"), np.load(os.path.join(tmp, "va.npy"), mmap_mode="r")
            again, curve2, _ = ep.train_probe(42, PRIOR, tr, targets[:32], va, targets[32:])
            del tr, va
        self.assertEqual(logits.shape, (16, 4, 4, 4))
        self.assertEqual([c["epoch"] for c in curve], [0, 1, 2, 3, 4, 5])
        self.assertLess(curve[-1]["val_loss"], curve[0]["val_loss"])
        self.assertGreater(final["val_scores"]["mean"], curve[0]["val_mean_cell_auroc"])
        self.assertGreater(final["kernel_moved"], 0)
        np.testing.assert_allclose(logits, again, atol=1e-5)               # deterministic order and initialisation
        self.assertEqual(keras.mixed_precision.global_policy().name, "float32")


class AnalysisTests(unittest.TestCase):
    def _write(self, tmp, shift):
        rng = np.random.default_rng(3)
        n = 60
        targets = (rng.random((n, 16, 16, 4)) < 0.15).astype("float32")
        ids = np.asarray([f"{i:012x}" for i in range(n)])
        np.savez_compressed(os.path.join(tmp, "validation_targets.npz"), targets=targets, image_ids=ids)
        for seed in ep.SEEDS:
            for kind, strength in (("p", 1.0), ("e1", 1.0 + shift)):
                logits = (rng.normal(0, 1, targets.shape) + strength * targets).astype("float32")
                np.savez_compressed(os.path.join(tmp, f"probe_{kind}_seed{seed}.npz"), val_logits=logits,
                                    val_grading_logits=np.zeros((n, 4)), image_ids=ids)
        return np.asarray([i % 5 for i in range(n)])

    def test_criterion_supported_when_e1_is_clearly_better(self):
        with tempfile.TemporaryDirectory() as tmp:
            grades = self._write(tmp, shift=0.5)
            r = ep.analyse(tmp, grades, n_boot=60, log=None)
            self.assertEqual(r["positive_seeds"], 3)
            self.assertTrue(r["ci_excludes_zero"] and r["mechanism_supported"])
            self.assertGreater(r["mean_difference_ci"][0], 0)
            self.assertAlmostEqual(r["mean_difference"], np.mean([r["per_seed"][s]["difference"]["mean"] for s in ep.SEEDS]))

    def test_criterion_not_supported_when_the_probes_are_equivalent(self):
        with tempfile.TemporaryDirectory() as tmp:
            grades = self._write(tmp, shift=0.0)
            r = ep.analyse(tmp, grades, n_boot=60, log=None)
            self.assertFalse(r["mechanism_supported"])
            with self.assertRaises(RuntimeError):
                ep.analyse(tmp, grades[:-1], n_boot=5, log=None)


class EncoderTests(unittest.TestCase):
    def test_frozen_encoders_share_one_graph_and_p_features_are_p(self):
        from training import checkpointing as ckpt
        with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as tmp:
            b = fx.build(os.path.join(tmp, "data"))
            bundle = ad.Arch1Bundle(expected_bundle_id=b["bundle"]["bundle_id"], expected_split_sha256=v2cfg.SPLIT_SHA256,
                                    expected_stage4_sha256=fx.MODEL_SHA, roots=b["roots"], expected_population=5)
            keras.utils.set_random_seed(7)
            ref = keras.applications.ConvNeXtTiny(include_top=False, weights=None, include_preprocessing=True, pooling="avg",
                                                  input_shape=(512, 512, 3), name=pl.BACKBONE_NAME)
            arrays = [np.asarray(v.numpy()) for v in ref.weights if "prestem" not in v.path]
            p = pl.build_pl_model("P", 42, arrays)
            p_path = os.path.join(tmp, "p.weights.h5")
            ckpt.save_model_weights_only(p, p_path)
            e1 = em.build_e1_model(42, PRIOR, ref)
            e1_path = os.path.join(tmp, "e1.weights.h5")
            ckpt.save_model_weights_only(e1, e1_path)
            ids = list(bundle.val_ids)
            out = {}
            for kind, path in (("p", p_path), ("e1", e1_path)):
                features, full = ep.build_frozen_encoder(kind, 42, path, PRIOR, arrays, policy="float32")
                self.assertFalse(full.trainable_variables)                # nothing can be trained
                out[kind] = ep.extract(bundle, features, full, ids, os.path.join(tmp, "feat", f"{kind}.npy"))
                self.assertIsInstance(out[kind][0], np.memmap)             # on disk, not held in RAM
                self.assertEqual((out[kind][0].shape, out[kind][0].dtype), ((2, 16, 16, 768), np.float16))
            np.testing.assert_array_equal(np.asarray(out["p"][0]), np.asarray(out["e1"][0]))   # same weights -> same features
            mapped = np.asarray(out["p"][0])
            out = {k: (np.array(v[0]), v[1]) for k, v in out.items()}       # release the files before cleanup
            del mapped
            import e1_gates as eg
            rgb = np.stack([__import__("e1_data").load_inputs(bundle, i)["rgb"] for i in ids])
            np.testing.assert_allclose(out["p"][1], np.asarray(p.predict_on_batch(eg.p_inputs(rgb)), np.float64), atol=1e-4)
            self.assertEqual(ep.targets_for(bundle, ids).shape, (2, 16, 16, 4))
            self.assertEqual(keras.mixed_precision.global_policy().name, "float32")


if __name__ == "__main__":
    unittest.main()
