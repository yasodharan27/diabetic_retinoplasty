"""CPU tests for arch1_data.py on a tiny synthetic v2 bundle built through the real APTOS-cache pipeline
(tests/v2_bundle_fixture.py): loading, every downstream-safety refusal, and P-protocol epoch batches."""
import json
import os
import shutil
import tempfile
import unittest

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "-1")

import numpy as np

import arch1_data as ad
import pipeline_v2_config as v2cfg
import stage34_cache_v2 as cache

try:
    from tests import v2_bundle_fixture as fx
except ImportError:
    import v2_bundle_fixture as fx


class BundleTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tmp = tempfile.TemporaryDirectory()
        cls.b = fx.build(cls.tmp.name)

    @classmethod
    def tearDownClass(cls):
        cls.tmp.cleanup()

    def open(self, roots=None, **kw):
        args = dict(expected_bundle_id=self.b["bundle"]["bundle_id"], expected_split_sha256=v2cfg.SPLIT_SHA256,
                    expected_stage4_sha256=fx.MODEL_SHA, roots=roots or self.b["roots"],
                    expected_population=len(self.b["ids"]))
        args.update(kw)
        return ad.Arch1Bundle(**args)

    def test_loads_exact_values(self):
        bundle = self.open()
        self.assertEqual(bundle.channels, cache.channel_names(v2cfg.STAGE4_V2A_CLASSES))
        self.assertEqual(set(bundle.train_ids), {i for i, _ in fx.TRAIN})
        i = bundle.val_ids[0]
        s = bundle.load_sample(i)
        np.testing.assert_array_equal(s["rgb"], self.b["rgbs"][i])
        np.testing.assert_array_equal(s["vessel"][..., 0], self.b["vessels"][i])
        np.testing.assert_array_equal(s["pathology"], fx.fake_maps(fx.fake_native(i)).astype(np.float32) / 255.0)
        self.assertEqual(bundle.describe()["stage4_sha256"], fx.MODEL_SHA)

    def test_model_sha_must_be_stated_and_match(self):
        with self.assertRaises(TypeError):
            ad.Arch1Bundle(expected_bundle_id=self.b["bundle"]["bundle_id"], expected_split_sha256=v2cfg.SPLIT_SHA256,
                           roots=self.b["roots"])
        with self.assertRaises(cache.ManifestMismatchError):
            self.open(expected_stage4_sha256="c" * 64)
        with self.assertRaises(cache.DenyListedModelError):
            self.open(expected_stage4_sha256=v2cfg.LEGACY_STAGE4_SHA256)

    def test_split_population_and_ids(self):
        with self.assertRaises(cache.ManifestMismatchError):
            self.open(expected_split_sha256="0" * 64)
        with self.assertRaises(cache.ManifestMismatchError):
            self.open(expected_population=v2cfg.APTOS_POPULATION)      # 5 != 3651
        with self.assertRaises(FileNotFoundError):
            self.open(expected_bundle_id="rgb-v1__s3-91f0cada__s4v2-cccccccccccc-K4")

    def test_legacy_root_refused(self):
        with self.assertRaises(cache.LegacyArtifactError):
            self.open(roots=dict(self.b["roots"], stage4_cache_v2=v2cfg.LEGACY_DIRS[0]))


class TamperTests(unittest.TestCase):
    """Each case corrupts a fresh copy of the bundle on disk."""

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.b = fx.build(self.tmp.name)
        self.bundle_path = os.path.join(self.b["roots"]["bundle_v2"], self.b["bundle"]["bundle_id"], "bundle_manifest.json")

    def tearDown(self):
        self.tmp.cleanup()

    def open(self):
        return ad.Arch1Bundle(expected_bundle_id=self.b["bundle"]["bundle_id"], expected_split_sha256=v2cfg.SPLIT_SHA256,
                              expected_stage4_sha256=fx.MODEL_SHA, roots=self.b["roots"],
                              expected_population=len(self.b["ids"]))

    def _edit_json(self, path, fn):
        with open(path, encoding="utf-8") as fh:
            m = json.load(fh)
        fn(m)
        with open(path, "w", encoding="utf-8") as fh:
            json.dump(m, fh)

    def test_stage3_parity_not_passed(self):
        p = os.path.join(self.b["roots"]["stage3_cache_v2"], v2cfg.STAGE3_GENERATION, "stage3_manifest.json")
        self._edit_json(p, lambda m: m["parity"].update(passed=False))
        with self.assertRaises(cache.StaleCacheError):
            self.open()

    def test_legacy_generation_name(self):
        self._edit_json(self.bundle_path, lambda m: m.update(stage4_generation="LocalFeatureExtraction_lesion"))
        with self.assertRaises(cache.LegacyArtifactError):
            self.open()

    def test_inconsistent_fingerprint(self):
        self._edit_json(self.bundle_path, lambda m: m.update(population_sha256="0" * 64))
        with self.assertRaises(cache.ManifestMismatchError):
            self.open()

    def test_idrid_id_in_population(self):
        def inject(m):
            m["val_ids"] = m["val_ids"][:-1] + ["IDRiD_60"]
        self._edit_json(self.bundle_path, inject)
        with self.assertRaises(cache.ManifestMismatchError):
            self.open()

    def test_tampered_file_and_missing_file(self):
        bundle = self.open()
        i = bundle.train_ids[0]
        np.save(os.path.join(bundle.dirs["stage3_cache_v2"], cache.vessel_filename(i)), np.zeros((512, 512), np.float32))
        with self.assertRaises(cache.ManifestMismatchError):
            bundle.load_sample(i)
        os.remove(os.path.join(bundle.dirs["stage4_cache_v2"], cache.pathology_filename(bundle.val_ids[0])))
        with self.assertRaises(cache.IncompleteCacheError):
            bundle.require(bundle.val_ids)


class EpochTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tmp = tempfile.TemporaryDirectory()
        cls.b = fx.build(cls.tmp.name)
        cls.bundle = ad.Arch1Bundle(expected_bundle_id=cls.b["bundle"]["bundle_id"],
                                    expected_split_sha256=v2cfg.SPLIT_SHA256, expected_stage4_sha256=fx.MODEL_SHA,
                                    roots=cls.b["roots"], expected_population=len(cls.b["ids"]))

    @classmethod
    def tearDownClass(cls):
        cls.tmp.cleanup()

    def test_augmentation_keeps_channels_aligned_and_priors_unjittered(self):
        rng = np.random.default_rng(0)
        base = rng.uniform(0.2, 0.8, (512, 512)).astype(np.float32)
        sample = {"rgb": np.stack([base] * 3, -1), "vessel": base[..., None],
                  "pathology": np.repeat(base[..., None], 8, -1)}
        for seed in range(6):
            out = ad.augment_sample(sample, np.random.default_rng(seed))
            np.testing.assert_array_equal(out["vessel"][..., 0], out["pathology"][..., 0])   # same spatial transform
            np.testing.assert_array_equal(out["pathology"][..., 0], out["pathology"][..., 7])
            self.assertAlmostEqual(float(out["vessel"].sum()), float(base.sum()), places=0)  # priors not jittered

    def test_train_sequence_follows_p_order_and_is_deterministic(self):
        import improved_training_data as itd
        seq = ad.make_epoch_sequence(self.bundle, fx.TRAIN, epoch=3, run_seed=42, batch_size=2, augment=True)
        x, y = seq[0]
        order = itd.epoch_training_order(fx.TRAIN, 42, 3)
        self.assertEqual(y.tolist(), [g for _, g in order[:2]])
        self.assertEqual(x["rgb"].shape, (2, 512, 512, 3))
        self.assertEqual(x["pathology"].shape, (2, 512, 512, 8))
        x2, _ = ad.make_epoch_sequence(self.bundle, fx.TRAIN, 3, 42, 2, True)[0]
        np.testing.assert_array_equal(x["rgb"], x2["rgb"])
        self.assertEqual(len(seq), 2)

    def test_validation_sequence_unaugmented_in_given_order(self):
        seq = ad.make_epoch_sequence(self.bundle, fx.VAL, epoch=0, run_seed=42, batch_size=8, augment=False)
        x, y = seq[0]
        self.assertEqual(y.tolist(), [g for _, g in fx.VAL])
        np.testing.assert_array_equal(x["rgb"][0], self.b["rgbs"][fx.VAL[0][0]])


if __name__ == "__main__":
    unittest.main()
