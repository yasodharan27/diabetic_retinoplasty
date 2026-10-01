"""CPU unit tests for stage34_cache_v2.py (v2 cache lineage, guards, pooling, pyramid features)."""
import os
import tempfile
import unittest

import numpy as np

import pipeline_v2_config as v2cfg
import stage34_cache_v2 as cache

S4 = "a" * 64                      # a stand-in trained Stage-4 v2 sha
S3 = v2cfg.STAGE3_LWNET_SHA256
SPLIT = v2cfg.SPLIT_SHA256
CLASSES = v2cfg.STAGE4_V2A_CLASSES
CHANNELS = cache.channel_names(CLASSES)
GEN = cache.stage4_generation_id(S4, len(CLASSES))


def _maps(seed=0, k2=len(CHANNELS)):
    return np.random.default_rng(seed).integers(0, 256, (512, 512, k2), dtype=np.uint8)


class ConfigTests(unittest.TestCase):
    def test_identities_and_geometry(self):
        self.assertEqual(v2cfg.STAGE4_V2A_CLASSES, ("MA", "HE", "EX", "SE"))
        self.assertNotIn("OD", v2cfg.STAGE4_V2A_CLASSES)
        self.assertEqual(v2cfg.POOL_FACTOR, 3)
        self.assertEqual(v2cfg.STAGE4_ENCODER, "se_resnet101")
        self.assertEqual(v2cfg.STAGE3_GENERATION, "s3-91f0cada")
        for sha in (v2cfg.LEGACY_STAGE4_SHA256, v2cfg.STAGE3_LWNET_SHA256, v2cfg.SPLIT_SHA256,
                    v2cfg.STAGE4_ENCODER_SHA256, v2cfg.CONVNEXT_TINY_WEIGHTS_SHA256):
            self.assertRegex(sha, r"^[0-9a-f]{64}$")

    def test_v2_roots_are_outside_every_legacy_location(self):
        for root in (v2cfg.STAGE2_ROOT, v2cfg.STAGE3_ROOT, v2cfg.STAGE4_ROOT, v2cfg.BUNDLE_ROOT,
                     v2cfg.STAGE4_V2_MODEL_ROOT):
            cache.assert_not_legacy_path(os.path.join(root, "x"))


class ChannelTests(unittest.TestCase):
    def test_channel_order_is_interleaved_mean_max(self):
        self.assertEqual(CHANNELS, ("MA:mean", "MA:max", "HE:mean", "HE:max", "EX:mean", "EX:max",
                                    "SE:mean", "SE:max"))
        self.assertEqual(cache.classes_from_channels(CHANNELS), CLASSES)

    def test_k_configurable(self):
        six = ("MA", "HE", "EX", "SE", "NV", "IRMA")
        self.assertEqual(len(cache.channel_names(six)), 12)
        self.assertEqual(cache.stage4_generation_id(S4, 6), "s4v2-aaaaaaaaaaaa-K6")

    def test_bad_channel_lists_rejected(self):
        with self.assertRaises(ValueError):
            cache.channel_names(("MA", "MA"))
        with self.assertRaises(ValueError):
            cache.channel_names(("lesion",))
        with self.assertRaises(cache.ManifestMismatchError):
            cache.classes_from_channels(("MA:max", "MA:mean"))


class PoolingTests(unittest.TestCase):
    def test_block_pooling_is_exact(self):
        rng = np.random.default_rng(1)
        p = rng.uniform(0, 1, (12, 9, 2)).astype(np.float32)
        mean, maximum = cache.block_pool_mean_max(p, 3)
        self.assertEqual(mean.shape, (4, 3, 2))
        for i in range(4):
            for j in range(3):
                block = p[3 * i:3 * i + 3, 3 * j:3 * j + 3]
                np.testing.assert_allclose(mean[i, j], block.astype(np.float64).mean(axis=(0, 1)),
                                           rtol=0, atol=1e-7)
                np.testing.assert_array_equal(maximum[i, j], block.max(axis=(0, 1)))

    def test_single_pixel_lesion_survives_in_max(self):
        p = np.zeros((1536, 1536, 1), np.float32)
        p[700, 1001, 0] = 1.0
        packed = cache.pack_pathology_maps(p)
        self.assertEqual(packed.shape, (512, 512, 2))
        self.assertEqual(packed[233, 333, 1], 255)                   # max keeps the MA
        self.assertEqual(packed[233, 333, 0], round(255 / 9))        # mean dilutes it
        self.assertEqual(int(packed.sum()), 255 + round(255 / 9))

    def test_pack_order_and_uint8(self):
        p = np.zeros((6, 6, 2), np.float32)
        p[..., 0] = 0.2
        p[0, 0, 1] = 0.72
        packed = cache.pack_pathology_maps(p, 3)
        self.assertEqual(packed.dtype, np.uint8)
        np.testing.assert_array_equal(packed[..., 0], 51)      # class 0 mean
        np.testing.assert_array_equal(packed[..., 1], 51)      # class 0 max
        self.assertEqual(packed[0, 0, 2], 20)                  # class 1 mean: 0.08 * 255 = 20.4
        self.assertEqual(packed[0, 0, 3], 184)                 # class 1 max: 0.72 * 255 = 183.6
        with self.assertRaises(ValueError):
            cache.block_pool_mean_max(np.zeros((10, 9, 1)), 3)
        with self.assertRaises(ValueError):
            cache.to_uint8(np.array([np.nan]))


class GuardTests(unittest.TestCase):
    def test_legacy_directories_and_names_refused(self):
        for root in v2cfg.LEGACY_DIRS:
            with self.assertRaises(cache.LegacyArtifactError):
                cache.assert_not_legacy_path(os.path.join(root, "APTOS_x_pathology-s4v2_512x512.npz"))
        for name in ("APTOS_abc_lesion_512x512.npy", "/tmp/cache_archive/shard.tar"):
            with self.assertRaises(cache.LegacyArtifactError):
                cache.assert_not_legacy_path(os.path.join(tempfile.gettempdir(), name))

    def test_deny_list(self):
        with self.assertRaises(cache.DenyListedModelError):
            cache.assert_not_deny_listed(v2cfg.LEGACY_STAGE4_SHA256)
        with self.assertRaises(cache.DenyListedModelError):
            cache.stage4_generation_id(v2cfg.LEGACY_STAGE4_SHA256, 4)

    def test_only_loose_vessel_files_may_come_from_legacy(self):
        legacy = v2cfg.LEGACY_DIRS[0]
        cache.assert_legacy_vessel_source(os.path.join(legacy, "APTOS_1a2b3c_vessel_512x512.npy"))
        for bad in ("APTOS_1a2b3c_lesion_512x512.npy", "APTOS_1a2b3c_rgb_512x512.npy"):
            with self.assertRaises(cache.LegacyArtifactError):
                cache.assert_legacy_vessel_source(os.path.join(legacy, bad))
        with self.assertRaises(cache.LegacyArtifactError):
            cache.assert_legacy_vessel_source("/x/cache_archive/APTOS_1a2b3c_vessel_512x512.npy")


class PathologyNpzTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.path = os.path.join(self.tmp.name, cache.pathology_filename("1a2b3c"))
        self.maps = _maps()
        self.file_sha = cache.write_pathology_npz(self.path, self.maps, channels=CHANNELS,
                                                  image_id="1a2b3c", stage4_sha256=S4,
                                                  stage3_sha256=S3, gen_id=GEN)

    def tearDown(self):
        self.tmp.cleanup()

    def _read(self, **overrides):
        kwargs = dict(expected_stage4_sha256=S4, expected_stage3_sha256=S3, expected_gen_id=GEN,
                      expected_channels=CHANNELS, expected_file_sha256=self.file_sha)
        kwargs.update(overrides)
        return cache.read_pathology_npz(self.path, **kwargs)

    def test_round_trip(self):
        np.testing.assert_array_equal(self._read(), self.maps)
        with np.load(self.path) as data:
            self.assertEqual(str(data["stage3_sha256"]), S3)
            self.assertEqual(str(data["gen_id"]), GEN)
            self.assertEqual(tuple(data["channels"]), CHANNELS)

    def test_every_mismatch_raises(self):
        for kw in ({"expected_stage4_sha256": "b" * 64}, {"expected_stage3_sha256": "c" * 64},
                   {"expected_gen_id": "s4v2-bbbbbbbbbbbb-K4"},
                   {"expected_channels": CHANNELS[2:] + CHANNELS[:2]},
                   {"expected_file_sha256": "d" * 64}):
            with self.assertRaises(cache.CacheLineageError, msg=str(kw)):
                self._read(**kw)
        with self.assertRaises(cache.DenyListedModelError):
            self._read(expected_stage4_sha256=v2cfg.LEGACY_STAGE4_SHA256)

    def test_reader_has_no_defaults(self):
        with self.assertRaises(TypeError):
            cache.read_pathology_npz(self.path)

    def test_legacy_npy_rejected(self):
        legacy = os.path.join(self.tmp.name, "APTOS_1a2b3c_lesion_512x512.npy")
        np.save(legacy, np.zeros((512, 512, 4), np.float32))
        with self.assertRaises(cache.LegacyArtifactError):
            cache.read_pathology_npz(legacy, expected_stage4_sha256=S4, expected_stage3_sha256=S3,
                                     expected_gen_id=GEN, expected_channels=CHANNELS)

    def test_writer_refuses_bad_inputs_and_legacy_locations(self):
        with self.assertRaises(ValueError):
            cache.write_pathology_npz(self.path, self.maps.astype(np.float32), channels=CHANNELS,
                                      image_id="1a2b3c", stage4_sha256=S4, stage3_sha256=S3, gen_id=GEN)
        with self.assertRaises(cache.ManifestMismatchError):
            cache.write_pathology_npz(self.path, self.maps, channels=CHANNELS, image_id="1a2b3c",
                                      stage4_sha256=S4, stage3_sha256=S3, gen_id="s4v2-x-K4")
        with self.assertRaises(cache.DenyListedModelError):
            cache.write_pathology_npz(self.path, self.maps, channels=CHANNELS, image_id="1a2b3c",
                                      stage4_sha256=v2cfg.LEGACY_STAGE4_SHA256, stage3_sha256=S3,
                                      gen_id=GEN)
        legacy_target = os.path.join(v2cfg.LEGACY_DIRS[0], cache.pathology_filename("1a2b3c"))
        with self.assertRaises(cache.LegacyArtifactError):
            cache.write_pathology_npz(legacy_target, self.maps, channels=CHANNELS, image_id="1a2b3c",
                                      stage4_sha256=S4, stage3_sha256=S3, gen_id=GEN)
        self.assertFalse(os.path.exists(legacy_target))


class ManifestTests(unittest.TestCase):
    def _manifests(self, ids=("a1", "b2")):
        files = {i: "e" * 64 for i in ids}
        s2 = cache.stage2_cache_manifest(split_sha256=SPLIT, files=files)
        s3 = cache.stage3_cache_manifest(lwnet_sha256=S3, tta=True, parity={"passed": True, "ids": 25},
                                         split_sha256=SPLIT, files=files)
        s4 = cache.stage4_cache_manifest(stage4_sha256=S4, stage3_sha256=S3, classes=CLASSES,
                                         split_sha256=SPLIT, files=files)
        return s2, s3, s4

    def test_bundle_binds_generations(self):
        s2, s3, s4 = self._manifests()
        bundle = cache.bundle_manifest(s2, s3, s4)
        self.assertEqual(bundle["bundle_id"], f"rgb-v1__s3-91f0cada__{GEN}")
        self.assertEqual(bundle["stage3_sha256"], S3)
        self.assertEqual(bundle["channels"], list(CHANNELS))

    def test_stage3_requires_pinned_lwnet_and_passed_parity(self):
        with self.assertRaises(cache.ManifestMismatchError):
            cache.stage3_cache_manifest(lwnet_sha256="f" * 64, tta=True, parity={"passed": True},
                                        split_sha256=SPLIT, files={"a": "e" * 64})
        with self.assertRaises(cache.StaleCacheError):
            cache.stage3_cache_manifest(lwnet_sha256=S3, tta=True, parity={"passed": False},
                                        split_sha256=SPLIT, files={"a": "e" * 64})

    def test_split_disagreement_rejected(self):
        s2, s3, s4 = self._manifests()
        s3["split_sha256"] = "0" * 64
        with self.assertRaises(cache.ManifestMismatchError):
            cache.bundle_manifest(s2, s3, s4)

    def test_write_load_and_completeness(self):
        _, _, s4 = self._manifests()
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "stage4_manifest.json")
            cache.write_manifest(path, s4)
            loaded = cache.load_manifest(path, expected_kind="stage4_cache_v2",
                                         expected_generation_id=GEN)
            self.assertEqual(loaded["channels"], list(CHANNELS))
            with self.assertRaises(cache.ManifestMismatchError):
                cache.load_manifest(path, expected_kind="stage3_cache_v2")
            with self.assertRaises(cache.ManifestMismatchError):
                cache.load_manifest(path, expected_kind="stage4_cache_v2",
                                    expected_generation_id="s4v2-bbbbbbbbbbbb-K4")
            cache.assert_complete(loaded, ["a1", "b2"])
            with self.assertRaises(cache.IncompleteCacheError):
                cache.assert_complete(loaded, ["a1", "zz"])
            with self.assertRaises(cache.IncompleteCacheError):       # listed but not on disk
                cache.assert_complete(loaded, ["a1"], tmp, cache.pathology_filename)


class CanaryParityPyramidTests(unittest.TestCase):
    def test_canary_and_parity(self):
        a = _maps(3)
        b = a.copy()
        b[0, 0, 0] = np.uint8(min(255, int(a[0, 0, 0]) + 2) if a[0, 0, 0] < 254 else a[0, 0, 0] - 2)
        self.assertLessEqual(cache.freshness_canary(a, b), 2)
        b[1, 1, 1] = np.uint8((int(a[1, 1, 1]) + 3) % 256)
        with self.assertRaises(cache.StaleCacheError):
            cache.freshness_canary(a, b)
        v = np.random.default_rng(0).uniform(0, 1, (512, 512)).astype(np.float32)
        self.assertEqual(cache.stage3_parity(v, v[..., None]), 0.0)
        with self.assertRaises(cache.StaleCacheError):
            cache.stage3_parity(v, v + 2e-4)

    def test_pyramid_features(self):
        maps = _maps(4)
        feats = cache.pyramid_features(maps)
        names = cache.pyramid_feature_names(CHANNELS)
        self.assertEqual(feats.shape, (len(CHANNELS) * 2 * 21,))
        self.assertEqual(len(names), feats.size)
        x = maps.astype(np.float32) / 255.0
        np.testing.assert_allclose(feats[names.index("HE:max|g1|r0c0|mean")], x[..., 3].mean(), atol=1e-6)
        np.testing.assert_allclose(feats[names.index("EX:mean|g2|r1c0|max")],
                                   x[256:, :256, 4].max(), atol=0)
        np.testing.assert_allclose(feats[names.index("SE:max|g4|r3c2|mean")],
                                   x[384:, 256:384, 7].mean(), atol=1e-6)
        vessel = np.random.default_rng(5).uniform(0, 1, (512, 512, 1)).astype(np.float32)
        self.assertEqual(cache.pyramid_features(vessel).shape, (42,))


if __name__ == "__main__":
    unittest.main()
