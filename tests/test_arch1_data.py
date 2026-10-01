"""CPU unit tests for arch1_data.py: a tiny, fully synthetic v2 bundle (2 images) in a temp dir."""
import os
import tempfile
import unittest

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "-1")

import numpy as np

import arch1_data as ad
import pipeline_v2_config as v2cfg
import stage34_cache_v2 as cache

S4 = "a" * 64
S3 = v2cfg.STAGE3_LWNET_SHA256
SPLIT = v2cfg.SPLIT_SHA256
CLASSES = v2cfg.STAGE4_V2A_CLASSES
CHANNELS = cache.channel_names(CLASSES)
GEN4 = cache.stage4_generation_id(S4, 4)
IDS = ("0a1b", "2c3d")


def _write_bundle(tmp):
    roots = {k: os.path.join(tmp, name) for k, name in
             (("stage2_rgb_v2", "Stage2"), ("stage3_cache_v2", "Stage3"),
              ("stage4_cache_v2", "Stage4"), ("bundle_v2", "Bundle"))}
    rng = np.random.default_rng(0)
    data, files = {}, {k: {} for k in cache.DATA_SUBDIRS}
    gens = {"stage2_rgb_v2": v2cfg.STAGE2_RGB_GENERATION, "stage3_cache_v2": v2cfg.STAGE3_GENERATION,
            "stage4_cache_v2": GEN4}
    dirs = {k: os.path.join(roots[k], gens[k], cache.DATA_SUBDIRS[k]) for k in gens}
    for d in dirs.values():
        os.makedirs(d)
    for i in IDS:
        rgb = rng.uniform(0, 1, (512, 512, 3)).astype(np.float32)
        vessel = rng.uniform(0, 1, (512, 512)).astype(np.float32)
        maps = rng.integers(0, 256, (512, 512, 8), dtype=np.uint8)
        p = os.path.join(dirs["stage2_rgb_v2"], cache.rgb_filename(i))
        np.save(p, rgb)
        files["stage2_rgb_v2"][i] = cache.sha256_file(p)
        p = os.path.join(dirs["stage3_cache_v2"], cache.vessel_filename(i))
        np.save(p, vessel)
        files["stage3_cache_v2"][i] = cache.sha256_file(p)
        p = os.path.join(dirs["stage4_cache_v2"], cache.pathology_filename(i))
        files["stage4_cache_v2"][i] = cache.write_pathology_npz(
            p, maps, channels=CHANNELS, image_id=i, stage4_sha256=S4, stage3_sha256=S3, gen_id=GEN4)
        data[i] = (rgb, vessel, maps)
    s2 = cache.stage2_cache_manifest(split_sha256=SPLIT, files=files["stage2_rgb_v2"])
    s3 = cache.stage3_cache_manifest(lwnet_sha256=S3, tta=True, parity={"passed": True},
                                     split_sha256=SPLIT, files=files["stage3_cache_v2"])
    s4 = cache.stage4_cache_manifest(stage4_sha256=S4, stage3_sha256=S3, classes=CLASSES,
                                     split_sha256=SPLIT, files=files["stage4_cache_v2"])
    for kind, manifest in (("stage2_rgb_v2", s2), ("stage3_cache_v2", s3), ("stage4_cache_v2", s4)):
        cache.write_manifest(os.path.join(roots[kind], gens[kind], cache.MANIFEST_NAMES[kind]), manifest)
    bundle = cache.bundle_manifest(s2, s3, s4)
    cache.write_manifest(os.path.join(roots["bundle_v2"], bundle["bundle_id"],
                                      cache.MANIFEST_NAMES["bundle_v2"]), bundle)
    return roots, bundle["bundle_id"], data, dirs


class BundleTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.roots, self.bundle_id, self.data, self.dirs = _write_bundle(self.tmp.name)

    def tearDown(self):
        self.tmp.cleanup()

    def _bundle(self, **kw):
        args = dict(expected_bundle_id=self.bundle_id, expected_split_sha256=SPLIT, roots=self.roots)
        args.update(kw)
        return ad.Arch1Bundle(**args)

    def test_loads_exact_values_with_manifest_channels(self):
        bundle = self._bundle()
        self.assertEqual(bundle.channels, CHANNELS)
        self.assertEqual(bundle.population, IDS)
        bundle.require(IDS)
        rgb, vessel, maps = self.data[IDS[1]]
        sample = bundle.load_sample(IDS[1])
        np.testing.assert_array_equal(sample["rgb"], rgb)
        np.testing.assert_array_equal(sample["vessel"][..., 0], vessel)
        np.testing.assert_array_equal(sample["pathology"], maps.astype(np.float32) / 255.0)

    def test_wrong_bundle_or_split_rejected(self):
        with self.assertRaises(FileNotFoundError):
            self._bundle(expected_bundle_id="rgb-v1__s3-91f0cada__s4v2-bbbbbbbbbbbb-K4")
        with self.assertRaises(cache.ManifestMismatchError):
            self._bundle(expected_split_sha256="0" * 64)

    def test_tampered_file_and_missing_id_raise(self):
        bundle = self._bundle()
        path = os.path.join(self.dirs["stage3_cache_v2"], cache.vessel_filename(IDS[0]))
        np.save(path, np.zeros((512, 512), np.float32))
        with self.assertRaises(cache.ManifestMismatchError):
            bundle.load_sample(IDS[0])
        with self.assertRaises(cache.IncompleteCacheError):
            bundle.load_sample("ffff")
        with self.assertRaises(cache.IncompleteCacheError):
            bundle.require(IDS + ("ffff",))
        os.remove(os.path.join(self.dirs["stage4_cache_v2"], cache.pathology_filename(IDS[1])))
        with self.assertRaises(cache.IncompleteCacheError):
            bundle.require(IDS)

    def test_legacy_root_refused(self):
        roots = dict(self.roots, stage4_cache_v2=v2cfg.LEGACY_DIRS[0])
        with self.assertRaises(cache.LegacyArtifactError):
            ad.Arch1Bundle(expected_bundle_id=self.bundle_id, expected_split_sha256=SPLIT, roots=roots)

    def test_sequence_batches(self):
        seq = ad.make_sequence(self._bundle(), IDS, [0, 4], batch_size=2, shuffle=True, seed=42)
        self.assertEqual(len(seq), 1)
        inputs, grades = seq[0]
        self.assertEqual(inputs["rgb"].shape, (2, 512, 512, 3))
        self.assertEqual(inputs["vessel"].shape, (2, 512, 512, 1))
        self.assertEqual(inputs["pathology"].shape, (2, 512, 512, 8))
        self.assertEqual(sorted(grades.tolist()), [0, 4])


if __name__ == "__main__":
    unittest.main()
