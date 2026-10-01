"""CPU tests for stage4_v2_data.py: IDRiD v2 split and isolation, Stage 2, training-input cache,
balanced sampling, patch dataset. Synthetic data unless the local datasets are present."""
import json
import os
import tempfile
import unittest

import numpy as np

import pipeline_v2_config as v2cfg
import stage4_v2_data as data
import tjdr_dataset as tj

IDRID_RAW = os.path.join(data.idrid_raw_dir(), "2. All Segmentation Groundtruths", "a. Training Set")


class IdridSplitTests(unittest.TestCase):
    def test_pinned_split_44_10_disjoint_and_hash(self):
        split = data.idrid_split()
        self.assertEqual((len(split["train"]), len(split["val"])), (44, 10))
        self.assertFalse(set(split["train"]) & set(split["val"]))
        self.assertEqual(set(split["train"]) | set(split["val"]), set(v2cfg.IDRID_SEG_TRAIN_IDS))
        self.assertFalse((set(split["train"]) | set(split["val"])) & set(v2cfg.IDRID_SEG_TEST_IDS))
        self.assertEqual(data.idrid_split_sha256(split["train"], split["val"]), v2cfg.IDRID_V2_SPLIT_SHA256)
        self.assertEqual(v2cfg.IDRID_V2_SPLIT_SHA256, "2e5bf1c36f6126e68d0356c6252665bbb6871aab0d725bff08efb11439142f30")

    def test_split_rule_is_deterministic_and_stratified(self):
        se_pos = v2cfg.IDRID_V2_VAL_SE_POSITIVE + tuple(f"IDRiD_{i:02d}" for i in (1, 2, 3, 4, 6, 7, 8, 9, 12, 13, 16, 17,
                                                                                         19, 20, 21, 22, 23, 25, 26, 27, 28))
        a = data.compute_idrid_split(se_pos)
        b = data.compute_idrid_split(se_pos)
        self.assertEqual(a, b)
        self.assertEqual(sum(v in se_pos for v in a[1]), round(10 * len(se_pos) / 54))

    @unittest.skipUnless(os.path.isdir(IDRID_RAW), "IDRiD raw masks not present")
    def test_real_split_reproduces_pinned_ids_with_5_se_positive(self):
        se_pos = data.idrid_se_positive_ids()
        self.assertEqual(len(se_pos), 26)
        train, val = data.compute_idrid_split(se_pos)
        self.assertEqual(val, v2cfg.IDRID_V2_VAL_IDS)
        self.assertEqual(tuple(v for v in val if v in se_pos), v2cfg.IDRID_V2_VAL_SE_POSITIVE)
        self.assertEqual(len(train), 44)

    def test_test_and_grading_images_refused(self):
        for image_id in ("IDRiD_55", "IDRiD_81"):
            with self.assertRaises(data.Stage4DataError):
                data.assert_idrid_training_id(image_id)
            with self.assertRaises(data.Stage4DataError):
                data.load_idrid_pair(image_id)
        for image_id in ("IDRiD_001", "IDRiD_516"):          # grading ids
            with self.assertRaises(data.Stage4DataError):
                data.assert_idrid_training_id(image_id)
        with self.assertRaises(data.Stage4DataError):
            data.read_idrid_masks("IDRiD_01", "a. Training Set", raw_dir="/x/IDRiD/grading/raw")
        data.assert_idrid_training_id("IDRiD_54")


class Stage2Tests(unittest.TestCase):
    def setUp(self):
        import cv2
        self.tmp = tempfile.TemporaryDirectory()
        rng = np.random.default_rng(0)
        img = np.zeros((120, 160, 3), np.uint8)
        yy, xx = np.mgrid[:120, :160]
        inside = (yy - 60) ** 2 + (xx - 80) ** 2 < 55 ** 2
        img[inside] = rng.integers(20, 200, (int(inside.sum()), 3))
        self.raw = os.path.join(self.tmp.name, "raw.png")
        cv2.imwrite(self.raw, img)

    def tearDown(self):
        self.tmp.cleanup()

    def test_shape_range_deterministic_and_canonical(self):
        import cv2
        from image_preprocessing import preprocess_array, preprocess_image
        a, b = data.stage2_rgb(self.raw), data.stage2_rgb(self.raw)
        self.assertEqual((a.shape, a.dtype), ((120, 160, 3), np.uint8))       # geometry preserved
        np.testing.assert_array_equal(a, b)
        ref = cv2.cvtColor(preprocess_array(cv2.imread(self.raw), profile="DR"), cv2.COLOR_BGR2RGB)
        np.testing.assert_array_equal(a, ref)
        out = os.path.join(self.tmp.name, "processed.png")
        preprocess_image(self.raw, out, profile="DR")                         # the notebook's TJDR path
        self.assertEqual(data.stage2_parity(self.raw, out)["max_abs"], 0.0)
        self.assertFalse(np.array_equal(a, cv2.cvtColor(cv2.imread(self.raw), cv2.COLOR_BGR2RGB)))

    def test_cache_sample_geometry(self):
        rgb = data.stage2_rgb(self.raw)
        masks = np.zeros((4, 120, 160), np.uint8)
        masks[1, 50:60, 70:90] = 1
        r, m = data.make_cache_sample(rgb, masks, size=96)
        self.assertEqual((r.shape, r.dtype, m.shape, m.dtype), ((96, 96, 3), np.uint8, (4, 96, 96), np.uint8))
        self.assertTrue(m[1].any() and not m[0].any())


def _synthetic_cache(tmp, per=(3, 2)):
    """Tiny cache (size 64) with IDRiD/TJDR train/val entries; manifest completeness is not required."""
    rng = np.random.default_rng(1)
    entries = [("IDRiD", "train", "a. Training Set", f"IDRiD_{i:02d}", "IDRiD-Kowa") for i in (1, 2, 3)][:per[0]]
    entries += [("IDRiD", "val", "a. Training Set", "IDRiD_05", "IDRiD-Kowa")]
    entries += [("TJDR", "train", "train", f"TJDR_train_{i:03d}", "TJDR-TRC50DX") for i in (1, 2, 3)][:per[0]]
    entries += [("TJDR", "val", "test", "TJDR_test_001", "TJDR-CLARUS500")]
    done = {}
    for e in entries:
        rgb = rng.integers(20, 230, (64, 64, 3), dtype=np.uint8)
        masks = np.zeros((4, 64, 64), np.uint8)
        masks[:, 20:30, 20:30] = 1
        name = data.sample_filename(e[0], e[1], e[3])
        _, sha, meta = data.write_cache_sample(tmp, e, rgb, masks)
        done[name] = {"sha256": sha, **meta}
    return data.write_training_manifest(tmp, done, entries), entries


class TrainingCacheTests(unittest.TestCase):
    def test_write_read_and_integrity(self):
        with tempfile.TemporaryDirectory() as tmp:
            manifest, entries = _synthetic_cache(tmp)
            self.assertFalse(manifest["complete"])
            tc = data.TrainingCache(tmp, verify_files=True, require_complete=False)
            self.assertEqual(len(tc.names("train")), 6)
            self.assertEqual(tc.names("val", "TJDR"), ["TJDR__val__TJDR_test_001.npz"])
            rgb, masks, meta = tc.load("IDRiD__train__IDRiD_01.npz")
            self.assertEqual(meta["camera"], "IDRiD-Kowa")
            with self.assertRaises(data.Stage4DataError):
                data.TrainingCache(tmp, verify_files=True, require_complete=True)
            with open(os.path.join(tmp, "IDRiD__train__IDRiD_01.npz"), "ab") as fh:
                fh.write(b"tamper")
            with self.assertRaises(data.Stage4DataError):
                data.TrainingCache(tmp, verify_files=True, require_complete=False)

    def test_test_images_and_excluded_tjdr_cannot_enter(self):
        with tempfile.TemporaryDirectory() as tmp:
            rgb, masks = np.zeros((8, 8, 3), np.uint8), np.zeros((4, 8, 8), np.uint8)
            for bad in (("IDRiD", "train", "a. Training Set", "IDRiD_60", None),
                        ("IDRiD", "test", "b. Testing Set", "IDRiD_60", None),
                        ("TJDR", "train", "train", "TJDR_train_041", None),
                        ("TJDR", "val", "test", "TJDR_test_023", None),
                        ("IDRiD", "train", "a. Training Set", "IDRiD_101", None)):
                with self.assertRaises((data.Stage4DataError, tj.TJDRDataError)):
                    data.write_cache_sample(tmp, bad, rgb, masks)
            # a manifest that lists a test image is refused on read
            manifest, _ = _synthetic_cache(tmp)
            with open(os.path.join(tmp, "manifest.json"), encoding="utf-8") as fh:
                m = json.load(fh)
            m["files"]["IDRiD__train__IDRiD_60.npz"] = {**m["files"]["IDRiD__train__IDRiD_01.npz"], "image_id": "IDRiD_60"}
            with open(os.path.join(tmp, "manifest.json"), "w", encoding="utf-8") as fh:
                json.dump(m, fh)
            with self.assertRaises(data.Stage4DataError):
                data.TrainingCache(tmp, verify_files=False, require_complete=False)

    def test_source_entries_counts(self):
        if not os.path.isdir(os.path.join(tj.raw_root(), "train", "image")):
            self.skipTest("TJDR not present")
        entries = data.source_entries()
        counts = {}
        for e in entries:
            counts[(e[0], e[1])] = counts.get((e[0], e[1]), 0) + 1
        self.assertEqual(counts, v2cfg.STAGE4_TRAIN_EXPECTED)
        self.assertFalse({e[3] for e in entries} & set(v2cfg.IDRID_SEG_TEST_IDS))


class SamplingTests(unittest.TestCase):
    def test_balanced_batches_and_reproducibility(self):
        names = {"IDRiD": [f"i{k}" for k in range(44)], "TJDR": [f"t{k}" for k in range(443)]}
        for step in range(50):
            plan = data.balanced_batch_plan(names, 8, step, seed=42)
            self.assertEqual(sum(n.startswith("i") for n, _ in plan), 4)
            self.assertEqual(sum(n.startswith("t") for n, _ in plan), 4)
        self.assertEqual(data.balanced_batch_plan(names, 8, 7, 42), data.balanced_batch_plan(names, 8, 7, 42))
        self.assertNotEqual(data.balanced_batch_plan(names, 8, 7, 42), data.balanced_batch_plan(names, 8, 8, 42))
        with self.assertRaises(data.Stage4DataError):
            data.balanced_batch_plan(names, 7, 0, 42)

    def test_patch_dataset_item(self):
        with tempfile.TemporaryDirectory() as tmp:
            _synthetic_cache(tmp)
            tc = data.TrainingCache(tmp, require_complete=False)
            ds = data.PatchDataset(tc, patch=32, p_lesion=1.0)
            item = ds[("TJDR__train__TJDR_train_001.npz", 123)]
            self.assertEqual(tuple(item["x"].shape), (3, 32, 32))
            self.assertEqual(tuple(item["y"].shape), (4, 32, 32))
            self.assertEqual(tuple(item["fov"].shape), (1, 32, 32))
            self.assertTrue(bool(item["annotated"].all()))
            self.assertGreater(float(item["y"].sum()), 0)                 # p_lesion=1 centres on a lesion
            again = ds[("TJDR__train__TJDR_train_001.npz", 123)]
            self.assertTrue(bool((item["x"] == again["x"]).all()))       # deterministic per item seed
            batch = data.collate([item, again])
            self.assertEqual(tuple(batch["x"].shape), (2, 3, 32, 32))


if __name__ == "__main__":
    unittest.main()
