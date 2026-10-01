"""CPU tests for tjdr_dataset.py (research record §42 R1-R3, §43). Synthetic palette PNGs cover the
contract; the real-data checks run only when datasets/TJDR/raw is present."""
import os
import tempfile
import unittest

import numpy as np
from PIL import Image

import pipeline_v2_config as v2cfg
import stage4_v2 as s4
import tjdr_dataset as tj

CLASSES = v2cfg.STAGE4_V2A_CLASSES
VOC = [0, 0, 0, 128, 0, 0, 0, 128, 0, 128, 128, 0, 0, 0, 128] + [0] * (768 - 15)


def _palette_png(path, index):
    im = Image.fromarray(index.astype(np.uint8), mode="P")
    im.putpalette(VOC)
    im.save(path)


def _fake_root(tmp, n_train=448, n_test=113):
    for split, n in (("train", n_train), ("test", n_test)):
        for sub in ("image", "annotation"):
            os.makedirs(os.path.join(tmp, split, sub))
        for i in range(1, n + 1):
            name = f"TJDR_{split}_{i:03d}.png"
            for sub in ("image", "annotation"):
                open(os.path.join(tmp, split, sub, name), "wb").close()
    return tmp


class ExclusionTests(unittest.TestCase):
    def test_pinned_list_and_counts(self):
        self.assertEqual(v2cfg.TJDR_EXCLUDED["train"],
                         ("TJDR_train_041", "TJDR_train_042", "TJDR_train_091", "TJDR_train_174", "TJDR_train_105"))
        self.assertEqual(v2cfg.TJDR_EXCLUDED["test"], ("TJDR_test_021", "TJDR_test_023", "TJDR_test_003"))
        self.assertEqual(set(v2cfg.TJDR_EXCLUSION_REASONS),
                         set(v2cfg.TJDR_EXCLUDED["train"]) | set(v2cfg.TJDR_EXCLUDED["test"]))
        with tempfile.TemporaryDirectory() as tmp:
            root = _fake_root(tmp)
            train, test = tj.usable_ids("train", root), tj.usable_ids("test", root)
            self.assertEqual((len(train), len(test)), (443, 110))
            for split, ids in (("train", train), ("test", test)):
                self.assertFalse(set(ids) & set(v2cfg.TJDR_EXCLUDED[split]))

    def test_count_drift_and_missing_exclusion_raise(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = _fake_root(tmp, n_train=449)
            with self.assertRaises(tj.TJDRDataError):
                tj.usable_ids("train", root)
        with tempfile.TemporaryDirectory() as tmp:
            root = _fake_root(tmp, n_train=40)              # excluded ids 041+ absent
            with self.assertRaises(tj.TJDRDataError):
                tj.usable_ids("train", root, check_counts=False)

    def test_excluded_ids_cannot_be_loaded(self):
        for split, ids in v2cfg.TJDR_EXCLUDED.items():
            for image_id in ids:
                with self.assertRaises(tj.TJDRDataError):
                    tj.load_pair(split, image_id, root="/nonexistent")
        with self.assertRaises(tj.TJDRDataError):
            tj.assert_usable("train", "TJDR_test_050")        # wrong split


class MaskTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.index = np.zeros((30, 30), np.uint8)
        self.index[2:5, 2:5] = 1            # EX
        self.index[10:14, 10:14] = 2        # HE
        self.index[20, 20] = 3              # MA
        self.index[25:28, 3:9] = 4          # SE
        self.path = os.path.join(self.tmp.name, "m.png")
        _palette_png(self.path, self.index)

    def tearDown(self):
        self.tmp.cleanup()

    def test_reads_palette_indices_without_convert(self):
        mask = tj.read_index_mask(self.path)
        np.testing.assert_array_equal(mask, self.index)
        # what .convert("L") would have produced: luminance, not class codes
        lum = np.asarray(Image.open(self.path).convert("L"))
        self.assertFalse(np.array_equal(lum, self.index))

    def test_mode_and_values_validated(self):
        rgb_path = os.path.join(self.tmp.name, "rgb.png")
        Image.fromarray(np.zeros((8, 8, 3), np.uint8)).save(rgb_path)
        with self.assertRaises(tj.TJDRDataError):
            tj.read_index_mask(rgb_path)
        l_path = os.path.join(self.tmp.name, "l.png")
        Image.fromarray(self.index).save(l_path)          # mode "L"
        with self.assertRaises(tj.TJDRDataError):
            tj.read_index_mask(l_path)
        bad = self.index.copy()
        bad[0, 0] = 5
        bad_path = os.path.join(self.tmp.name, "bad.png")
        _palette_png(bad_path, bad)
        with self.assertRaises(tj.TJDRDataError):
            tj.read_index_mask(bad_path)

    def test_exact_code_mapping(self):
        self.assertEqual(v2cfg.TJDR_MASK_CODES, {"MA": 3, "HE": 2, "EX": 1, "SE": 4})
        self.assertEqual(s4.TJDR.metadata["mask_codes"], {"MA": 3, "HE": 2, "EX": 1, "SE": 4})

    def test_four_channel_binary_output_in_class_order(self):
        masks = tj.split_index_mask(self.index)
        self.assertEqual((masks.shape, masks.dtype), ((4, 30, 30), np.uint8))
        self.assertEqual(set(np.unique(masks).tolist()), {0, 1})
        for k, c in enumerate(CLASSES):                      # MA, HE, EX, SE
            np.testing.assert_array_equal(masks[k], (self.index == v2cfg.TJDR_MASK_CODES[c]).astype(np.uint8))
        self.assertEqual(int(masks[0].sum()), 1)             # MA: one pixel
        self.assertEqual(int(masks[1].sum()), 16)            # HE
        self.assertEqual(int(masks[2].sum()), 9)             # EX
        self.assertEqual(int(masks[3].sum()), 18)            # SE
        self.assertEqual(int(masks.sum(axis=0).max()), 1)    # mutually exclusive
        with self.assertRaises(tj.TJDRDataError):
            tj.split_index_mask(self.index, ("MA", "NV"))

    def test_raw_index_mask_rejected_by_resize(self):
        with self.assertRaises(ValueError):
            s4.resize_mask_full_frame(self.index, 96)
        with self.assertRaises(ValueError):
            s4.resize_mask_full_frame(tj.split_index_mask(self.index), 96)     # (K,H,W) is not one mask
        for m in (self.index == 2, (self.index == 2).astype(np.uint8), (self.index == 2).astype(np.uint8) * 255):
            self.assertEqual(s4.resize_mask_full_frame(m, 96).shape, (96, 96))

    def test_resized_class_masks_use_the_unchanged_rule(self):
        out = tj.resized_class_masks(self.index, size=60)
        self.assertEqual(out.shape, (4, 60, 60))
        for k, m in enumerate(tj.split_index_mask(self.index)):
            np.testing.assert_array_equal(out[k], s4.resize_mask_full_frame(m, 60, threshold=0.5))


class DatasetSpecTests(unittest.TestCase):
    def test_tjdr_verified_with_metadata(self):
        spec = s4.DATASET_SPECS["TJDR"]
        self.assertTrue(spec.verified)
        self.assertEqual(spec.annotated_classes, ("MA", "HE", "EX", "SE"))
        md = spec.metadata
        self.assertEqual(md["usable_counts"], {"train": 443, "test": 110})
        self.assertEqual(md["official_counts"], {"train": 448, "test": 113})
        self.assertEqual(md["mask_mode"], "P")
        self.assertEqual(tuple(md["mask_values"]), (0, 1, 2, 3, 4))
        self.assertEqual(md["excluded"]["test"], ("TJDR_test_021", "TJDR_test_023", "TJDR_test_003"))
        self.assertEqual(md["class_image_counts_official"]["train"], {"MA": 137, "HE": 249, "EX": 255, "SE": 151})
        usable = md["class_image_counts_usable"]
        self.assertIsNotNone(usable)
        for split in ("train", "test"):
            for c in CLASSES:
                self.assertLessEqual(usable[split][c], md["class_image_counts_official"][split][c])
        with self.assertRaises(TypeError):
            md["verified_by"] = "x"                          # read-only mapping
        self.assertTrue(s4.require_verified([s4.IDRID_SEG, s4.TJDR]))


@unittest.skipUnless(os.path.isdir(os.path.join(tj.raw_root(), "train", "image")), "TJDR not present locally")
class RealDataTests(unittest.TestCase):
    def test_real_usable_counts(self):
        self.assertEqual(len(tj.usable_ids("train")), 443)
        self.assertEqual(len(tj.usable_ids("test")), 110)

    def test_real_pair_loads_as_four_binary_masks(self):
        rgb, masks = tj.load_pair("test", "TJDR_test_002")
        self.assertEqual(masks.shape, (4,) + rgb.shape[:2])
        self.assertEqual(set(np.unique(masks).tolist()) <= {0, 1}, True)
        self.assertGreater(int(masks[2].sum()), 0)           # EX present (§42: values {0,1,2})
        self.assertGreater(int(masks[1].sum()), 0)           # HE present


if __name__ == "__main__":
    unittest.main()
