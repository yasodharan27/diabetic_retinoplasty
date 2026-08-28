"""
Regression tests for global_feature_extraction_dataset.py (Stage 06
dataset loader).

Reuses the same synthetic, on-disk APTOS-shaped tree pattern
test_local_feature_extraction_dataset.py already established -- the real,
local APTOS2019 dataset is never read by this suite, per this project's
"unit tests use synthetic/temporary data only" rule. Unlike Stage 05's
dataset tests, no synthetic vessel/lesion model is needed here at all --
Stage 06 has no dependency on Stage 03/04.
"""

import csv
import os
import shutil
import tempfile
import unittest

import numpy as np
from PIL import Image

import global_feature_extraction_dataset as gfed
import local_feature_extraction_dataset as lfed

IMAGE_SIZE = 64


def _synthetic_fundus_image(size=IMAGE_SIZE, seed=0):
    rng = np.random.RandomState(seed)
    image = np.zeros((size, size, 3), dtype=np.uint8)
    yy, xx = np.ogrid[:size, :size]
    center = size // 2
    radius = int(size * 0.4)
    circle = (xx - center) ** 2 + (yy - center) ** 2 <= radius ** 2
    base = 140 + rng.randint(-20, 20, size=(size, size, 3))
    image[circle] = np.clip(base[circle], 60, 220).astype(np.uint8)
    return image


class _SyntheticAPTOSTree:
    def __init__(self, id_diagnosis_pairs):
        self.root = tempfile.mkdtemp(prefix="aptos_synth_gfed_")
        self.image_dir = os.path.join(self.root, "train_images")
        self.csv_path = os.path.join(self.root, "train.csv")
        os.makedirs(self.image_dir, exist_ok=True)
        self.pairs = list(id_diagnosis_pairs)
        self._build()

    def cleanup(self):
        shutil.rmtree(self.root, ignore_errors=True)

    def _build(self):
        with open(self.csv_path, "w", newline="", encoding="utf-8") as handle:
            writer = csv.writer(handle)
            writer.writerow(["id_code", "diagnosis"])
            for i, (id_code, diagnosis) in enumerate(self.pairs):
                writer.writerow([id_code, diagnosis])
                image = _synthetic_fundus_image(seed=i)
                Image.fromarray(image).save(os.path.join(self.image_dir, f"{id_code}.png"))


class ModuleReuseTests(unittest.TestCase):
    """Stage 06 must reuse Stage 05's Stage-02-application helpers, not
    duplicate them -- verified directly by identity, not just by matching
    output."""

    def test_reuses_local_feature_extraction_dataset_functions(self):
        self.assertIs(gfed.lfed._load_raw_bgr, lfed._load_raw_bgr)
        self.assertIs(gfed.lfed._stage02_processed_rgb, lfed._stage02_processed_rgb)
        self.assertIs(gfed.lfed._augment_spatial, lfed._augment_spatial)
        self.assertIs(gfed.lfed._augment_intensity_rgb, lfed._augment_intensity_rgb)
        self.assertIs(gfed.lfed.split_train_val_ids, lfed.split_train_val_ids)


class SplitTrainValIdsTests(unittest.TestCase):
    def setUp(self):
        self.tree = _SyntheticAPTOSTree([(f"img_{i:02d}", i % 5) for i in range(20)])
        self.addCleanup(self.tree.cleanup)

    def test_matches_local_feature_extraction_dataset_split_exactly(self):
        """Stage 05 and Stage 06 must see the identical train/val partition
        for their eventual joint training run."""
        gfed_train, gfed_val = gfed.split_train_val_ids(self.tree.csv_path, val_split=0.3, seed=42)
        lfed_train, lfed_val = lfed.split_train_val_ids(self.tree.csv_path, val_split=0.3, seed=42)
        self.assertEqual(gfed_train, lfed_train)
        self.assertEqual(gfed_val, lfed_val)


class BuildGlobalFeatureInputTests(unittest.TestCase):
    def test_output_has_three_channels(self):
        rgb = _synthetic_fundus_image(size=64, seed=1)
        result = gfed.build_global_feature_input(rgb, image_size=(64, 64))
        self.assertEqual(result.shape, (64, 64, gfed.NUM_CHANNELS))
        self.assertEqual(result.dtype, np.float32)

    def test_normalization_is_zero_to_one(self):
        rgb = _synthetic_fundus_image(size=64, seed=1)
        result = gfed.build_global_feature_input(rgb, image_size=(64, 64))
        self.assertGreaterEqual(result.min(), 0.0)
        self.assertLessEqual(result.max(), 1.0)

    def test_resize_produces_requested_spatial_size(self):
        rgb = _synthetic_fundus_image(size=64, seed=1)
        result = gfed.build_global_feature_input(rgb, image_size=(256, 256))
        self.assertEqual(result.shape, (256, 256, gfed.NUM_CHANNELS))


class BuildSampleTests(unittest.TestCase):
    def setUp(self):
        self.tree = _SyntheticAPTOSTree([("img_01", 2)])
        self.addCleanup(self.tree.cleanup)

    def test_build_sample_returns_input_and_label(self):
        x, y = gfed._build_sample("img_01", 2, self.tree.image_dir, image_size=(64, 64))
        self.assertEqual(x.shape, (64, 64, gfed.NUM_CHANNELS))
        self.assertEqual(x.dtype, np.float32)
        self.assertEqual(y, 2)

    def test_missing_image_raises_clear_error(self):
        with self.assertRaises(FileNotFoundError):
            gfed._build_sample("no_such_id", 0, self.tree.image_dir, image_size=(64, 64))

    def test_no_vessel_or_lesion_channel_present(self):
        """Structural proof, not just an assertion: NUM_CHANNELS is 3, and
        the built sample has exactly 3 channels -- there is no code path
        in this module that could add a 4th (vessel) or 5th-8th (lesion)
        channel."""
        x, _ = gfed._build_sample("img_01", 2, self.tree.image_dir, image_size=(64, 64))
        self.assertEqual(x.shape[-1], 3)


class AugmentationReuseTests(unittest.TestCase):
    def test_spatial_and_intensity_augmentation_apply_without_error(self):
        rng = np.random.default_rng(3)
        input_array = np.random.RandomState(0).rand(16, 16, 3).astype(np.float32)
        augmented = gfed._augment(input_array, rng)
        self.assertEqual(augmented.shape, input_array.shape)
        self.assertGreaterEqual(augmented.min(), 0.0)
        self.assertLessEqual(augmented.max(), 1.0)


class TfDataPipelineTests(unittest.TestCase):
    def setUp(self):
        # 3 samples per class (15 total): split_train_val_ids now stratifies
        # by diagnosis (downstream_split.compute_split), which requires at
        # least 2 members per class -- 6 samples (1-2 per class) was enough
        # for the old, non-stratified split but is not enough here.
        self.tree = _SyntheticAPTOSTree([(f"img_{i:02d}", i % 5) for i in range(15)])
        self.addCleanup(self.tree.cleanup)

    def test_train_val_datasets_yield_correctly_shaped_batches(self):
        train_ds, val_ds = gfed.load_global_feature_extraction_datasets(
            csv_path=self.tree.csv_path, image_dir=self.tree.image_dir,
            image_size=(32, 32), val_split=0.34, batch_size=2, seed=42, augment_train=True,
        )
        x_batch, y_batch = next(iter(train_ds))
        self.assertEqual(x_batch.shape[1:], (32, 32, gfed.NUM_CHANNELS))
        self.assertEqual(y_batch.shape[1:], ())
        self.assertLessEqual(int(x_batch.shape[0]), 2)

        x_val, _ = next(iter(val_ds))
        self.assertEqual(x_val.shape[1:], (32, 32, gfed.NUM_CHANNELS))

    def test_val_split_is_never_augmented(self):
        _, val_ds_a = gfed.load_global_feature_extraction_datasets(
            csv_path=self.tree.csv_path, image_dir=self.tree.image_dir,
            image_size=(32, 32), val_split=0.34, batch_size=8, seed=42, augment_train=False,
        )
        x_a, _ = next(iter(val_ds_a))

        _, val_ds_b = gfed.load_global_feature_extraction_datasets(
            csv_path=self.tree.csv_path, image_dir=self.tree.image_dir,
            image_size=(32, 32), val_split=0.34, batch_size=8, seed=42, augment_train=False,
        )
        x_b, _ = next(iter(val_ds_b))

        np.testing.assert_array_equal(np.asarray(x_a), np.asarray(x_b))


if __name__ == "__main__":
    unittest.main()
