"""
Regression tests for lesion_segmentation_dataset.py (Stage 04 dataset
loader).

Builds a small, synthetic, on-disk directory tree shaped exactly like
IDRiD's real segmentation subset (same "1. Original Images"/"2. All
Segmentation Groundtruths" folder names, same "IDRiD_<id>[_<SUFFIX>].<ext>"
filename convention, verified against the real dataset before writing this
module) inside a temporary directory -- the real, local IDRiD dataset is
never read by this suite, per this project's "unit tests use synthetic/
temporary data only" rule (PROJECT_CODE.md) and per the explicit instruction
that this stage's dataset loader and its local tests must not require the
full dataset.

The Stage 03 vessel model used here is the same synthetic, untrained,
from-scratch checkpoint pattern already established in
tests/test_vessel_segmentation_device.py (build_vessel_segmentation_model()
+ a temporary torch.save()'d checkpoint) -- the real, gitignored LWNet
checkpoint is never touched.
"""

import os
import shutil
import tempfile
import unittest
from unittest import mock

import numpy as np
import torch
from PIL import Image

import lesion_segmentation_dataset as lsd
from vessel_segmentation_model import build_vessel_segmentation_model, load_state_dict_from_checkpoint

IMAGE_SIZE = 256  # large enough for compute_fov_mask's circle-fit to succeed reliably


def _synthetic_fundus_image(size=IMAGE_SIZE, seed=0):
    """A bright filled circle on a black background -- enough like a real
    fundus photo's basic light/dark structure for Stage 03's FOV circle-fit
    to succeed on. Identical recipe to
    tests/test_vessel_segmentation_device.py's helper of the same name."""
    rng = np.random.RandomState(seed)
    image = np.zeros((size, size, 3), dtype=np.uint8)
    yy, xx = np.ogrid[:size, :size]
    center = size // 2
    radius = int(size * 0.4)
    circle = (xx - center) ** 2 + (yy - center) ** 2 <= radius ** 2
    base = 140 + rng.randint(-20, 20, size=(size, size, 3))
    image[circle] = np.clip(base[circle], 60, 220).astype(np.uint8)
    return image


def _synthetic_lesion_mask(size, seed):
    """A small square blob of lesion pixels at a deterministic,
    seed-dependent location -- enough to verify real (nonzero) lesion
    content survives loading/resizing without needing real annotations."""
    rng = np.random.RandomState(seed + 1000)
    mask = np.zeros((size, size), dtype=np.uint8)
    r0 = int(rng.randint(0, size - 12))
    c0 = int(rng.randint(0, size - 12))
    mask[r0:r0 + 8, c0:c0 + 8] = 1
    return mask


class _SyntheticIDRiDTree:
    """Builds a temporary directory shaped like
    datasets/IDRiD/segmentation/{raw,processed}/, with a small, fully
    synthetic set of training/testing images and masks. `missing_se_ids`
    controls which training/testing ids get no Soft Exudates mask file at
    all -- IDRiD's own real convention for "no soft exudates present",
    reproduced here deliberately to test that exact behavior."""

    def __init__(self, train_ids, test_ids, missing_se_ids=frozenset()):
        self.root = tempfile.mkdtemp(prefix="idrid_seg_synth_")
        self.raw_dir = os.path.join(self.root, "raw")
        self.processed_dir = os.path.join(self.root, "processed")
        self.vessel_cache_dir = os.path.join(self.root, "vessel_cache")
        self.train_ids = list(train_ids)
        self.test_ids = list(test_ids)
        self.missing_se_ids = set(missing_se_ids)
        self._build()

    def cleanup(self):
        shutil.rmtree(self.root, ignore_errors=True)

    def _build(self):
        for image_id in self.train_ids:
            self._write_sample(image_id, lsd.TRAIN_SPLIT_DIR)
        for image_id in self.test_ids:
            self._write_sample(image_id, lsd.TEST_SPLIT_DIR)

    def _write_sample(self, image_id, split_dir):
        filename = f"IDRiD_{image_id}.jpg"
        image = _synthetic_fundus_image(seed=int(image_id))

        raw_image_dir = os.path.join(self.raw_dir, lsd._ORIGINAL_IMAGES_SUBDIR, split_dir)
        processed_image_dir = os.path.join(self.processed_dir, lsd._ORIGINAL_IMAGES_SUBDIR, split_dir)
        os.makedirs(raw_image_dir, exist_ok=True)
        os.makedirs(processed_image_dir, exist_ok=True)
        Image.fromarray(image).save(os.path.join(raw_image_dir, filename))
        # Stand-in for "already Stage 02-processed" -- this test never runs
        # image_preprocessing.py itself, it only tests that the loader reads
        # whatever it finds under processed_dir, exactly as the real Stage
        # 04 loader must (Stage 02 has already run for real, elsewhere).
        Image.fromarray(image).save(os.path.join(processed_image_dir, filename))

        for class_name, category_folder, suffix in lsd._MASK_SPECS:
            if class_name == "SoftExudate" and image_id in self.missing_se_ids:
                continue  # deliberately no mask file for this id
            mask_dir = os.path.join(self.raw_dir, lsd._GROUNDTRUTHS_SUBDIR, split_dir, category_folder)
            os.makedirs(mask_dir, exist_ok=True)
            mask = _synthetic_lesion_mask(IMAGE_SIZE, seed=int(image_id) + hash(suffix) % 1000)
            Image.fromarray(mask, mode="L").save(os.path.join(mask_dir, f"IDRiD_{image_id}_{suffix}.tif"))


def _build_synthetic_vessel_model():
    """An untrained WNet with a synthetic, from-scratch checkpoint -- same
    pattern as test_vessel_segmentation_device.py. Never touches the real,
    gitignored LWNet checkpoint."""
    model = build_vessel_segmentation_model()
    tmp_dir = tempfile.mkdtemp(prefix="synthetic_vessel_ckpt_")
    checkpoint_path = os.path.join(tmp_dir, "synthetic_checkpoint.pth")
    torch.save(
        {"model_state_dict": model.state_dict(), "optimizer_state_dict": {}, "stats": None},
        checkpoint_path,
    )
    loaded = load_state_dict_from_checkpoint(model, checkpoint_path, device="cpu")
    shutil.rmtree(tmp_dir, ignore_errors=True)
    return loaded


class LesionClassesTests(unittest.TestCase):
    def test_exactly_four_classes_no_optic_disc(self):
        self.assertEqual(len(lsd.LESION_CLASSES), 4)
        self.assertEqual(
            lsd.LESION_CLASSES,
            ("Microaneurysm", "Haemorrhage", "HardExudate", "SoftExudate"),
        )
        self.assertNotIn("OpticDisc", lsd.LESION_CLASSES)
        self.assertEqual(len(lsd._MASK_SPECS), 4)
        self.assertTrue(all("Optic Disc" not in spec[1] for spec in lsd._MASK_SPECS))


class DiscoveryAndSplitTests(unittest.TestCase):
    def setUp(self):
        self.tree = _SyntheticIDRiDTree(
            train_ids=[f"{i:02d}" for i in range(1, 11)],  # 10 synthetic training ids
            test_ids=[f"{i:02d}" for i in range(11, 14)],   # 3 synthetic testing ids
        )
        self.addCleanup(self.tree.cleanup)

    def test_list_original_images_matches_written_files(self):
        entries = lsd._list_original_images(self.tree.raw_dir, lsd.TRAIN_SPLIT_DIR)
        ids = sorted(image_id for image_id, _ in entries)
        self.assertEqual(ids, [f"{i:02d}" for i in range(1, 11)])

    def test_split_is_deterministic_across_calls(self):
        train_a, val_a = lsd.split_train_val_ids(self.tree.raw_dir, val_split=0.3, seed=42)
        train_b, val_b = lsd.split_train_val_ids(self.tree.raw_dir, val_split=0.3, seed=42)
        self.assertEqual(train_a, train_b)
        self.assertEqual(val_a, val_b)

    def test_split_has_no_overlap_and_covers_all_training_ids(self):
        train_ids, val_ids = lsd.split_train_val_ids(self.tree.raw_dir, val_split=0.3, seed=42)
        self.assertEqual(set(train_ids) & set(val_ids), set())
        self.assertEqual(set(train_ids) | set(val_ids), set(f"{i:02d}" for i in range(1, 11)))

    def test_split_never_touches_the_official_test_set(self):
        train_ids, val_ids = lsd.split_train_val_ids(self.tree.raw_dir, val_split=0.3, seed=42)
        test_entries = lsd._list_original_images(self.tree.raw_dir, lsd.TEST_SPLIT_DIR)
        test_ids = {image_id for image_id, _ in test_entries}
        self.assertEqual(test_ids & set(train_ids), set())
        self.assertEqual(test_ids & set(val_ids), set())
        self.assertEqual(test_ids, {"11", "12", "13"})


class MissingSoftExudateMaskTests(unittest.TestCase):
    """The exact behavior called out in the task instructions: a missing
    Soft Exudate mask file means an all-zero channel, not an error."""

    def setUp(self):
        self.vessel_model = _build_synthetic_vessel_model()
        self.tree = _SyntheticIDRiDTree(
            train_ids=["01", "02"], test_ids=[], missing_se_ids={"01"},
        )
        self.addCleanup(self.tree.cleanup)

    def test_missing_se_mask_produces_all_zero_channel(self):
        x, y = lsd._build_sample(
            "01", "IDRiD_01.jpg", self.tree.raw_dir, self.tree.processed_dir,
            lsd.TRAIN_SPLIT_DIR, self.tree.vessel_cache_dir, self.vessel_model,
            image_size=(64, 64),
        )
        se_channel = y[..., lsd.LESION_CLASSES.index("SoftExudate")]
        self.assertTrue(np.all(se_channel == 0.0))

    def test_present_se_mask_produces_nonzero_channel(self):
        x, y = lsd._build_sample(
            "02", "IDRiD_02.jpg", self.tree.raw_dir, self.tree.processed_dir,
            lsd.TRAIN_SPLIT_DIR, self.tree.vessel_cache_dir, self.vessel_model,
            image_size=(64, 64),
        )
        se_channel = y[..., lsd.LESION_CLASSES.index("SoftExudate")]
        self.assertTrue(np.any(se_channel > 0.0))

    def test_missing_processed_image_raises_clear_error(self):
        os.remove(os.path.join(
            self.tree.processed_dir, lsd._ORIGINAL_IMAGES_SUBDIR, lsd.TRAIN_SPLIT_DIR, "IDRiD_01.jpg",
        ))
        with self.assertRaises(FileNotFoundError):
            lsd._build_sample(
                "01", "IDRiD_01.jpg", self.tree.raw_dir, self.tree.processed_dir,
                lsd.TRAIN_SPLIT_DIR, self.tree.vessel_cache_dir, self.vessel_model,
                image_size=(64, 64),
            )


class MultiChannelMaskTests(unittest.TestCase):
    """Regression tests for the real IDRiD_81_EX.tif bug and its fix.

    The real file is RGBA: channel 0 holds the actual HardExudate mask
    (sparse foreground), channels 1-2 are all-zero, and channel 3 (alpha) is
    255 -- fully opaque -- at literally every pixel. The original
    multi-channel-collapse fix (`np.any(mask != 0, axis=-1)`, unqualified)
    treated that constant-255 alpha channel as foreground signal, making the
    entire mask register as foreground. `_load_binary_mask()` must now
    ignore the alpha/opacity channel (detected via PIL's own `mode`, e.g.
    "RGBA"/"LA"/"PA") before collapsing whatever channels remain via "any
    non-alpha channel nonzero" -- not crash, not silently resize, and not
    treat alpha as lesion content."""

    def setUp(self):
        self.tmp_dir = tempfile.mkdtemp(prefix="multichannel_mask_test_")
        self.addCleanup(shutil.rmtree, self.tmp_dir, ignore_errors=True)

    def test_rgba_mask_with_constant_alpha_ignores_alpha_channel(self):
        """The exact real-world IDRiD_81_EX.tif pattern: sparse real content
        in channel 0, channels 1-2 all-zero, alpha (channel 3) opaque (255)
        EVERYWHERE -- not just at one pixel, matching the actual file, not a
        weaker synthetic stand-in. Before the fix this mask would have
        collapsed to 100% foreground; after the fix it must match channel 0
        exactly."""
        h, w = 10, 12
        mask_4ch = np.zeros((h, w, 4), dtype=np.uint8)
        mask_4ch[2, 3, 0] = 255   # the only real lesion pixel, via channel 0
        mask_4ch[6, 1, 0] = 255   # a second real lesion pixel, via channel 0
        mask_4ch[..., 3] = 255    # alpha opaque everywhere, like the real file
        path = os.path.join(self.tmp_dir, "rgba_constant_alpha_mask.tif")
        Image.fromarray(mask_4ch, mode="RGBA").save(path)

        result = lsd._load_binary_mask(path, (h, w))

        self.assertEqual(result.shape, (h, w))
        self.assertEqual(result.dtype, np.uint8)
        self.assertTrue(np.isin(result, [0, 1]).all())
        self.assertEqual(result.sum(), 2, "must match channel 0's 2 real foreground pixels exactly, "
                                           "not the entire (h, w) image via the constant alpha channel")
        self.assertEqual(result[2, 3], 1)
        self.assertEqual(result[6, 1], 1)
        self.assertEqual(result[0, 0], 0, "alpha=255 alone must NOT make this pixel foreground")
        self.assertEqual(result[9, 11], 0, "alpha=255 alone must NOT make this pixel foreground")

    def test_genuine_multichannel_mask_without_alpha_still_collapses_via_any_nonzero(self):
        """A genuinely multi-channel mask that carries NO alpha channel at
        all (mode "RGB", 3 bands) -- the pre-existing "foreground if ANY
        channel is nonzero" collapse must still apply to every channel here,
        proving the alpha-specific carve-out doesn't disable multi-channel
        handling in general. Real content deliberately placed in channel 1
        (not channel 0) to prove this isn't hardcoded to one channel index."""
        h, w = 10, 12
        mask_3ch = np.zeros((h, w, 3), dtype=np.uint8)
        mask_3ch[4, 7, 1] = 255   # foreground via channel 1 only, no alpha channel exists
        path = os.path.join(self.tmp_dir, "rgb_no_alpha_mask.tif")
        Image.fromarray(mask_3ch, mode="RGB").save(path)

        result = lsd._load_binary_mask(path, (h, w))

        self.assertEqual(result.shape, (h, w))
        self.assertEqual(result.sum(), 1)
        self.assertEqual(result[4, 7], 1, "foreground pixel via a single non-alpha channel (mode RGB, no alpha band)")

    def test_normal_2d_mask_behavior_is_unchanged(self):
        h, w = 10, 12
        mask_2d = np.zeros((h, w), dtype=np.uint8)
        mask_2d[4, 6] = 1
        path = os.path.join(self.tmp_dir, "2d_mask.tif")
        Image.fromarray(mask_2d, mode="L").save(path)

        result = lsd._load_binary_mask(path, (h, w))

        self.assertEqual(result.shape, (h, w))
        self.assertEqual(result[4, 6], 1)
        self.assertEqual(result.sum(), 1)

    def test_genuinely_wrong_spatial_shape_still_raises(self):
        h, w = 10, 12
        mask_4ch = np.zeros((h, w, 4), dtype=np.uint8)
        path = os.path.join(self.tmp_dir, "wrong_shape_mask.tif")
        Image.fromarray(mask_4ch, mode="RGBA").save(path)

        with self.assertRaises(RuntimeError):
            lsd._load_binary_mask(path, (h + 1, w))  # real (H, W) mismatch -- must still raise


class SampleShapeAndValueTests(unittest.TestCase):
    def setUp(self):
        self.vessel_model = _build_synthetic_vessel_model()
        self.tree = _SyntheticIDRiDTree(train_ids=["01"], test_ids=[])
        self.addCleanup(self.tree.cleanup)

    def test_input_and_target_shapes_and_dtypes(self):
        x, y = lsd._build_sample(
            "01", "IDRiD_01.jpg", self.tree.raw_dir, self.tree.processed_dir,
            lsd.TRAIN_SPLIT_DIR, self.tree.vessel_cache_dir, self.vessel_model,
            image_size=(48, 48),
        )
        self.assertEqual(x.shape, (48, 48, 4))
        self.assertEqual(y.shape, (48, 48, 4))
        self.assertEqual(x.dtype, np.float32)
        self.assertEqual(y.dtype, np.float32)

    def test_input_values_in_unit_range(self):
        x, _ = lsd._build_sample(
            "01", "IDRiD_01.jpg", self.tree.raw_dir, self.tree.processed_dir,
            lsd.TRAIN_SPLIT_DIR, self.tree.vessel_cache_dir, self.vessel_model,
            image_size=(48, 48),
        )
        self.assertTrue(np.isfinite(x).all())
        self.assertGreaterEqual(x.min(), 0.0)
        self.assertLessEqual(x.max(), 1.0)

    def test_target_values_are_strictly_binary(self):
        _, y = lsd._build_sample(
            "01", "IDRiD_01.jpg", self.tree.raw_dir, self.tree.processed_dir,
            lsd.TRAIN_SPLIT_DIR, self.tree.vessel_cache_dir, self.vessel_model,
            image_size=(48, 48),
        )
        self.assertTrue(np.isin(y, [0.0, 1.0]).all())


class VesselCacheTests(unittest.TestCase):
    def setUp(self):
        self.vessel_model = _build_synthetic_vessel_model()
        self.tree = _SyntheticIDRiDTree(train_ids=["01"], test_ids=[])
        self.addCleanup(self.tree.cleanup)

    def test_second_call_reuses_cache_without_recomputing_inference(self):
        with mock.patch(
            "lesion_segmentation_dataset.predict_vessel_mask", wraps=lsd.predict_vessel_mask,
        ) as mocked_predict:
            lsd._build_sample(
                "01", "IDRiD_01.jpg", self.tree.raw_dir, self.tree.processed_dir,
                lsd.TRAIN_SPLIT_DIR, self.tree.vessel_cache_dir, self.vessel_model,
                image_size=(48, 48),
            )
            self.assertEqual(mocked_predict.call_count, 1)

            lsd._build_sample(
                "01", "IDRiD_01.jpg", self.tree.raw_dir, self.tree.processed_dir,
                lsd.TRAIN_SPLIT_DIR, self.tree.vessel_cache_dir, self.vessel_model,
                image_size=(48, 48),
            )
            # Still 1 -- the second call must hit the on-disk .npy cache,
            # never re-running Stage 03 inference for the same image.
            self.assertEqual(mocked_predict.call_count, 1)

    def test_cache_file_is_written_to_disk(self):
        lsd._build_sample(
            "01", "IDRiD_01.jpg", self.tree.raw_dir, self.tree.processed_dir,
            lsd.TRAIN_SPLIT_DIR, self.tree.vessel_cache_dir, self.vessel_model,
            image_size=(48, 48),
        )
        cache_path = lsd._vessel_cache_path(self.tree.vessel_cache_dir, lsd.TRAIN_SPLIT_DIR, "01")
        self.assertTrue(os.path.exists(cache_path))


class TfDataPipelineTests(unittest.TestCase):
    """End-to-end: the public tf.data-producing functions, over a tiny
    synthetic dataset."""

    def setUp(self):
        self.vessel_model = _build_synthetic_vessel_model()
        self.tree = _SyntheticIDRiDTree(
            train_ids=[f"{i:02d}" for i in range(1, 7)],  # 6 synthetic training ids
            test_ids=["07", "08"],
        )
        self.addCleanup(self.tree.cleanup)

    def test_train_val_datasets_yield_correctly_shaped_batches(self):
        train_ds, val_ds = lsd.load_lesion_segmentation_datasets(
            raw_dir=self.tree.raw_dir, processed_dir=self.tree.processed_dir,
            vessel_cache_dir=self.tree.vessel_cache_dir, vessel_model=self.vessel_model,
            image_size=(32, 32), val_split=0.34, batch_size=2, seed=42, augment_train=True,
        )
        x_batch, y_batch = next(iter(train_ds))
        self.assertEqual(x_batch.shape[1:], (32, 32, 4))
        self.assertEqual(y_batch.shape[1:], (32, 32, 4))
        self.assertLessEqual(int(x_batch.shape[0]), 2)

        x_val, y_val = next(iter(val_ds))
        self.assertEqual(x_val.shape[1:], (32, 32, 4))
        self.assertEqual(y_val.shape[1:], (32, 32, 4))

    def test_test_dataset_uses_only_official_test_split(self):
        test_ds = lsd.load_lesion_segmentation_test_dataset(
            raw_dir=self.tree.raw_dir, processed_dir=self.tree.processed_dir,
            vessel_cache_dir=self.tree.vessel_cache_dir, vessel_model=self.vessel_model,
            image_size=(32, 32), batch_size=2,
        )
        total = sum(int(x.shape[0]) for x, _ in test_ds)
        self.assertEqual(total, 2)  # exactly the 2 synthetic test ids, never train/val ids

    def test_val_split_is_never_augmented(self):
        # augment=False for val -- two independent builds of the same val
        # sample must be pixel-identical (no random flips/rotation applied).
        train_ds_a, val_ds_a = lsd.load_lesion_segmentation_datasets(
            raw_dir=self.tree.raw_dir, processed_dir=self.tree.processed_dir,
            vessel_cache_dir=self.tree.vessel_cache_dir, vessel_model=self.vessel_model,
            image_size=(32, 32), val_split=0.34, batch_size=8, seed=42, augment_train=False,
        )
        x_a, _ = next(iter(val_ds_a))

        train_ds_b, val_ds_b = lsd.load_lesion_segmentation_datasets(
            raw_dir=self.tree.raw_dir, processed_dir=self.tree.processed_dir,
            vessel_cache_dir=self.tree.vessel_cache_dir, vessel_model=self.vessel_model,
            image_size=(32, 32), val_split=0.34, batch_size=8, seed=42, augment_train=False,
        )
        x_b, _ = next(iter(val_ds_b))

        np.testing.assert_array_equal(np.asarray(x_a), np.asarray(x_b))


if __name__ == "__main__":
    unittest.main()
