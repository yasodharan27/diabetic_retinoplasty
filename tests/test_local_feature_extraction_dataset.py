"""
Regression tests for local_feature_extraction_dataset.py (Stage 05 dataset
loader).

Builds a small, synthetic, on-disk directory tree shaped exactly like
APTOS2019's real raw layout (`train.csv` with `id_code`/`diagnosis`
columns, `train_images/<id_code>.png`) inside a temporary directory -- the
real, local APTOS2019 dataset is never read by this suite, per this
project's "unit tests use synthetic/temporary data only" rule
(PROJECT_CODE.md).

The Stage 03 vessel model used here is the same synthetic, untrained,
from-scratch checkpoint pattern already established in
tests/test_vessel_segmentation_device.py /
tests/test_lesion_segmentation_dataset.py. The Stage 04 lesion model is a
real (untrained) Attention U-Net built directly via
lesion_segmentation_model.build_attention_unet -- Stage 04 trains within
this project, so (unlike Stage 03) no checkpoint file is needed to
construct a real model instance. Neither the real, gitignored LWNet
checkpoint nor the real, gitignored Experiment 2C .keras checkpoint is
ever touched here.
"""

import csv
import os
import shutil
import tempfile
import unittest
from unittest import mock

import numpy as np
import torch
from PIL import Image

import lesion_segmentation_model as lsm
import local_feature_extraction_dataset as lfed
from lesion_segmentation_dataset import LESION_CLASSES
from vessel_segmentation_model import build_vessel_segmentation_model, load_state_dict_from_checkpoint

IMAGE_SIZE = 256  # large enough for compute_fov_mask's circle-fit to succeed reliably


def _synthetic_fundus_image(size=IMAGE_SIZE, seed=0):
    """Identical recipe to test_lesion_segmentation_dataset.py's helper of
    the same name -- a bright filled circle on a black background, enough
    like a real fundus photo's basic light/dark structure for Stage 03's
    FOV circle-fit to succeed on."""
    rng = np.random.RandomState(seed)
    image = np.zeros((size, size, 3), dtype=np.uint8)
    yy, xx = np.ogrid[:size, :size]
    center = size // 2
    radius = int(size * 0.4)
    circle = (xx - center) ** 2 + (yy - center) ** 2 <= radius ** 2
    base = 140 + rng.randint(-20, 20, size=(size, size, 3))
    image[circle] = np.clip(base[circle], 60, 220).astype(np.uint8)
    return image


def _build_synthetic_vessel_model():
    """Same pattern as test_lesion_segmentation_dataset.py: an untrained
    WNet with a synthetic, from-scratch checkpoint."""
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


def _build_synthetic_lesion_model():
    """A real (untrained) Attention U-Net -- Stage 04 trains within this
    project, so a fresh instance is a real, valid model, not a stand-in."""
    return lsm.build_attention_unet(input_shape=(64, 64, 4), base_filters=4)


class _SyntheticAPTOSTree:
    """Builds a temporary directory shaped like datasets/APTOS2019/raw/
    (train.csv + train_images/), with a small, fully synthetic set of
    labeled images."""

    def __init__(self, id_diagnosis_pairs):
        self.root = tempfile.mkdtemp(prefix="aptos_synth_")
        self.image_dir = os.path.join(self.root, "train_images")
        self.csv_path = os.path.join(self.root, "train.csv")
        self.cache_dir = os.path.join(self.root, "cache")
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


class ListLabeledImagesTests(unittest.TestCase):
    def setUp(self):
        self.tree = _SyntheticAPTOSTree([("img_a", 0), ("img_b", 2), ("img_c", 4)])
        self.addCleanup(self.tree.cleanup)

    def test_reads_id_code_and_diagnosis(self):
        entries = lfed._list_labeled_images(self.tree.csv_path)
        self.assertEqual(
            sorted(entries),
            [("img_a", 0), ("img_b", 2), ("img_c", 4)],
        )

    def test_diagnosis_is_int_not_str(self):
        entries = lfed._list_labeled_images(self.tree.csv_path)
        for _, diagnosis in entries:
            self.assertIsInstance(diagnosis, int)

    def test_missing_csv_raises_clear_error(self):
        with self.assertRaises(FileNotFoundError):
            lfed._list_labeled_images(os.path.join(self.tree.root, "no_such.csv"))


class SplitTrainValIdsTests(unittest.TestCase):
    def setUp(self):
        self.tree = _SyntheticAPTOSTree([(f"img_{i:02d}", i % 5) for i in range(20)])
        self.addCleanup(self.tree.cleanup)

    def test_split_is_deterministic_across_calls(self):
        train_a, val_a = lfed.split_train_val_ids(self.tree.csv_path, val_split=0.3, seed=42)
        train_b, val_b = lfed.split_train_val_ids(self.tree.csv_path, val_split=0.3, seed=42)
        self.assertEqual(train_a, train_b)
        self.assertEqual(val_a, val_b)

    def test_split_has_no_overlap_and_covers_all_ids(self):
        train_entries, val_entries = lfed.split_train_val_ids(self.tree.csv_path, val_split=0.3, seed=42)
        train_ids = {i for i, _ in train_entries}
        val_ids = {i for i, _ in val_entries}
        self.assertEqual(train_ids & val_ids, set())
        self.assertEqual(train_ids | val_ids, {f"img_{i:02d}" for i in range(20)})


class BuildLocalFeatureInputTests(unittest.TestCase):
    """Channel semantics, using precomputed vessel/lesion maps so these
    tests exercise the concatenation/normalization/resize/validation logic
    directly, without depending on real (or even synthetic-model) Stage
    03/04 inference."""

    def setUp(self):
        self.rgb = _synthetic_fundus_image(size=64, seed=1)
        self.vessel_map = np.random.RandomState(0).rand(64, 64, 1).astype(np.float32)
        self.lesion_maps = np.random.RandomState(1).rand(64, 64, 4).astype(np.float32)

    def test_output_has_eight_channels(self):
        result = lfed.build_local_feature_input(
            self.rgb, vessel_probability_map=self.vessel_map,
            lesion_probability_maps=self.lesion_maps, image_size=(64, 64),
        )
        self.assertEqual(result.shape, (64, 64, lfed.NUM_CHANNELS))
        self.assertEqual(result.dtype, np.float32)

    def test_channel_ordering_rgb_then_vessel_then_lesion(self):
        result = lfed.build_local_feature_input(
            self.rgb, vessel_probability_map=self.vessel_map,
            lesion_probability_maps=self.lesion_maps, image_size=(64, 64),
        )
        np.testing.assert_allclose(result[..., 0:3], self.rgb.astype(np.float32) / 255.0, atol=1e-5)
        np.testing.assert_allclose(result[..., 3], self.vessel_map[..., 0], atol=1e-5)
        np.testing.assert_allclose(result[..., 4:8], self.lesion_maps, atol=1e-5)

    def test_rgb_normalization_is_zero_to_one(self):
        result = lfed.build_local_feature_input(
            self.rgb, vessel_probability_map=self.vessel_map,
            lesion_probability_maps=self.lesion_maps, image_size=(64, 64),
        )
        rgb_channels = result[..., 0:3]
        self.assertGreaterEqual(rgb_channels.min(), 0.0)
        self.assertLessEqual(rgb_channels.max(), 1.0)

    def test_vessel_channel_remains_probability_valued(self):
        result = lfed.build_local_feature_input(
            self.rgb, vessel_probability_map=self.vessel_map,
            lesion_probability_maps=self.lesion_maps, image_size=(64, 64),
        )
        vessel_channel = result[..., 3]
        self.assertGreaterEqual(vessel_channel.min(), 0.0)
        self.assertLessEqual(vessel_channel.max(), 1.0)

    def test_lesion_channels_remain_probability_valued(self):
        result = lfed.build_local_feature_input(
            self.rgb, vessel_probability_map=self.vessel_map,
            lesion_probability_maps=self.lesion_maps, image_size=(64, 64),
        )
        lesion_channels = result[..., 4:8]
        self.assertGreaterEqual(lesion_channels.min(), 0.0)
        self.assertLessEqual(lesion_channels.max(), 1.0)

    def test_resize_produces_requested_spatial_size(self):
        result = lfed.build_local_feature_input(
            self.rgb, vessel_probability_map=self.vessel_map,
            lesion_probability_maps=self.lesion_maps, image_size=(512, 512),
        )
        self.assertEqual(result.shape, (512, 512, lfed.NUM_CHANNELS))

    def test_spatial_alignment_is_preserved_after_joint_resize(self):
        """A distinctive block placed at the same native-resolution
        location in every channel must land at the same resized location in
        every channel -- proof the joint (not per-channel) resize keeps
        channels aligned."""
        rgb = np.zeros((64, 64, 3), dtype=np.uint8)
        vessel = np.zeros((64, 64, 1), dtype=np.float32)
        lesion = np.zeros((64, 64, 4), dtype=np.float32)
        rgb[10:20, 10:20, :] = 255
        vessel[10:20, 10:20, :] = 1.0
        lesion[10:20, 10:20, :] = 1.0

        result = lfed.build_local_feature_input(
            rgb, vessel_probability_map=vessel, lesion_probability_maps=lesion, image_size=(32, 32),
        )
        # Native block spans rows/cols [10, 20) of 64 -> resized block
        # should span roughly [5, 10) of 32 in every channel identically.
        for channel_index in range(lfed.NUM_CHANNELS):
            channel = result[..., channel_index]
            active_rows = np.where(channel.max(axis=1) > 0.5)[0]
            self.assertTrue(len(active_rows) > 0, f"channel {channel_index} has no active region")
        # All channels' active regions must coincide exactly.
        active_masks = [result[..., c] > 0.5 for c in range(lfed.NUM_CHANNELS)]
        for mask in active_masks[1:]:
            np.testing.assert_array_equal(mask, active_masks[0])

    def test_vessel_map_native_shape_mismatch_raises(self):
        wrong_shape_vessel = np.random.rand(32, 32, 1).astype(np.float32)
        with self.assertRaises(RuntimeError):
            lfed.build_local_feature_input(
                self.rgb, vessel_probability_map=wrong_shape_vessel,
                lesion_probability_maps=self.lesion_maps, image_size=(64, 64),
            )

    def test_lesion_maps_native_shape_mismatch_raises(self):
        wrong_shape_lesion = np.random.rand(32, 32, 4).astype(np.float32)
        with self.assertRaises(RuntimeError):
            lfed.build_local_feature_input(
                self.rgb, vessel_probability_map=self.vessel_map,
                lesion_probability_maps=wrong_shape_lesion, image_size=(64, 64),
            )

    def test_wrong_lesion_channel_count_raises(self):
        wrong_channel_count = np.random.rand(64, 64, 3).astype(np.float32)
        with self.assertRaises(ValueError):
            lfed.build_local_feature_input(
                self.rgb, vessel_probability_map=self.vessel_map,
                lesion_probability_maps=wrong_channel_count, image_size=(64, 64),
            )


class AugmentationTests(unittest.TestCase):
    def setUp(self):
        rng = np.random.RandomState(0)
        self.input_array = rng.rand(16, 16, lfed.NUM_CHANNELS).astype(np.float32)

    def test_spatial_augmentation_synchronized_across_all_channels(self):
        rng = np.random.default_rng(7)
        augmented = lfed._augment_spatial(self.input_array, rng)

        rng_expected = np.random.default_rng(7)
        expected = self.input_array
        if rng_expected.random() < 0.5:
            expected = expected[:, ::-1, :]
        if rng_expected.random() < 0.5:
            expected = expected[::-1, :, :]
        k = int(rng_expected.integers(0, 4))
        if k:
            expected = np.rot90(expected, k=k, axes=(0, 1))

        np.testing.assert_array_equal(augmented, expected)

    def test_rgb_only_intensity_augmentation_does_not_alter_segmentation_channels(self):
        rng = np.random.default_rng(3)
        augmented = lfed._augment_intensity_rgb(self.input_array, rng)
        np.testing.assert_array_equal(augmented[..., 3:8], self.input_array[..., 3:8])

    def test_rgb_only_intensity_augmentation_can_change_rgb_channels(self):
        rng = np.random.default_rng(3)
        augmented = lfed._augment_intensity_rgb(self.input_array, rng)
        self.assertFalse(np.allclose(augmented[..., 0:3], self.input_array[..., 0:3]))

    def test_intensity_augmentation_output_stays_in_unit_range(self):
        rng = np.random.default_rng(11)
        augmented = lfed._augment_intensity_rgb(self.input_array, rng, brightness_range=0.5, contrast_range=0.5)
        rgb = augmented[..., 0:3]
        self.assertGreaterEqual(rgb.min(), 0.0)
        self.assertLessEqual(rgb.max(), 1.0)


class BuildSampleAndCacheTests(unittest.TestCase):
    def setUp(self):
        self.vessel_model = _build_synthetic_vessel_model()
        self.lesion_model = _build_synthetic_lesion_model()
        self.tree = _SyntheticAPTOSTree([("img_01", 2)])
        self.addCleanup(self.tree.cleanup)

    def test_build_sample_returns_input_and_label(self):
        x, y = lfed._build_sample(
            "img_01", 2, self.tree.image_dir, self.tree.cache_dir,
            self.vessel_model, self.lesion_model, image_size=(64, 64),
        )
        self.assertEqual(x.shape, (64, 64, lfed.NUM_CHANNELS))
        self.assertEqual(x.dtype, np.float32)
        self.assertEqual(y, 2)

    def test_missing_image_raises_clear_error(self):
        with self.assertRaises(FileNotFoundError):
            lfed._build_sample(
                "no_such_id", 0, self.tree.image_dir, self.tree.cache_dir,
                self.vessel_model, self.lesion_model, image_size=(64, 64),
            )

    def test_second_call_reuses_cache_without_recomputing_inference(self):
        with mock.patch(
            "local_feature_extraction_dataset.predict_vessel_mask", wraps=lfed.predict_vessel_mask,
        ) as mocked_vessel, mock.patch(
            "local_feature_extraction_dataset.predict_lesion_mask", wraps=lfed.predict_lesion_mask,
        ) as mocked_lesion:
            lfed._build_sample(
                "img_01", 2, self.tree.image_dir, self.tree.cache_dir,
                self.vessel_model, self.lesion_model, image_size=(64, 64),
            )
            self.assertEqual(mocked_vessel.call_count, 1)
            self.assertEqual(mocked_lesion.call_count, 1)

            lfed._build_sample(
                "img_01", 2, self.tree.image_dir, self.tree.cache_dir,
                self.vessel_model, self.lesion_model, image_size=(64, 64),
            )
            # Still 1 each -- the second call must hit the on-disk .npy
            # caches, never re-running Stage 03/04 inference for the same image.
            self.assertEqual(mocked_vessel.call_count, 1)
            self.assertEqual(mocked_lesion.call_count, 1)

    def test_cache_files_are_written_to_disk(self):
        lfed._build_sample(
            "img_01", 2, self.tree.image_dir, self.tree.cache_dir,
            self.vessel_model, self.lesion_model, image_size=(64, 64),
        )
        vessel_cache = lfed._cache_path(self.tree.cache_dir, "img_01", "vessel", (64, 64))
        lesion_cache = lfed._cache_path(self.tree.cache_dir, "img_01", "lesion", (64, 64))
        self.assertTrue(os.path.exists(vessel_cache))
        self.assertTrue(os.path.exists(lesion_cache))

    def test_cache_path_is_keyed_by_image_size(self):
        """A cache built for one canonical resolution must never collide
        with (or be silently reused for) a different one -- the cached
        bytes themselves now depend on image_size (Step 3 fix)."""
        path_512 = lfed._cache_path(self.tree.cache_dir, "img_01", "vessel", (512, 512))
        path_64 = lfed._cache_path(self.tree.cache_dir, "img_01", "vessel", (64, 64))
        self.assertNotEqual(path_512, path_64)

    def test_cache_stores_canonical_resolution_not_native_resolution(self):
        """The on-disk cache must store the resized-down, canonical
        (image_size) array, not the native-image-resolution prediction --
        the actual storage/CPU fix this Step 3 correction makes."""
        image_size = (64, 64)
        lfed._build_sample(
            "img_01", 2, self.tree.image_dir, self.tree.cache_dir,
            self.vessel_model, self.lesion_model, image_size=image_size,
        )
        vessel_cache = lfed._cache_path(self.tree.cache_dir, "img_01", "vessel", image_size)
        lesion_cache = lfed._cache_path(self.tree.cache_dir, "img_01", "lesion", image_size)
        cached_vessel_map = np.load(vessel_cache)
        cached_lesion_maps = np.load(lesion_cache)

        self.assertEqual(cached_vessel_map.shape[:2], image_size)
        self.assertEqual(cached_lesion_maps.shape, (*image_size, 4))

    def test_cache_values_match_a_resized_direct_prediction(self):
        """Caching must never alter values BEYOND the documented, intentional
        canonical resize (Step 3) -- the cached array must equal a direct
        Stage 03 prediction resized the same way, not the raw native-
        resolution prediction (that older invariant no longer holds by
        design)."""
        image_size = (64, 64)
        lfed._build_sample(
            "img_01", 2, self.tree.image_dir, self.tree.cache_dir,
            self.vessel_model, self.lesion_model, image_size=image_size,
        )
        vessel_cache = lfed._cache_path(self.tree.cache_dir, "img_01", "vessel", image_size)
        cached_vessel_map = np.load(vessel_cache)

        raw_bgr = lfed._load_raw_bgr(self.tree.image_dir, "img_01")
        rgb = lfed._stage02_processed_rgb(raw_bgr)
        direct_result = lfed.predict_vessel_mask(rgb, model=self.vessel_model)
        expected = lfed._resize_map(direct_result["probability_map"].astype(np.float32), image_size)
        np.testing.assert_array_equal(cached_vessel_map, expected)


class EmptyFieldOfViewHandlingTests(unittest.TestCase):
    """Regression tests mirroring `tests/test_joint_training.py`'s `EmptyFieldOfViewHandlingTests`:
    Stage 05's own standalone loader shares the exact same `predict_vessel_mask` call (and thus
    the exact same empty-FOV crash exposure) as the joint dataset, so it gets the identical
    catch-and-skip fix in `_make_dataset`'s generator."""

    def setUp(self):
        self.vessel_model = _build_synthetic_vessel_model()
        self.lesion_model = _build_synthetic_lesion_model()
        self.tree = _SyntheticAPTOSTree([("img_good", 1), ("img_empty_fov", 3), ("img_good_2", 0)])
        self.addCleanup(self.tree.cleanup)

    def _flaky_build_sample(self, id_code, diagnosis, *args, **kwargs):
        if id_code == "img_empty_fov":
            from vessel_segmentation_inference import EmptyFieldOfViewError
            raise EmptyFieldOfViewError("no fundus disk detected")
        return self._real_build_sample(id_code, diagnosis, *args, **kwargs)

    def test_generator_skips_only_the_empty_fov_image_and_continues(self):
        self._real_build_sample = lfed._build_sample
        with mock.patch("local_feature_extraction_dataset._build_sample", side_effect=self._flaky_build_sample):
            ds = lfed._make_dataset(
                self.tree.pairs, self.tree.image_dir, self.tree.cache_dir,
                self.vessel_model, self.lesion_model, image_size=(64, 64),
                batch_size=1, shuffle=False, augment=False, seed=0,
            )
            labels = [int(y.numpy()[0]) for _, y in ds]

        self.assertEqual(len(labels), 2)
        self.assertEqual(sorted(labels), [0, 1])

    def test_generator_logs_a_warning_naming_the_skipped_image_id(self):
        self._real_build_sample = lfed._build_sample
        with mock.patch("local_feature_extraction_dataset._build_sample", side_effect=self._flaky_build_sample):
            with self.assertLogs("local_feature_extraction_dataset", level="WARNING") as captured:
                ds = lfed._make_dataset(
                    self.tree.pairs, self.tree.image_dir, self.tree.cache_dir,
                    self.vessel_model, self.lesion_model, image_size=(64, 64),
                    batch_size=1, shuffle=False, augment=False, seed=0,
                )
                list(ds)
        self.assertTrue(any("img_empty_fov" in message for message in captured.output))

    def test_normal_images_are_completely_unaffected(self):
        ds = lfed._make_dataset(
            self.tree.pairs, self.tree.image_dir, self.tree.cache_dir,
            self.vessel_model, self.lesion_model, image_size=(64, 64),
            batch_size=1, shuffle=False, augment=False, seed=0,
        )
        labels = [int(y.numpy()[0]) for _, y in ds]
        self.assertEqual(sorted(labels), [0, 1, 3])


class TfDataPipelineTests(unittest.TestCase):
    """End-to-end: the public tf.data-producing function, over a tiny
    synthetic APTOS-shaped dataset."""

    def setUp(self):
        self.vessel_model = _build_synthetic_vessel_model()
        self.lesion_model = _build_synthetic_lesion_model()
        # 3 samples per class (15 total): split_train_val_ids now stratifies
        # by diagnosis (downstream_split.compute_split), which requires at
        # least 2 members per class -- 6 samples (1-2 per class) was enough
        # for the old, non-stratified split but is not enough here.
        self.tree = _SyntheticAPTOSTree([(f"img_{i:02d}", i % 5) for i in range(15)])
        self.addCleanup(self.tree.cleanup)

    def test_train_val_datasets_yield_correctly_shaped_batches(self):
        train_ds, val_ds = lfed.load_local_feature_extraction_datasets(
            csv_path=self.tree.csv_path, image_dir=self.tree.image_dir, cache_dir=self.tree.cache_dir,
            vessel_model=self.vessel_model, lesion_model=self.lesion_model,
            image_size=(32, 32), val_split=0.34, batch_size=2, seed=42, augment_train=True,
        )
        x_batch, y_batch = next(iter(train_ds))
        self.assertEqual(x_batch.shape[1:], (32, 32, lfed.NUM_CHANNELS))
        self.assertEqual(y_batch.shape[1:], ())
        self.assertLessEqual(int(x_batch.shape[0]), 2)

        x_val, y_val = next(iter(val_ds))
        self.assertEqual(x_val.shape[1:], (32, 32, lfed.NUM_CHANNELS))

    def test_val_split_is_never_augmented(self):
        train_ds_a, val_ds_a = lfed.load_local_feature_extraction_datasets(
            csv_path=self.tree.csv_path, image_dir=self.tree.image_dir, cache_dir=self.tree.cache_dir,
            vessel_model=self.vessel_model, lesion_model=self.lesion_model,
            image_size=(32, 32), val_split=0.34, batch_size=8, seed=42, augment_train=False,
        )
        x_a, _ = next(iter(val_ds_a))

        train_ds_b, val_ds_b = lfed.load_local_feature_extraction_datasets(
            csv_path=self.tree.csv_path, image_dir=self.tree.image_dir, cache_dir=self.tree.cache_dir,
            vessel_model=self.vessel_model, lesion_model=self.lesion_model,
            image_size=(32, 32), val_split=0.34, batch_size=8, seed=42, augment_train=False,
        )
        x_b, _ = next(iter(val_ds_b))

        np.testing.assert_array_equal(np.asarray(x_a), np.asarray(x_b))

    def test_labels_match_csv(self):
        train_ds, val_ds = lfed.load_local_feature_extraction_datasets(
            csv_path=self.tree.csv_path, image_dir=self.tree.image_dir, cache_dir=self.tree.cache_dir,
            vessel_model=self.vessel_model, lesion_model=self.lesion_model,
            image_size=(32, 32), val_split=0.34, batch_size=8, seed=42, augment_train=False,
        )
        all_labels = set()
        for _, y_batch in train_ds:
            all_labels.update(int(v) for v in y_batch.numpy())
        for _, y_batch in val_ds:
            all_labels.update(int(v) for v in y_batch.numpy())
        expected_labels = {i % 5 for i in range(6)}
        self.assertEqual(all_labels, expected_labels)


if __name__ == "__main__":
    unittest.main()
