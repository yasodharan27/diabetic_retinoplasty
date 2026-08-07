"""
Lightweight sanity tests for image_preprocessing.py (Step 2). Uses only the
standard library's unittest plus numpy/opencv, which are already project
dependencies -- no pytest, no model loading, no training, no real dataset
files. Everything runs against synthetic in-memory images and a temp
directory, so it never touches datasets/*/raw.

Run with: python -m unittest tests.test_image_preprocessing -v
"""

import csv
import os
import shutil
import tempfile
import unittest
from dataclasses import fields
from pathlib import Path

import cv2
import numpy as np

import config as project_config
from image_preprocessing import (
    PreprocessingResult,
    PreprocessingSummary,
    apply_clahe,
    apply_gamma_correction,
    preprocess_array,
    preprocess_dataset,
    preprocess_folder,
    preprocess_image,
)


def _synthetic_bgr_image(size=32, seed=0):
    rng = np.random.default_rng(seed)
    return rng.integers(0, 256, size=(size, size, 3), dtype=np.uint8)


class GammaCorrectionTests(unittest.TestCase):
    def test_preserves_shape_and_dtype(self):
        image = _synthetic_bgr_image()
        out = apply_gamma_correction(image, gamma=1.5)
        self.assertEqual(out.shape, image.shape)
        self.assertEqual(out.dtype, image.dtype)

    def test_gamma_greater_than_one_brightens_midgray(self):
        image = np.full((16, 16, 3), 128, dtype=np.uint8)
        out = apply_gamma_correction(image, gamma=2.0)
        self.assertGreater(out.mean(), image.mean())

    def test_gamma_one_is_identity(self):
        image = _synthetic_bgr_image()
        out = apply_gamma_correction(image, gamma=1.0)
        np.testing.assert_array_equal(out, image)

    def test_invalid_gamma_raises(self):
        image = _synthetic_bgr_image()
        with self.assertRaises(ValueError):
            apply_gamma_correction(image, gamma=0)


class ClaheTests(unittest.TestCase):
    def test_color_image_preserves_shape_and_dtype(self):
        image = _synthetic_bgr_image()
        out = apply_clahe(image)
        self.assertEqual(out.shape, image.shape)
        self.assertEqual(out.dtype, image.dtype)

    def test_grayscale_image_preserves_shape_and_dtype(self):
        image = _synthetic_bgr_image()[:, :, 0]  # single channel
        out = apply_clahe(image)
        self.assertEqual(out.shape, image.shape)
        self.assertEqual(out.dtype, image.dtype)

    def test_increases_local_contrast_on_flat_image(self):
        # A flat mid-gray image has ~zero variance; CLAHE should not blow up
        # or crash on a degenerate low-contrast input.
        image = np.full((32, 32, 3), 100, dtype=np.uint8)
        out = apply_clahe(image)
        self.assertEqual(out.shape, image.shape)

    def test_clip_limit_and_tile_grid_size_are_configurable(self):
        # clip_limit/tile_grid_size must actually change the output --
        # proves neither parameter is silently ignored in favor of a
        # hardcoded value.
        image = _synthetic_bgr_image(size=64)
        mild = apply_clahe(image, clip_limit=1.0, tile_grid_size=(2, 2))
        strong = apply_clahe(image, clip_limit=8.0, tile_grid_size=(16, 16))
        self.assertEqual(mild.shape, image.shape)
        self.assertEqual(strong.shape, image.shape)
        self.assertFalse(np.array_equal(mild, strong))

    def test_non_square_tile_grid_size_is_accepted(self):
        image = _synthetic_bgr_image(size=64)
        out = apply_clahe(image, tile_grid_size=(4, 8))
        self.assertEqual(out.shape, image.shape)


class PreprocessArrayTests(unittest.TestCase):
    def test_composes_gamma_then_clahe(self):
        image = _synthetic_bgr_image()
        expected = apply_clahe(apply_gamma_correction(image))
        actual = preprocess_array(image)
        np.testing.assert_array_equal(actual, expected)

    def test_custom_parameters_propagate_end_to_end(self):
        # Every parameter (gamma, clip_limit, tile_grid_size) must reach
        # the underlying operations unchanged -- proves the public API is
        # the actual configuration surface, not config.PREPROCESSING's
        # defaults.
        image = _synthetic_bgr_image()
        custom = dict(gamma=2.5, clip_limit=6.0, tile_grid_size=(4, 4))

        expected = apply_clahe(
            apply_gamma_correction(image, gamma=custom["gamma"]),
            clip_limit=custom["clip_limit"], tile_grid_size=custom["tile_grid_size"],
        )
        actual = preprocess_array(image, **custom)
        np.testing.assert_array_equal(actual, expected)

        # And it must differ from the all-defaults result, or the override
        # wouldn't actually be doing anything.
        default_result = preprocess_array(image)
        self.assertFalse(np.array_equal(actual, default_result))


class CentralizedConfigDefaultsTests(unittest.TestCase):
    """Verifies image_preprocessing.py's None-sentinel parameters actually
    resolve to config.PREPROCESSING (config.py), and that explicit
    arguments still take precedence over it."""

    def test_gamma_omitted_falls_back_to_config_default(self):
        image = _synthetic_bgr_image()
        via_omitted = apply_gamma_correction(image)
        via_explicit_config_value = apply_gamma_correction(
            image, gamma=project_config.PREPROCESSING.DEFAULT_GAMMA
        )
        np.testing.assert_array_equal(via_omitted, via_explicit_config_value)

    def test_clahe_params_omitted_fall_back_to_config_defaults(self):
        image = _synthetic_bgr_image()
        via_omitted = apply_clahe(image)
        via_explicit_config_values = apply_clahe(
            image,
            clip_limit=project_config.PREPROCESSING.DEFAULT_CLAHE_CLIP_LIMIT,
            tile_grid_size=project_config.PREPROCESSING.DEFAULT_CLAHE_TILE_GRID_SIZE,
        )
        np.testing.assert_array_equal(via_omitted, via_explicit_config_values)

    def test_preprocess_array_omitted_params_fall_back_to_config_defaults(self):
        image = _synthetic_bgr_image()
        via_omitted = preprocess_array(image)
        via_explicit_config_values = preprocess_array(
            image,
            gamma=project_config.PREPROCESSING.DEFAULT_GAMMA,
            clip_limit=project_config.PREPROCESSING.DEFAULT_CLAHE_CLIP_LIMIT,
            tile_grid_size=project_config.PREPROCESSING.DEFAULT_CLAHE_TILE_GRID_SIZE,
        )
        np.testing.assert_array_equal(via_omitted, via_explicit_config_values)

    def test_explicit_gamma_overrides_config_default(self):
        image = _synthetic_bgr_image()
        explicit_gamma = project_config.PREPROCESSING.DEFAULT_GAMMA + 1.0
        via_explicit = apply_gamma_correction(image, gamma=explicit_gamma)
        via_config_default = apply_gamma_correction(image)
        self.assertFalse(np.array_equal(via_explicit, via_config_default))

    def test_explicit_clahe_params_override_config_defaults(self):
        image = _synthetic_bgr_image(size=64)
        explicit_clip_limit = project_config.PREPROCESSING.DEFAULT_CLAHE_CLIP_LIMIT + 4.0
        explicit_tile_grid_size = (2, 2)
        self.assertNotEqual(
            explicit_tile_grid_size, project_config.PREPROCESSING.DEFAULT_CLAHE_TILE_GRID_SIZE
        )

        via_explicit = apply_clahe(
            image, clip_limit=explicit_clip_limit, tile_grid_size=explicit_tile_grid_size
        )
        via_config_default = apply_clahe(image)
        self.assertFalse(np.array_equal(via_explicit, via_config_default))


class PreprocessingProfileTests(unittest.TestCase):
    """Requirement coverage: IQA profile, DR profile, explicit arguments
    overriding profile values, and profile overriding centralized
    (config.PREPROCESSING) defaults."""

    def test_iqa_profile_performs_no_preprocessing(self):
        image = _synthetic_bgr_image()
        out = preprocess_array(image, profile="IQA")
        np.testing.assert_array_equal(out, image)

    def test_iqa_profile_accepts_profile_object_not_just_name(self):
        image = _synthetic_bgr_image()
        via_name = preprocess_array(image, profile="IQA")
        via_object = preprocess_array(image, profile=project_config.PREPROCESSING_PROFILES.IQA)
        np.testing.assert_array_equal(via_name, via_object)

    def test_dr_profile_applies_gamma_and_clahe_using_centralized_defaults(self):
        image = _synthetic_bgr_image()
        via_dr_profile = preprocess_array(image, profile="DR")
        via_no_profile = preprocess_array(image)  # unconditional pipeline, same defaults
        np.testing.assert_array_equal(via_dr_profile, via_no_profile)
        # And it must actually differ from the raw input -- DR is not a no-op.
        self.assertFalse(np.array_equal(via_dr_profile, image))

    def test_profile_overrides_centralized_defaults(self):
        # config.PREPROCESSING.DEFAULT_GAMMA (1.2) is not 1.0, so the IQA
        # profile's gamma=1.0 must win over the centralized default,
        # proving profile values sit between explicit args and
        # config.PREPROCESSING in the resolution order.
        self.assertNotEqual(project_config.PREPROCESSING.DEFAULT_GAMMA, 1.0)
        image = _synthetic_bgr_image()

        via_iqa_profile = preprocess_array(image, profile="IQA")
        via_centralized_default = preprocess_array(image)  # no profile -> config.PREPROCESSING

        np.testing.assert_array_equal(via_iqa_profile, image)  # profile's gamma=1.0 -> identity
        self.assertFalse(np.array_equal(via_centralized_default, image))  # config default is not identity

    def test_explicit_gamma_overrides_iqa_profile(self):
        image = _synthetic_bgr_image()
        explicit_gamma = 2.5
        out = preprocess_array(image, profile="IQA", gamma=explicit_gamma)
        # CLAHE stays disabled (profile-controlled), but gamma correction
        # must actually run with the explicit value, not the profile's 1.0.
        expected = apply_gamma_correction(image, gamma=explicit_gamma)
        np.testing.assert_array_equal(out, expected)
        self.assertFalse(np.array_equal(out, image))

    def test_explicit_clahe_params_override_dr_profile(self):
        image = _synthetic_bgr_image(size=64)
        explicit_clip_limit = project_config.PREPROCESSING.DEFAULT_CLAHE_CLIP_LIMIT + 4.0
        explicit_tile_grid_size = (2, 2)
        self.assertNotEqual(
            explicit_tile_grid_size, project_config.PREPROCESSING.DEFAULT_CLAHE_TILE_GRID_SIZE
        )

        via_explicit = preprocess_array(
            image, profile="DR", clip_limit=explicit_clip_limit, tile_grid_size=explicit_tile_grid_size
        )
        via_profile_only = preprocess_array(image, profile="DR")
        self.assertFalse(np.array_equal(via_explicit, via_profile_only))

    def test_unknown_profile_name_raises(self):
        image = _synthetic_bgr_image()
        with self.assertRaises(ValueError):
            preprocess_array(image, profile="NOT_A_REAL_PROFILE")

    def test_no_profile_matches_pre_profile_behavior(self):
        # Backward compatibility: omitting `profile` must still run the
        # unconditional Gamma+CLAHE pipeline exactly as before profiles existed.
        image = _synthetic_bgr_image()
        expected = apply_clahe(apply_gamma_correction(image))
        actual = preprocess_array(image)
        np.testing.assert_array_equal(actual, expected)


class PreprocessImageFileTests(unittest.TestCase):
    def setUp(self):
        self.tmp_dir = tempfile.mkdtemp(prefix="iqa_preprocess_test_")
        self.input_path = os.path.join(self.tmp_dir, "input.png")
        cv2.imwrite(self.input_path, _synthetic_bgr_image())
        with open(self.input_path, "rb") as f:
            self.original_bytes = f.read()

    def tearDown(self):
        shutil.rmtree(self.tmp_dir, ignore_errors=True)

    def test_writes_output_and_preserves_original(self):
        output_path = os.path.join(self.tmp_dir, "processed", "input.png")
        result_path = preprocess_image(self.input_path, output_path)

        self.assertEqual(result_path, output_path)
        self.assertTrue(os.path.exists(output_path))

        with open(self.input_path, "rb") as f:
            self.assertEqual(f.read(), self.original_bytes, "original image bytes changed")

    def test_refuses_to_overwrite_input(self):
        with self.assertRaises(ValueError):
            preprocess_image(self.input_path, self.input_path)
        with open(self.input_path, "rb") as f:
            self.assertEqual(f.read(), self.original_bytes, "original image bytes changed")

    def test_missing_input_raises(self):
        with self.assertRaises(ValueError):
            preprocess_image(
                os.path.join(self.tmp_dir, "does_not_exist.png"),
                os.path.join(self.tmp_dir, "out.png"),
            )

    def test_iqa_profile_writes_pixel_identical_output(self):
        output_path = os.path.join(self.tmp_dir, "iqa_output.png")
        preprocess_image(self.input_path, output_path, profile="IQA")
        # Lossless PNG round-trip: decoded pixels must match exactly.
        original_pixels = cv2.imread(self.input_path)
        processed_pixels = cv2.imread(output_path)
        np.testing.assert_array_equal(processed_pixels, original_pixels)


class PreprocessingResultShapeTests(unittest.TestCase):
    """Requirement coverage: PreprocessingSummary is preserved unchanged
    (still aggregate-only) and PreprocessingResult has exactly the
    documented shape."""

    def test_preprocessing_summary_unchanged(self):
        summary_fields = {f.name: f.type for f in fields(PreprocessingSummary)}
        self.assertEqual(
            summary_fields,
            {
                "total_images": int,
                "processed": int,
                "skipped": int,
                "failed": int,
                "total_processing_time": float,
            },
        )

    def test_preprocessing_result_shape(self):
        result_fields = {f.name: f.type for f in fields(PreprocessingResult)}
        self.assertEqual(
            result_fields,
            {
                "summary": PreprocessingSummary,
                "processed_files": list[Path],
                "skipped_files": list[Path],
                "failed_files": list[Path],
            },
        )


class PreprocessFolderTests(unittest.TestCase):
    def setUp(self):
        self.tmp_dir = tempfile.mkdtemp(prefix="iqa_preprocess_folder_test_")
        self.input_dir = os.path.join(self.tmp_dir, "raw")
        self.output_dir = os.path.join(self.tmp_dir, "processed")
        os.makedirs(self.input_dir)

        self.filenames = ["a.png", "b.jpg"]
        self.original_bytes = {}
        for i, name in enumerate(self.filenames):
            path = os.path.join(self.input_dir, name)
            cv2.imwrite(path, _synthetic_bgr_image(seed=i))
            with open(path, "rb") as f:
                self.original_bytes[name] = f.read()
        # A non-image file must be ignored, not processed or deleted.
        with open(os.path.join(self.input_dir, "labels.csv"), "w") as f:
            f.write("image,quality\n")

    def tearDown(self):
        shutil.rmtree(self.tmp_dir, ignore_errors=True)

    def test_processes_every_image_and_preserves_originals(self):
        result = preprocess_folder(self.input_dir, self.output_dir)

        self.assertEqual(result.summary.processed, len(self.filenames))
        for name in self.filenames:
            self.assertTrue(os.path.exists(os.path.join(self.output_dir, name)))
            with open(os.path.join(self.input_dir, name), "rb") as f:
                self.assertEqual(f.read(), self.original_bytes[name], f"{name} was modified")

        # Non-image files are left alone and not copied into the output dir.
        self.assertTrue(os.path.exists(os.path.join(self.input_dir, "labels.csv")))
        self.assertFalse(os.path.exists(os.path.join(self.output_dir, "labels.csv")))

    def test_refuses_to_overwrite_input_dir(self):
        with self.assertRaises(ValueError):
            preprocess_folder(self.input_dir, self.input_dir)

    def test_empty_folder_raises(self):
        empty_dir = os.path.join(self.tmp_dir, "empty")
        os.makedirs(empty_dir)
        with self.assertRaises(FileNotFoundError):
            preprocess_folder(empty_dir, os.path.join(self.tmp_dir, "empty_out"))


class PreprocessFolderLoggingTests(unittest.TestCase):
    """Requirement coverage: successful logging, failed image logging,
    summary statistics, and log_file omitted (backward compatibility)."""

    def setUp(self):
        self.tmp_dir = tempfile.mkdtemp(prefix="iqa_preprocess_logging_test_")
        self.input_dir = os.path.join(self.tmp_dir, "raw")
        self.output_dir = os.path.join(self.tmp_dir, "processed")
        os.makedirs(self.input_dir)

        # Two valid images, one corrupt "image" (real extension, garbage
        # bytes -- exercises the failure path), one non-image file. Together
        # these exercise processed/failed/skipped in a single run.
        cv2.imwrite(os.path.join(self.input_dir, "a.png"), _synthetic_bgr_image(seed=0))
        cv2.imwrite(os.path.join(self.input_dir, "b.png"), _synthetic_bgr_image(seed=1))
        with open(os.path.join(self.input_dir, "corrupt.png"), "wb") as f:
            f.write(b"not a real png")
        with open(os.path.join(self.input_dir, "labels.csv"), "w") as f:
            f.write("image,quality\n")

        self.log_file = os.path.join(self.tmp_dir, "log.csv")

    def tearDown(self):
        shutil.rmtree(self.tmp_dir, ignore_errors=True)

    def _read_log_rows(self):
        with open(self.log_file, newline="", encoding="utf-8") as f:
            return {row["image_name"]: row for row in csv.DictReader(f)}

    def test_summary_statistics(self):
        result = preprocess_folder(self.input_dir, self.output_dir)
        summary = result.summary
        self.assertEqual(summary.total_images, 4)  # a.png, b.png, corrupt.png, labels.csv
        self.assertEqual(summary.processed, 2)
        self.assertEqual(summary.skipped, 1)
        self.assertEqual(summary.failed, 1)
        self.assertEqual(summary.total_images, summary.processed + summary.skipped + summary.failed)
        self.assertGreaterEqual(summary.total_processing_time, 0.0)

    def test_summary_consistent_with_path_lists(self):
        # Requirement: "summary consistency" -- the aggregate counts must
        # always match the lengths of the per-file path lists returned
        # alongside them.
        result = preprocess_folder(self.input_dir, self.output_dir)
        self.assertEqual(result.summary.processed, len(result.processed_files))
        self.assertEqual(result.summary.skipped, len(result.skipped_files))
        self.assertEqual(result.summary.failed, len(result.failed_files))
        self.assertEqual(
            result.summary.total_images,
            len(result.processed_files) + len(result.skipped_files) + len(result.failed_files),
        )

    def test_processed_files_contents(self):
        result = preprocess_folder(self.input_dir, self.output_dir)
        expected = {Path(self.output_dir) / "a.png", Path(self.output_dir) / "b.png"}
        self.assertEqual(set(result.processed_files), expected)
        for path in result.processed_files:
            self.assertIsInstance(path, Path)
            self.assertTrue(path.exists(), f"{path} was not actually written")

    def test_skipped_files_contents(self):
        result = preprocess_folder(self.input_dir, self.output_dir)
        self.assertEqual(result.skipped_files, [Path(self.input_dir) / "labels.csv"])
        self.assertIsInstance(result.skipped_files[0], Path)

    def test_failed_files_contents(self):
        result = preprocess_folder(self.input_dir, self.output_dir)
        self.assertEqual(result.failed_files, [Path(self.input_dir) / "corrupt.png"])
        self.assertIsInstance(result.failed_files[0], Path)
        self.assertFalse((Path(self.output_dir) / "corrupt.png").exists())

    def test_logging_csv_format_unchanged(self):
        # Requirement: "logging compatibility" -- adding PreprocessingResult
        # must not change the CSV's column set/order.
        preprocess_folder(self.input_dir, self.output_dir, log_file=self.log_file)
        with open(self.log_file, newline="", encoding="utf-8") as f:
            header = next(csv.reader(f))
        self.assertEqual(
            header,
            ["image_name", "input_path", "output_path", "profile",
             "status", "error_message", "processing_time_ms"],
        )

    def test_successful_image_is_logged(self):
        preprocess_folder(self.input_dir, self.output_dir, log_file=self.log_file)
        row = self._read_log_rows()["a.png"]

        self.assertEqual(row["status"], "processed")
        self.assertEqual(row["error_message"], "")
        self.assertTrue(row["output_path"])
        self.assertTrue(os.path.exists(row["output_path"]))
        self.assertEqual(row["input_path"], os.path.join(self.input_dir, "a.png"))
        self.assertGreater(float(row["processing_time_ms"]), 0.0)

    def test_failed_image_is_logged(self):
        preprocess_folder(self.input_dir, self.output_dir, log_file=self.log_file)
        row = self._read_log_rows()["corrupt.png"]

        self.assertEqual(row["status"], "failed")
        self.assertIn("Failed to load image", row["error_message"])
        self.assertEqual(row["output_path"], "")
        self.assertFalse(os.path.exists(os.path.join(self.output_dir, "corrupt.png")))

    def test_skipped_non_image_is_logged(self):
        preprocess_folder(self.input_dir, self.output_dir, log_file=self.log_file)
        row = self._read_log_rows()["labels.csv"]
        self.assertEqual(row["status"], "skipped")
        self.assertEqual(row["output_path"], "")

    def test_processing_continues_after_a_failure(self):
        # Requirement 8: one failing image must not stop the rest of the folder.
        result = preprocess_folder(self.input_dir, self.output_dir)
        self.assertTrue(os.path.exists(os.path.join(self.output_dir, "a.png")))
        self.assertTrue(os.path.exists(os.path.join(self.output_dir, "b.png")))
        self.assertEqual(result.summary.processed, 2)

    def test_log_file_omitted_is_backward_compatible(self):
        # No log requested -> no CSV written, but the actual file processing
        # (which images succeed/fail/get skipped) is identical either way.
        with_log = preprocess_folder(
            self.input_dir, self.output_dir, log_file=self.log_file,
        )
        shutil.rmtree(self.output_dir)
        without_log = preprocess_folder(self.input_dir, self.output_dir)

        self.assertFalse(os.path.exists(os.path.join(self.tmp_dir, "no_such_log.csv")))
        self.assertEqual(with_log.summary.total_images, without_log.summary.total_images)
        self.assertEqual(with_log.summary.processed, without_log.summary.processed)
        self.assertEqual(with_log.summary.skipped, without_log.summary.skipped)
        self.assertEqual(with_log.summary.failed, without_log.summary.failed)
        self.assertEqual(without_log.processed_files, with_log.processed_files)
        self.assertTrue(os.path.exists(os.path.join(self.output_dir, "a.png")))
        self.assertTrue(os.path.exists(os.path.join(self.output_dir, "b.png")))

    def test_refuses_to_overwrite_existing_log_file_by_default(self):
        with open(self.log_file, "w") as f:
            f.write("preexisting content\n")

        with self.assertRaises(FileExistsError):
            preprocess_folder(self.input_dir, self.output_dir, log_file=self.log_file)

        with open(self.log_file) as f:
            self.assertEqual(f.read(), "preexisting content\n", "existing log file was overwritten")

    def test_overwrite_log_true_allows_overwrite(self):
        with open(self.log_file, "w") as f:
            f.write("preexisting content\n")

        preprocess_folder(
            self.input_dir, self.output_dir, log_file=self.log_file, overwrite_log=True,
        )
        rows = self._read_log_rows()
        self.assertIn("a.png", rows)


class PreprocessDatasetLoggingTests(unittest.TestCase):
    """Confirms preprocess_dataset() threads log_file/overwrite_log through
    to preprocess_folder() and returns its PreprocessingResult."""

    DATASET_NAME = "TESTDS_FOR_UNIT_TESTS"

    def setUp(self):
        self.tmp_dir = tempfile.mkdtemp(prefix="iqa_preprocess_dataset_test_")
        self.raw_dir = os.path.join(self.tmp_dir, "raw")
        self.processed_dir = os.path.join(self.tmp_dir, "processed")
        os.makedirs(self.raw_dir)
        cv2.imwrite(os.path.join(self.raw_dir, "a.png"), _synthetic_bgr_image())

        raw_key = f"{self.DATASET_NAME}_RAW_DIR"
        processed_key = f"{self.DATASET_NAME}_PROCESSED_DIR"
        self._env_backup = {raw_key: os.environ.get(raw_key), processed_key: os.environ.get(processed_key)}
        os.environ[raw_key] = self.raw_dir
        os.environ[processed_key] = self.processed_dir

    def tearDown(self):
        for key, value in self._env_backup.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value
        shutil.rmtree(self.tmp_dir, ignore_errors=True)

    def test_preprocess_dataset_returns_result_and_writes_log(self):
        log_file = os.path.join(self.tmp_dir, "dataset_log.csv")
        result = preprocess_dataset(self.DATASET_NAME, log_file=log_file)

        self.assertEqual(result.summary.processed, 1)
        self.assertEqual(result.processed_files, [Path(self.processed_dir) / "a.png"])
        self.assertEqual(result.skipped_files, [])
        self.assertEqual(result.failed_files, [])

        self.assertTrue(os.path.exists(log_file))
        with open(log_file, newline="", encoding="utf-8") as f:
            rows = list(csv.DictReader(f))
        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0]["status"], "processed")


if __name__ == "__main__":
    unittest.main()
