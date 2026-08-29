"""
Tests for the Colab/Drive dataset path infrastructure:
`colab/common/drive_paths.py`'s APTOS2019/IDRiD path resolution,
`colab/common/setup.py`'s environment-variable wiring (the gap this task
fixes -- previously only EYEQ_RAW_DIR was set), and
`colab/common/verify_dataset.py`'s explicit IDRiD-subset existence check.

`colab/common/` modules use bare, sys.path-relative imports (`import
drive_paths`, not `from colab.common import drive_paths`) by design -- see
`colab/common/setup.py`'s own docstring on why (they're only importable
after the repository is cloned/on sys.path in a real Colab session). This
test file inserts `colab/common` onto `sys.path` itself, the same way
`clone_or_update_repo()` does, so these modules can be imported and tested
outside Colab.

No `google.colab` import happens anywhere in this suite -- `drive_paths.py`,
`colab_config.py`, and the parts of `setup.py`/`verify_dataset.py` tested
here are all pure path logic / local filesystem checks by design.
"""

import os
import shutil
import sys
import tempfile
import unittest
from unittest import mock

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_COLAB_COMMON = os.path.join(_REPO_ROOT, "colab", "common")
if _COLAB_COMMON not in sys.path:
    sys.path.insert(0, _COLAB_COMMON)

import colab_config  # noqa: E402
import drive_paths  # noqa: E402
import setup as colab_setup  # noqa: E402
import verify_dataset  # noqa: E402


class DrivePathsResolutionTests(unittest.TestCase):
    """Pure path-construction checks -- no filesystem access, matches
    drive_paths.py's own "no filesystem access at import time" design."""

    def setUp(self):
        self.paths = drive_paths.build_drive_paths("/content/drive")

    def test_eyeq_raw_and_processed(self):
        self.assertEqual(
            self.paths.eyeq_raw_dir,
            "/content/drive/MyDrive/DiabeticRetinopathy/datasets/EyeQ/raw",
        )
        self.assertEqual(
            self.paths.eyeq_processed_dir,
            "/content/drive/MyDrive/DiabeticRetinopathy/datasets/EyeQ/processed",
        )

    def test_aptos2019_raw_and_processed(self):
        self.assertEqual(
            self.paths.aptos2019_raw_dir,
            "/content/drive/MyDrive/DiabeticRetinopathy/datasets/APTOS2019/raw",
        )
        self.assertEqual(
            self.paths.aptos2019_processed_dir,
            "/content/drive/MyDrive/DiabeticRetinopathy/datasets/APTOS2019/processed",
        )

    def test_idrid_subset_raw_and_processed(self):
        base = "/content/drive/MyDrive/DiabeticRetinopathy/datasets/IDRiD"
        self.assertEqual(self.paths.idrid_grading_raw_dir, f"{base}/grading/raw")
        self.assertEqual(self.paths.idrid_grading_processed_dir, f"{base}/grading/processed")
        self.assertEqual(self.paths.idrid_localization_raw_dir, f"{base}/localization/raw")
        self.assertEqual(self.paths.idrid_localization_processed_dir, f"{base}/localization/processed")
        self.assertEqual(self.paths.idrid_segmentation_raw_dir, f"{base}/segmentation/raw")
        self.assertEqual(self.paths.idrid_segmentation_processed_dir, f"{base}/segmentation/processed")

    def test_idrid_root_unchanged(self):
        self.assertEqual(
            self.paths.idrid_dataset_dir,
            "/content/drive/MyDrive/DiabeticRetinopathy/datasets/IDRiD",
        )

    def test_paths_are_posix_style_regardless_of_host_os(self):
        """drive_paths.py builds with posixpath deliberately -- Colab is
        always Linux -- so no path here should contain a backslash even
        when this test runs on Windows."""
        for value in (
            self.paths.aptos2019_raw_dir, self.paths.idrid_grading_raw_dir,
            self.paths.idrid_segmentation_processed_dir,
        ):
            self.assertNotIn("\\", value)

    def test_existing_top_level_structure_unchanged(self):
        """Guards against accidentally altering the already-verified,
        already-relied-upon top-level layout while extending it."""
        self.assertEqual(
            self.paths.project_root, "/content/drive/MyDrive/DiabeticRetinopathy",
        )
        self.assertEqual(
            self.paths.experiment_dir("FinalClassification"),
            "/content/drive/MyDrive/DiabeticRetinopathy/experiments/FinalClassification",
        )

    def test_cache_root_and_module_dirs(self):
        """cache/ is the new, minimal, additive Drive bucket for persistent
        per-image derived caches (Step 2/3 of the joint-training
        infrastructure correction) -- must sit alongside, not replace or
        rename, the four already-verified buckets."""
        self.assertEqual(self.paths.cache_root, "/content/drive/MyDrive/DiabeticRetinopathy/cache")
        self.assertEqual(
            self.paths.cache_dir("LocalFeatureExtraction"),
            "/content/drive/MyDrive/DiabeticRetinopathy/cache/LocalFeatureExtraction",
        )
        self.assertEqual(
            self.paths.cache_dir("RACAF"),
            "/content/drive/MyDrive/DiabeticRetinopathy/cache/RACAF",
        )

    def test_cache_dir_unknown_module_raises(self):
        with self.assertRaises(ValueError):
            self.paths.cache_dir("NotARealModule")

    def test_output_directories_includes_cache(self):
        outputs = drive_paths.output_directories(self.paths)
        self.assertIn(self.paths.cache_root, outputs)
        for cache_dir in self.paths.cache_dirs.values():
            self.assertIn(cache_dir, outputs)


class ColabConfigFlatConstantsTests(unittest.TestCase):
    """colab_config.py's new flat constants must exactly match the
    DrivePaths fields they're sourced from -- not a second, independently
    computed value that could silently drift."""

    def test_aptos2019_constants_match_drive_paths(self):
        self.assertEqual(colab_config.APTOS2019_RAW_DIR, colab_config.DRIVE.aptos2019_raw_dir)
        self.assertEqual(colab_config.APTOS2019_PROCESSED_DIR, colab_config.DRIVE.aptos2019_processed_dir)

    def test_idrid_subset_constants_match_drive_paths(self):
        self.assertEqual(colab_config.IDRID_GRADING_RAW_DIR, colab_config.DRIVE.idrid_grading_raw_dir)
        self.assertEqual(colab_config.IDRID_LOCALIZATION_RAW_DIR, colab_config.DRIVE.idrid_localization_raw_dir)
        self.assertEqual(colab_config.IDRID_SEGMENTATION_RAW_DIR, colab_config.DRIVE.idrid_segmentation_raw_dir)
        self.assertEqual(
            colab_config.IDRID_SEGMENTATION_PROCESSED_DIR, colab_config.DRIVE.idrid_segmentation_processed_dir,
        )

    def test_frozen_checkpoint_constants_match_drive_paths(self):
        """Stage 1/3/4's checkpoint dirs -- the actual gap this task fixes:
        previously these were not resolved to Drive at all anywhere in
        colab/common/."""
        self.assertEqual(colab_config.IQA_MODEL_DIR, colab_config.DRIVE.exported_model_dir("IQA"))
        self.assertEqual(
            colab_config.VESSEL_SEG_MODEL_DIR, colab_config.DRIVE.exported_model_dir("VesselSegmentation"),
        )
        self.assertEqual(
            colab_config.LESION_SEG_MODEL_DIR, colab_config.DRIVE.exported_model_dir("LesionSegmentation"),
        )

    def test_cache_constants_match_drive_paths(self):
        self.assertEqual(colab_config.LOCAL_FEATURE_CACHE_DIR, colab_config.DRIVE.cache_dir("LocalFeatureExtraction"))
        self.assertEqual(colab_config.RACAF_CACHE_DIR, colab_config.DRIVE.cache_dir("RACAF"))

    def test_final_classification_stage_model_dirs_are_nested_under_the_reserved_bucket(self):
        """Stage 05-08+RACAF train jointly under the single, already-reserved
        'FinalClassification' module -- their five model dirs must each be a
        subdirectory of it, not five new top-level PIPELINE_MODULES entries."""
        base = colab_config.DRIVE.exported_model_dir("FinalClassification")
        for constant in (
            colab_config.LOCAL_FEATURE_MODEL_DIR, colab_config.GLOBAL_FEATURE_MODEL_DIR,
            colab_config.FEATURE_FUSION_MODEL_DIR, colab_config.RACAF_MODEL_DIR, colab_config.CORN_MODEL_DIR,
        ):
            self.assertTrue(constant.startswith(base + "/"), f"{constant} is not nested under {base}")
        # And each stage gets its own, distinct subdirectory.
        stage_dirs = {
            colab_config.LOCAL_FEATURE_MODEL_DIR, colab_config.GLOBAL_FEATURE_MODEL_DIR,
            colab_config.FEATURE_FUSION_MODEL_DIR, colab_config.RACAF_MODEL_DIR, colab_config.CORN_MODEL_DIR,
        }
        self.assertEqual(len(stage_dirs), 5)


class ConfigureEnvironmentVariablesTests(unittest.TestCase):
    """The actual gap this task fixes: previously only EYEQ_RAW_DIR was
    set, leaving APTOS2019/IDRiD unresolvable in a fresh Colab session."""

    _ENV_KEYS = (
        "EYEQ_RAW_DIR", "EYEQ_PROCESSED_DIR",
        "APTOS2019_RAW_DIR", "APTOS2019_PROCESSED_DIR",
        "IDRID/GRADING_RAW_DIR", "IDRID/GRADING_PROCESSED_DIR",
        "IDRID/LOCALIZATION_RAW_DIR", "IDRID/LOCALIZATION_PROCESSED_DIR",
        "IDRID/SEGMENTATION_RAW_DIR", "IDRID/SEGMENTATION_PROCESSED_DIR",
        "IQA_MODEL_DIR", "VESSEL_SEG_MODEL_DIR", "LESION_SEG_MODEL_DIR",
        "LOCAL_FEATURE_RESULTS_DIR", "RACAF_RESULTS_DIR",
        "LOCAL_FEATURE_MODEL_DIR", "GLOBAL_FEATURE_MODEL_DIR",
        "FEATURE_FUSION_MODEL_DIR", "RACAF_MODEL_DIR", "CORN_MODEL_DIR",
    )

    def setUp(self):
        self._backup = {key: os.environ.get(key) for key in self._ENV_KEYS}
        self.addCleanup(self._restore)

    def _restore(self):
        for key, value in self._backup.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value

    def test_sets_every_expected_key(self):
        env_vars = colab_setup.configure_environment_variables()
        for key in self._ENV_KEYS:
            self.assertIn(key, env_vars, f"{key} missing from configure_environment_variables() result")
            self.assertIn(key, os.environ, f"{key} not actually set in os.environ")

    def test_aptos2019_env_vars_resolve_config_dataset_raw_dir(self):
        """The actual, end-to-end proof this gap is fixed: after
        configure_environment_variables() runs, config.py's generic
        dataset_raw_dir("APTOS2019") must resolve to the Drive path, not
        the cloned repo's own empty datasets/APTOS2019/raw."""
        colab_setup.configure_environment_variables()
        import importlib
        import config as config_module
        importlib.reload(config_module)
        try:
            self.assertEqual(
                config_module.dataset_raw_dir("APTOS2019"), colab_config.APTOS2019_RAW_DIR,
            )
        finally:
            importlib.reload(config_module)

    def test_idrid_segmentation_env_var_resolves_config_dataset_raw_dir(self):
        """Same proof for IDRiD's compound "IDRiD/segmentation" name --
        the exact one lesion_segmentation_dataset.py (frozen, Stage 04)
        already depends on."""
        colab_setup.configure_environment_variables()
        import importlib
        import config as config_module
        importlib.reload(config_module)
        try:
            self.assertEqual(
                config_module.dataset_raw_dir("IDRiD/segmentation"),
                colab_config.IDRID_SEGMENTATION_RAW_DIR,
            )
        finally:
            importlib.reload(config_module)

    def test_idrid_root_env_var_not_fabricated(self):
        """IDRiD's bare root has no raw/processed of its own (verified
        locally) -- this must not be silently invented."""
        env_vars = colab_setup.configure_environment_variables()
        self.assertNotIn("IDRID_RAW_DIR", env_vars)
        self.assertNotIn("IDRID_PROCESSED_DIR", env_vars)

    def test_stages_with_no_frozen_upstream_cache_are_not_fabricated(self):
        """GLOBAL_FEATURE_RESULTS_DIR / FEATURE_FUSION_RESULTS_DIR /
        CORN_RESULTS_DIR are never written to by any of those three
        stages (config.py's own docstrings) -- must not be wired here."""
        env_vars = colab_setup.configure_environment_variables()
        for key in ("GLOBAL_FEATURE_RESULTS_DIR", "FEATURE_FUSION_RESULTS_DIR", "CORN_RESULTS_DIR"):
            self.assertNotIn(key, env_vars)

    def test_frozen_checkpoint_env_vars_resolve_config_model_dirs(self):
        """The actual proof Stage 1/3/4's checkpoints resolve to Drive, not
        the ephemeral cloned-repo checkout, in a fresh Colab session."""
        colab_setup.configure_environment_variables()
        import importlib
        import config as config_module
        importlib.reload(config_module)
        try:
            self.assertEqual(config_module.IQA_MODEL_DIR, colab_config.IQA_MODEL_DIR)
            self.assertEqual(config_module.VESSEL_SEG_MODEL_DIR, colab_config.VESSEL_SEG_MODEL_DIR)
            self.assertEqual(config_module.LESION_SEG_MODEL_DIR, colab_config.LESION_SEG_MODEL_DIR)
        finally:
            importlib.reload(config_module)

    def test_cache_and_final_classification_env_vars_resolve_config_dirs(self):
        colab_setup.configure_environment_variables()
        import importlib
        import config as config_module
        importlib.reload(config_module)
        try:
            self.assertEqual(config_module.LOCAL_FEATURE_RESULTS_DIR, colab_config.LOCAL_FEATURE_CACHE_DIR)
            self.assertEqual(config_module.RACAF_RESULTS_DIR, colab_config.RACAF_CACHE_DIR)
            self.assertEqual(config_module.LOCAL_FEATURE_MODEL_DIR, colab_config.LOCAL_FEATURE_MODEL_DIR)
            self.assertEqual(config_module.GLOBAL_FEATURE_MODEL_DIR, colab_config.GLOBAL_FEATURE_MODEL_DIR)
            self.assertEqual(config_module.FEATURE_FUSION_MODEL_DIR, colab_config.FEATURE_FUSION_MODEL_DIR)
            self.assertEqual(config_module.RACAF_MODEL_DIR, colab_config.RACAF_MODEL_DIR)
            self.assertEqual(config_module.CORN_MODEL_DIR, colab_config.CORN_MODEL_DIR)
        finally:
            importlib.reload(config_module)


class VerifyIdridDatasetDirTests(unittest.TestCase):
    def setUp(self):
        self.tmp_dir = tempfile.mkdtemp(prefix="idrid_verify_")
        self.addCleanup(shutil.rmtree, self.tmp_dir, True)

    def _make_subset(self, root, name):
        os.makedirs(os.path.join(root, name, "raw"), exist_ok=True)

    def test_all_three_subsets_present_passes(self):
        for subset in verify_dataset.IDRID_SUBSETS:
            self._make_subset(self.tmp_dir, subset)
        result = verify_dataset.verify_idrid_dataset_dir(self.tmp_dir)
        self.assertEqual(result, self.tmp_dir)

    def test_missing_root_raises(self):
        with self.assertRaises(RuntimeError):
            verify_dataset.verify_idrid_dataset_dir(os.path.join(self.tmp_dir, "does_not_exist"))

    def test_missing_one_subset_raises_and_names_it(self):
        self._make_subset(self.tmp_dir, "grading")
        self._make_subset(self.tmp_dir, "segmentation")
        # "localization" deliberately missing
        with self.assertRaises(RuntimeError) as ctx:
            verify_dataset.verify_idrid_dataset_dir(self.tmp_dir)
        self.assertIn("localization", str(ctx.exception))

    def test_missing_all_subsets_names_all_three(self):
        os.makedirs(self.tmp_dir, exist_ok=True)
        with self.assertRaises(RuntimeError) as ctx:
            verify_dataset.verify_idrid_dataset_dir(self.tmp_dir)
        message = str(ctx.exception)
        for subset in verify_dataset.IDRID_SUBSETS:
            self.assertIn(subset, message)

    def test_real_local_idrid_dataset_passes(self):
        """The one real-data check in this file: this project's actual
        local datasets/IDRiD/ directory (not Drive -- this environment
        cannot reach Drive) already has all three subsets, per this
        session's own repository audit."""
        local_idrid_root = os.path.join(_REPO_ROOT, "datasets", "IDRiD")
        if not os.path.isdir(local_idrid_root):
            self.skipTest("Local datasets/IDRiD/ not present in this environment.")
        verify_dataset.verify_idrid_dataset_dir(local_idrid_root)


if __name__ == "__main__":
    unittest.main()
