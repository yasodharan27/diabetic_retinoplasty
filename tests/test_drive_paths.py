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


class ConfigureEnvironmentVariablesTests(unittest.TestCase):
    """The actual gap this task fixes: previously only EYEQ_RAW_DIR was
    set, leaving APTOS2019/IDRiD unresolvable in a fresh Colab session."""

    _ENV_KEYS = (
        "EYEQ_RAW_DIR", "EYEQ_PROCESSED_DIR",
        "APTOS2019_RAW_DIR", "APTOS2019_PROCESSED_DIR",
        "IDRID/GRADING_RAW_DIR", "IDRID/GRADING_PROCESSED_DIR",
        "IDRID/LOCALIZATION_RAW_DIR", "IDRID/LOCALIZATION_PROCESSED_DIR",
        "IDRID/SEGMENTATION_RAW_DIR", "IDRID/SEGMENTATION_PROCESSED_DIR",
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
