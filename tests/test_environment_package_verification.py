"""
Regression tests for colab/common/environment.py's requirements.txt ->
importable-module resolution (`get_missing_packages` and its helpers).

Root cause under test: PyPI *distribution* names (as spelled in
requirements.txt) do not always equal their *import* module name --
`scikit-image` imports as `skimage`, `opencv-python` as `cv2`,
`python-dotenv` as `dotenv`. The pre-fix implementation only knew about a
small hardcoded table of these and otherwise guessed `name.replace("-",
"_")`, which is wrong for `scikit-image` (guesses `scikit_image`, not
`skimage`) -- so a genuinely-installed `scikit-image` was reported as
missing. The fix resolves import names generically via
`importlib.metadata`, which reflects what's actually installed rather than
guessing from the distribution's own name, so it isn't limited to a fixed
list of known exceptions.

`colab/common` has no `__init__.py` (flat-import style, matching every
notebook's own `sys.path` bootstrap) -- inserted here the same way.
"""

import os
import sys
import tempfile
import unittest
from unittest import mock

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "colab", "common"))

import environment  # noqa: E402


class NormalizeDistributionNameTests(unittest.TestCase):
    def test_case_and_separator_insensitive(self):
        self.assertEqual(
            environment._normalize_distribution_name("scikit-image"),
            environment._normalize_distribution_name("Scikit_Image"),
        )
        self.assertEqual(environment._normalize_distribution_name("scikit_image"), "scikit-image")


class DistributionImportNamesResolutionTests(unittest.TestCase):
    """Exercises the exact reported bug scenario with a mocked
    `importlib.metadata` index, so this test is deterministic regardless of
    whether scikit-image happens to be installed in whatever environment
    runs the suite (Colab has it; a bare local dev venv may not)."""

    def test_scikit_image_resolves_to_skimage_via_packages_distributions_index(self):
        fake_index = {"scikit-image": {"skimage"}}
        result = environment._distribution_import_names("scikit-image", fake_index)
        self.assertEqual(result, {"skimage"})

    def test_resolution_is_case_and_separator_insensitive(self):
        fake_index = {"scikit-image": {"skimage"}}
        # requirements.txt could plausibly spell it any of these ways
        for spelling in ("scikit-image", "scikit_image", "Scikit-Image"):
            with self.subTest(spelling=spelling):
                self.assertEqual(
                    environment._distribution_import_names(spelling, fake_index), {"skimage"}
                )

    def test_falls_back_to_top_level_txt_when_not_in_index(self):
        fake_dist = mock.Mock()
        fake_dist.read_text.return_value = "skimage\n"
        with mock.patch.object(environment.importlib.metadata, "distribution", return_value=fake_dist):
            result = environment._distribution_import_names("scikit-image", {})
        self.assertEqual(result, {"skimage"})

    def test_empty_when_distribution_not_installed_anywhere(self):
        with mock.patch.object(
            environment.importlib.metadata, "distribution",
            side_effect=environment.importlib.metadata.PackageNotFoundError,
        ):
            result = environment._distribution_import_names("totally-nonexistent-package", {})
        self.assertEqual(result, set())


class ImportNameCandidatesTests(unittest.TestCase):
    def test_metadata_resolution_takes_priority_over_naive_guess(self):
        # If metadata resolution disagreed with the naive guess, the
        # metadata-derived name must still be tried first.
        candidates = environment._import_name_candidates("scikit-image", {"scikit-image": {"skimage"}})
        self.assertEqual(candidates[0], "skimage")

    def test_falls_back_to_override_table_when_metadata_finds_nothing(self):
        candidates = environment._import_name_candidates("opencv-python", {})
        self.assertIn("cv2", candidates)

    def test_falls_back_to_naive_guess_as_last_resort(self):
        candidates = environment._import_name_candidates("some-plain-package", {})
        self.assertIn("some_plain_package", candidates)


class GetMissingPackagesRegressionTests(unittest.TestCase):
    """End-to-end: writes a real temporary requirements.txt and runs the
    real (unmocked) `get_missing_packages` against this process's actual
    installed packages."""

    def _write_requirements(self, *lines):
        f = tempfile.NamedTemporaryFile("w", suffix=".txt", delete=False, newline="")
        try:
            f.write("\n".join(lines) + "\n")
        finally:
            f.close()
        self.addCleanup(os.unlink, f.name)
        return f.name

    def test_scikit_image_not_reported_missing_when_installed(self):
        # This is the exact bug report: `skimage import: OK` at runtime,
        # but `verify_required_packages` said scikit-image was missing.
        try:
            import skimage  # noqa: F401
        except ImportError:
            self.skipTest("scikit-image is not installed in this test environment")

        path = self._write_requirements("scikit-image>=0.19.0")
        missing = environment.get_missing_packages(path)
        self.assertEqual(missing, [])

    def test_genuinely_missing_package_is_still_reported(self):
        path = self._write_requirements("this-package-does-not-exist-anywhere-12345>=1.0.0")
        missing = environment.get_missing_packages(path)
        self.assertEqual(missing, ["this-package-does-not-exist-anywhere-12345"])

    def test_always_installed_dependency_with_mismatched_import_name(self):
        # python-dotenv -> dotenv: config.py itself does `from dotenv import
        # load_dotenv`, so this is installed in any environment capable of
        # running this project's existing test suite at all -- a real
        # (non-mocked) proof the fix works end-to-end, not just on
        # scikit-image specifically.
        path = self._write_requirements("python-dotenv>=0.19.0")
        missing = environment.get_missing_packages(path)
        self.assertEqual(missing, [])

    def test_comments_and_blank_lines_are_ignored(self):
        path = self._write_requirements(
            "# a comment", "", "python-dotenv>=0.19.0", "  ", "# scikit-image>=0.19.0 (commented out)",
        )
        missing = environment.get_missing_packages(path)
        self.assertEqual(missing, [])

    def test_version_specifier_stripped_from_reported_name(self):
        path = self._write_requirements("this-package-does-not-exist-12345==1.2.3")
        missing = environment.get_missing_packages(path)
        self.assertEqual(missing, ["this-package-does-not-exist-12345"])


if __name__ == "__main__":
    unittest.main()
