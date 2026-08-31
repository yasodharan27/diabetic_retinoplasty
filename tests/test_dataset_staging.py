"""
Tests for `colab/common/dataset_staging.py`'s `sync_missing_files()` -- the incremental,
direction-agnostic copy helper added to make Phase 1 joint-training cache precomputation
practical on a T4 (`JOINT_TRAINING_ARCHITECTURE.md` §33): writing/reading a per-image cache
directly against Google Drive's FUSE mount pays its severe per-file-open latency on every single
`os.path.exists`/write, so Phase 1 instead runs against a local cache directory and uses this
function to pull an existing Drive cache down first (preserving resumability) and push newly
written entries back up afterward.

`stage_dataset()`/`verify_staged_copy()` (this module's pre-existing functions) are not
re-tested here -- they are unmodified by this change and already exercised implicitly by every
Colab notebook that uses them; this file covers only the new function.

Uses only local temporary directories -- no Drive, no `google.colab` import, matching
`test_drive_paths.py`'s established pattern for testing `colab/common/` modules outside Colab.
"""

import os
import shutil
import sys
import tempfile
import unittest

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_COLAB_COMMON = os.path.join(_REPO_ROOT, "colab", "common")
if _COLAB_COMMON not in sys.path:
    sys.path.insert(0, _COLAB_COMMON)

import dataset_staging  # noqa: E402


class SyncMissingFilesTests(unittest.TestCase):
    def setUp(self):
        self.tmp_dir = tempfile.mkdtemp(prefix="sync_missing_files_test_")
        self.addCleanup(shutil.rmtree, self.tmp_dir, True)
        self.source_dir = os.path.join(self.tmp_dir, "source")
        self.dest_dir = os.path.join(self.tmp_dir, "dest")
        os.makedirs(self.source_dir)

    def _write(self, root, rel_path, content=b"data"):
        full_path = os.path.join(root, rel_path)
        os.makedirs(os.path.dirname(full_path), exist_ok=True)
        with open(full_path, "wb") as handle:
            handle.write(content)
        return full_path

    def test_copies_every_missing_file(self):
        self._write(self.source_dir, "a.npy")
        self._write(self.source_dir, "b.npy")
        copied, already_present = dataset_staging.sync_missing_files(self.source_dir, self.dest_dir)
        self.assertEqual(copied, 2)
        self.assertEqual(already_present, 0)
        self.assertTrue(os.path.exists(os.path.join(self.dest_dir, "a.npy")))
        self.assertTrue(os.path.exists(os.path.join(self.dest_dir, "b.npy")))

    def test_preserves_relative_subdirectory_structure(self):
        self._write(self.source_dir, os.path.join("sub", "c.npy"))
        dataset_staging.sync_missing_files(self.source_dir, self.dest_dir)
        self.assertTrue(os.path.exists(os.path.join(self.dest_dir, "sub", "c.npy")))

    def test_never_recopies_a_file_already_present_at_the_destination(self):
        """The core resumability guarantee -- an already-cached entry must never be re-copied or
        overwritten, matching the project's own cache convention."""
        self._write(self.source_dir, "a.npy", content=b"original")
        self._write(self.dest_dir, "a.npy", content=b"already-here-do-not-touch")

        copied, already_present = dataset_staging.sync_missing_files(self.source_dir, self.dest_dir)

        self.assertEqual(copied, 0)
        self.assertEqual(already_present, 1)
        with open(os.path.join(self.dest_dir, "a.npy"), "rb") as handle:
            self.assertEqual(handle.read(), b"already-here-do-not-touch")

    def test_partial_overlap_copies_only_what_is_missing(self):
        self._write(self.source_dir, "a.npy")
        self._write(self.source_dir, "b.npy")
        self._write(self.dest_dir, "a.npy")

        copied, already_present = dataset_staging.sync_missing_files(self.source_dir, self.dest_dir)

        self.assertEqual(copied, 1)
        self.assertEqual(already_present, 1)
        self.assertTrue(os.path.exists(os.path.join(self.dest_dir, "b.npy")))

    def test_nonexistent_source_dir_is_a_no_op_not_an_error(self):
        """A first-ever run has no prior Drive cache to pull down -- must not raise."""
        missing_source = os.path.join(self.tmp_dir, "does_not_exist")
        copied, already_present = dataset_staging.sync_missing_files(missing_source, self.dest_dir)
        self.assertEqual((copied, already_present), (0, 0))
        self.assertFalse(os.path.exists(self.dest_dir))

    def test_empty_source_dir_copies_nothing(self):
        copied, already_present = dataset_staging.sync_missing_files(self.source_dir, self.dest_dir)
        self.assertEqual((copied, already_present), (0, 0))

    def test_is_reusable_in_either_direction(self):
        """The same function pulls Drive->local and pushes local->Drive -- proven here by round-
        tripping through it twice with the roles of source/dest swapped."""
        self._write(self.source_dir, "a.npy", content=b"drive-copy")
        dataset_staging.sync_missing_files(self.source_dir, self.dest_dir)  # "pull"

        second_dir = os.path.join(self.tmp_dir, "second")
        dataset_staging.sync_missing_files(self.dest_dir, second_dir)  # "push" (dest is now source)

        with open(os.path.join(second_dir, "a.npy"), "rb") as handle:
            self.assertEqual(handle.read(), b"drive-copy")

    def test_source_is_never_modified(self):
        source_path = self._write(self.source_dir, "a.npy", content=b"original")
        dataset_staging.sync_missing_files(self.source_dir, self.dest_dir)
        with open(source_path, "rb") as handle:
            self.assertEqual(handle.read(), b"original")

    def test_returns_correct_counts_with_many_files(self):
        for i in range(20):
            self._write(self.source_dir, f"file_{i}.npy")
        copied, already_present = dataset_staging.sync_missing_files(self.source_dir, self.dest_dir, max_workers=4)
        self.assertEqual(copied, 20)
        self.assertEqual(already_present, 0)
        self.assertEqual(len(os.listdir(self.dest_dir)), 20)


if __name__ == "__main__":
    unittest.main()
