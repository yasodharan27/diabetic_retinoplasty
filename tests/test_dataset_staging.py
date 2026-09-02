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

import errno
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
        copied, already_present, failures = dataset_staging.sync_missing_files(self.source_dir, self.dest_dir)
        self.assertEqual(copied, 2)
        self.assertEqual(already_present, 0)
        self.assertEqual(failures, [])
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

        copied, already_present, failures = dataset_staging.sync_missing_files(self.source_dir, self.dest_dir)

        self.assertEqual(copied, 0)
        self.assertEqual(already_present, 1)
        self.assertEqual(failures, [])
        with open(os.path.join(self.dest_dir, "a.npy"), "rb") as handle:
            self.assertEqual(handle.read(), b"already-here-do-not-touch")

    def test_partial_overlap_copies_only_what_is_missing(self):
        self._write(self.source_dir, "a.npy")
        self._write(self.source_dir, "b.npy")
        self._write(self.dest_dir, "a.npy")

        copied, already_present, failures = dataset_staging.sync_missing_files(self.source_dir, self.dest_dir)

        self.assertEqual(copied, 1)
        self.assertEqual(already_present, 1)
        self.assertEqual(failures, [])
        self.assertTrue(os.path.exists(os.path.join(self.dest_dir, "b.npy")))

    def test_nonexistent_source_dir_is_a_no_op_not_an_error(self):
        """A first-ever run has no prior Drive cache to pull down -- must not raise."""
        missing_source = os.path.join(self.tmp_dir, "does_not_exist")
        copied, already_present, failures = dataset_staging.sync_missing_files(missing_source, self.dest_dir)
        self.assertEqual((copied, already_present, failures), (0, 0, []))
        self.assertFalse(os.path.exists(self.dest_dir))

    def test_empty_source_dir_copies_nothing(self):
        copied, already_present, failures = dataset_staging.sync_missing_files(self.source_dir, self.dest_dir)
        self.assertEqual((copied, already_present, failures), (0, 0, []))

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
        copied, already_present, failures = dataset_staging.sync_missing_files(
            self.source_dir, self.dest_dir, max_workers=4,
        )
        self.assertEqual(copied, 20)
        self.assertEqual(already_present, 0)
        self.assertEqual(failures, [])
        self.assertEqual(len(os.listdir(self.dest_dir)), 20)

    # --- Hardening regression tests (JOINT_TRAINING_ARCHITECTURE.md Sec 35): a real Colab run hit
    # OSError: [Errno 107] Transport endpoint is not connected mid-copy against Drive's FUSE mount.
    # These verify _copy_one()'s atomic-write + transient-retry behavior directly, without any
    # real Drive mount, by simulating the same class of error via a patched shutil.copy2. ---

    def test_copies_are_atomic_no_partial_file_survives_a_non_transient_failure(self):
        """A failed copy must never leave a partial/renamed file at the destination path, and
        must never leave a stray .tmp- file behind either -- a half-written array would otherwise
        be silently treated as a valid cache entry by every os.path.exists check in this project."""
        self._write(self.source_dir, "a.npy", content=b"data")
        dest_path = os.path.join(self.dest_dir, "a.npy")

        with mock.patch("dataset_staging.shutil.copy2", side_effect=PermissionError("denied")):
            copied, already_present, failures = dataset_staging.sync_missing_files(self.source_dir, self.dest_dir)

        self.assertEqual(copied, 0)
        self.assertEqual(len(failures), 1)
        self.assertFalse(os.path.exists(dest_path))
        leftover = [n for n in os.listdir(self.dest_dir)] if os.path.isdir(self.dest_dir) else []
        self.assertEqual([n for n in leftover if ".tmp-" in n], [])

    def test_transient_errno_107_is_retried_and_eventually_succeeds(self):
        """The exact real-world failure (ENOTCONN / 'Transport endpoint is not connected') must be
        retried internally -- a caller should never see it for a transient Drive FUSE hiccup that
        clears up within a few attempts."""
        self._write(self.source_dir, "a.npy", content=b"data")
        real_copy2 = shutil.copy2
        call_count = {"n": 0}

        def flaky_copy2(src, dst):
            call_count["n"] += 1
            if call_count["n"] < 3:
                raise OSError(errno.ENOTCONN, "Transport endpoint is not connected")
            return real_copy2(src, dst)

        with mock.patch("dataset_staging.shutil.copy2", side_effect=flaky_copy2):
            copied, already_present, failures = dataset_staging.sync_missing_files(
                self.source_dir, self.dest_dir, max_workers=1,
            )

        self.assertEqual(copied, 1)
        self.assertEqual(failures, [])
        self.assertGreaterEqual(call_count["n"], 3)
        with open(os.path.join(self.dest_dir, "a.npy"), "rb") as handle:
            self.assertEqual(handle.read(), b"data")

    def test_non_transient_error_fails_without_retrying(self):
        """A real, non-retryable error (e.g. permission denied) must not waste time/retries --
        it should fail on the first attempt."""
        self._write(self.source_dir, "a.npy")
        call_count = {"n": 0}

        def always_denied(src, dst):
            call_count["n"] += 1
            raise PermissionError("denied")

        with mock.patch("dataset_staging.shutil.copy2", side_effect=always_denied):
            copied, already_present, failures = dataset_staging.sync_missing_files(self.source_dir, self.dest_dir)

        self.assertEqual(call_count["n"], 1)
        self.assertEqual(len(failures), 1)

    def test_one_file_failure_does_not_abort_or_block_other_files(self):
        """The bug this regression guards against: sync_missing_files used to re-raise (and abort
        the whole call, including files other threads had already started) on the FIRST failure.
        One persistently-failing file must not prevent every other independent file from being
        copied successfully."""
        self._write(self.source_dir, "bad.npy", content=b"x")
        self._write(self.source_dir, "good_1.npy", content=b"y")
        self._write(self.source_dir, "good_2.npy", content=b"z")
        real_copy2 = shutil.copy2

        def selectively_failing_copy2(src, dst):
            if "bad.npy" in src:
                raise PermissionError("denied")
            return real_copy2(src, dst)

        with mock.patch("dataset_staging.shutil.copy2", side_effect=selectively_failing_copy2):
            copied, already_present, failures = dataset_staging.sync_missing_files(
                self.source_dir, self.dest_dir, max_workers=1,
            )

        self.assertEqual(copied, 2)
        self.assertEqual(len(failures), 1)
        self.assertIn("bad.npy", failures[0][0])
        self.assertTrue(os.path.exists(os.path.join(self.dest_dir, "good_1.npy")))
        self.assertTrue(os.path.exists(os.path.join(self.dest_dir, "good_2.npy")))
        self.assertFalse(os.path.exists(os.path.join(self.dest_dir, "bad.npy")))

    def test_failed_file_is_retried_by_a_later_call_resumability(self):
        """A file that failed on one call must still be missing from dest afterward, so a later
        re-run of sync_missing_files (the project's existing resumability convention) picks it up
        again automatically -- no special "resume" handling is needed beyond calling it again."""
        self._write(self.source_dir, "a.npy", content=b"data")

        with mock.patch("dataset_staging.shutil.copy2", side_effect=PermissionError("denied")):
            dataset_staging.sync_missing_files(self.source_dir, self.dest_dir)

        self.assertFalse(os.path.exists(os.path.join(self.dest_dir, "a.npy")))

        copied, already_present, failures = dataset_staging.sync_missing_files(self.source_dir, self.dest_dir)
        self.assertEqual(copied, 1)
        self.assertEqual(failures, [])
        self.assertTrue(os.path.exists(os.path.join(self.dest_dir, "a.npy")))


if __name__ == "__main__":
    unittest.main()
