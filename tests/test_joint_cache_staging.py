"""
Tests for `joint_cache_staging.py` -- the one-time persistent-to-local cache mirror.

Synthetic/temporary data only, per `PROJECT_CODE.md`'s Implementation Rules. The "persistent"
cache is a second temp directory, which is exactly what `persistent_cache_dir` is: a directory the
pipeline stats and reads. No Drive, no real dataset, no real checkpoint, no training.

The properties that matter here are safety properties, so they are tested as such: the persistent
side must come back byte-identical and mtime-identical from every operation, a mount failure must
stop the mirror rather than being mistaken for absence, a partially written file must never appear
under a real cache filename, and nothing may ever be recomputed.
"""

import csv
import errno
import os
import shutil
import sys
import tempfile
import unittest
from unittest import mock

import numpy as np

import joint_cache_diagnostics as jcd
import joint_cache_staging as jcs
import joint_training_dataset as jtd
import local_feature_extraction_dataset as lfed
import racaf

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

SIZE = jtd.STAGE5_IMAGE_SIZE


def _write_entry(cache_dir, racaf_cache_dir, id_code, image_size=SIZE, seed=0):
    """Writes one complete, VALID set of four cache artifacts, with the real shapes the pipeline
    requires, using the real path builders."""
    os.makedirs(cache_dir, exist_ok=True)
    os.makedirs(racaf_cache_dir, exist_ok=True)
    rng = np.random.RandomState(seed)
    np.save(lfed._cache_path(cache_dir, id_code, "vessel", image_size),
            rng.rand(*image_size, 1).astype(np.float32))
    np.save(lfed._cache_path(cache_dir, id_code, "lesion", image_size),
            rng.rand(*image_size, 4).astype(np.float32))
    np.save(jtd._canonical_rgb_cache_path(id_code, cache_dir, image_size),
            rng.rand(*image_size, 3).astype(np.float32))
    np.savez(racaf.reliability_cache_path(racaf_cache_dir, id_code),
             kappa=rng.rand(4).astype(np.float32), r=np.float32(0.5))


class _Tree:
    def __init__(self, ids):
        self.root = tempfile.mkdtemp(prefix="jcs_")
        self.local = os.path.join(self.root, "content", "cache", "local_feature_extraction")
        self.local_racaf = os.path.join(self.root, "content", "cache", "racaf")
        self.drive = os.path.join(self.root, "drive", "LocalFeatureExtraction")
        self.drive_racaf = os.path.join(self.root, "drive", "RACAF")
        for directory in (self.local, self.local_racaf, self.drive, self.drive_racaf):
            os.makedirs(directory, exist_ok=True)
        self.entries = [(id_code, i % 5) for i, id_code in enumerate(ids)]

    def cleanup(self):
        shutil.rmtree(self.root, ignore_errors=True)


class _StagingTestBase(unittest.TestCase):
    IDS = ["img_a", "img_b", "img_c"]

    def setUp(self):
        self.tree = _Tree(self.IDS)
        self.addCleanup(self.tree.cleanup)

    def populate_drive(self, ids=None):
        for i, id_code in enumerate(ids if ids is not None else self.IDS):
            _write_entry(self.tree.drive, self.tree.drive_racaf, id_code, seed=i)

    def snapshot_drive(self):
        state = {}
        for directory in (self.tree.drive, self.tree.drive_racaf):
            for name in sorted(os.listdir(directory)):
                path = os.path.join(directory, name)
                with open(path, "rb") as handle:
                    state[path] = (handle.read(), os.stat(path).st_mtime_ns)
        return state

    def plan(self, **overrides):
        kwargs = dict(entries=self.tree.entries, cache_dir=self.tree.local,
                      racaf_cache_dir=self.tree.local_racaf,
                      persistent_cache_dir=self.tree.drive,
                      persistent_racaf_cache_dir=self.tree.drive_racaf)
        kwargs.update(overrides)
        return jcs.plan_mirror(**kwargs)

    def mirror(self, **overrides):
        kwargs = dict(entries=self.tree.entries, cache_dir=self.tree.local,
                      racaf_cache_dir=self.tree.local_racaf,
                      persistent_cache_dir=self.tree.drive,
                      persistent_racaf_cache_dir=self.tree.drive_racaf,
                      progress_every=0)
        kwargs.update(overrides)
        return jcs.mirror_persistent_cache_to_local(**kwargs)

    def verify(self, **overrides):
        kwargs = dict(entries=self.tree.entries, cache_dir=self.tree.local,
                      racaf_cache_dir=self.tree.local_racaf,
                      persistent_cache_dir=self.tree.drive,
                      persistent_racaf_cache_dir=self.tree.drive_racaf)
        kwargs.update(overrides)
        return jcs.verify_local_cache(**kwargs)


# =====================================================================
# Planning -- measure before copying
# =====================================================================

class PlanTests(_StagingTestBase):
    def test_plan_measures_real_source_sizes_not_an_assumed_per_artifact_size(self):
        self.populate_drive()
        plan = self.plan()
        self.assertEqual(len(plan["to_copy"]), 4 * len(self.IDS))
        actual = 0
        for directory in (self.tree.drive, self.tree.drive_racaf):
            for name in os.listdir(directory):
                actual += os.stat(os.path.join(directory, name)).st_size
        self.assertEqual(plan["bytes_to_copy"], actual)

    def test_plan_counts_files_and_bytes_per_artifact(self):
        self.populate_drive()
        plan = self.plan()
        for artifact in jcd.ARTIFACTS:
            self.assertEqual(plan["files_by_artifact"][artifact], len(self.IDS), artifact)
            self.assertGreater(plan["bytes_by_artifact"][artifact], 0, artifact)

    def test_an_already_local_entry_is_not_planned_for_copying(self):
        self.populate_drive()
        _write_entry(self.tree.local, self.tree.local_racaf, "img_a", seed=0)
        plan = self.plan()
        self.assertEqual(plan["already_local_entries"], 1)
        self.assertEqual(len(plan["to_copy"]), 4 * (len(self.IDS) - 1))
        self.assertFalse(any(item[0] == "img_a" for item in plan["to_copy"]))

    def test_entries_absent_from_drive_are_reported_never_planned(self):
        self.populate_drive(ids=["img_a"])
        plan = self.plan()
        missing_ids = [id_code for id_code, _ in plan["missing_everywhere"]]
        self.assertEqual(sorted(missing_ids), ["img_b", "img_c"])
        self.assertEqual(len(plan["to_copy"]), 4)

    def test_plan_reports_free_space_and_whether_it_fits(self):
        self.populate_drive()
        plan = self.plan()
        self.assertGreater(plan["local_free_bytes"], 0)
        self.assertEqual(plan["required_bytes_with_margin"],
                         int(plan["bytes_to_copy"] * jcs.FREE_SPACE_SAFETY_FACTOR)
                         + jcs.FREE_SPACE_MARGIN_BYTES)
        self.assertTrue(plan["fits"])

    def test_planning_performs_no_per_file_drive_stat(self):
        """The metadata-cost fix (§43): planning must cost two directory listings, not one Drive
        stat per file. A real run measured ~1.0 s per Drive file operation, so per-file stat-ing
        14,595 sources cost hours before the first byte moved."""
        self.populate_drive()
        drive_root = os.path.abspath(self.tree.root)
        stats = {"n": 0}
        real_stat = os.stat

        def counting_stat(path, *args, **kwargs):
            if os.path.abspath(str(path)).startswith(os.path.abspath(self.tree.drive)):
                stats["n"] += 1
            return real_stat(path, *args, **kwargs)

        with mock.patch("os.stat", side_effect=counting_stat):
            plan = self.plan()
        self.assertEqual(len(plan["to_copy"]), 4 * len(self.IDS))
        self.assertEqual(stats["n"], 0, "planning still stats individual persistent files")

    def test_plan_stops_at_the_first_mount_failure_instead_of_scanning_on(self):
        """Planning now reads directory listings, so that is the operation a dead mount fails on.
        The property under test is unchanged: it must stop, not read the failure as 'absent'."""
        self.populate_drive()
        calls = {"n": 0}
        real_listdir = os.listdir

        def flaky(path, *args, **kwargs):
            if os.path.abspath(str(path)).startswith(os.path.abspath(self.tree.root))                     and "drive" in str(path).lower():
                calls["n"] += 1
                raise OSError(errno.ENOTCONN, "Transport endpoint is not connected")
            return real_listdir(path, *args, **kwargs)

        with mock.patch("os.listdir", side_effect=flaky):
            with self.assertRaises(jtd.PersistentCacheUnavailableError):
                self.plan()
        self.assertLessEqual(calls["n"], 2, "kept probing a mount that had already failed")


# =====================================================================
# Copying -- atomicity, validation, and never recomputing
# =====================================================================

class MirrorTests(_StagingTestBase):
    def test_a_full_mirror_makes_every_entry_fully_local(self):
        self.populate_drive()
        result = self.mirror()
        self.assertEqual(result["copied"], 4 * len(self.IDS))
        self.assertEqual(result["corrupt"], [])
        self.assertFalse(result["drive_unreachable"])
        verification = self.verify()
        self.assertEqual(verification["fully_local"], len(self.IDS))
        self.assertEqual(verification["drive_fallback_required"], 0)
        self.assertEqual(verification["missing_everywhere"], 0)
        self.assertEqual(verification["corrupt_local"], [])

    def test_copied_files_are_byte_identical_to_their_persistent_originals(self):
        self.populate_drive()
        self.mirror()
        for id_code in self.IDS:
            local = jcd.artifact_paths(id_code, self.tree.local, self.tree.local_racaf, SIZE)
            drive = jcd.artifact_paths(id_code, self.tree.drive, self.tree.drive_racaf, SIZE)
            for artifact in jcd.ARTIFACTS:
                with open(local[artifact], "rb") as handle:
                    local_bytes = handle.read()
                with open(drive[artifact], "rb") as handle:
                    drive_bytes = handle.read()
                self.assertEqual(local_bytes, drive_bytes, f"{id_code}/{artifact}")

    def test_the_persistent_cache_is_byte_and_mtime_identical_afterwards(self):
        self.populate_drive()
        before = self.snapshot_drive()
        self.mirror()
        self.assertEqual(before, self.snapshot_drive())

    def test_the_mirror_never_loads_a_model_so_recomputation_is_impossible(self):
        """There is no model in this module at all -- assert the frozen entry points are never
        even referenced, which is a stronger guarantee than counting calls."""
        self.populate_drive()
        with mock.patch("joint_training_dataset.predict_vessel_mask") as vessel_spy, \
             mock.patch.object(jtd.racaf, "tta_views") as lesion_spy, \
             mock.patch.object(jtd, "_resize_rgb_01") as rgb_spy:
            self.mirror()
        self.assertEqual(vessel_spy.call_count, 0)
        self.assertEqual(lesion_spy.call_count, 0)
        self.assertEqual(rgb_spy.call_count, 0)

    def test_mirroring_is_idempotent_and_resumable(self):
        self.populate_drive()
        first = self.mirror()
        second = self.mirror()
        self.assertEqual(first["copied"], 4 * len(self.IDS))
        self.assertEqual(second["copied"], 0)
        self.assertEqual(len(second["plan"]["to_copy"]), 0)
        self.assertEqual(second["plan"]["already_local_entries"], len(self.IDS))

    def test_a_partial_copy_never_appears_under_a_real_cache_filename(self):
        """The atomicity property: a copy that dies mid-write must leave the final path absent,
        not a truncated file that later reads as a valid cache hit."""
        self.populate_drive()
        real_copyfile = shutil.copyfile

        def dying_copy(source, destination, *args, **kwargs):
            real_copyfile(source, destination)
            with open(destination, "r+b") as handle:  # truncate, as an interrupted write would
                handle.truncate(64)
            # EACCES deliberately: it is NOT in _TRANSIENT_FUSE_ERRNOS, so it propagates as a
            # plain OSError rather than being classified as a mount outage (that path is covered
            # by test_a_mount_failure_mid_copy_stops_immediately_and_reports). What is under test
            # here is atomicity, which must hold for either kind of failure.
            raise OSError(errno.EACCES, "died mid-copy")

        with mock.patch.object(jcs.shutil, "copyfile", side_effect=dying_copy):
            with self.assertRaises(OSError):
                jcs._copy_and_validate(
                    lfed._cache_path(self.tree.drive, "img_a", "vessel", SIZE),
                    lfed._cache_path(self.tree.local, "img_a", "vessel", SIZE),
                    "vessel", "img_a", SIZE)
        final = lfed._cache_path(self.tree.local, "img_a", "vessel", SIZE)
        self.assertFalse(os.path.exists(final), "a partial file was left at the real filename")
        leftovers = [f for f in os.listdir(self.tree.local) if "tmp" in f]
        self.assertEqual(leftovers, [], f"temp files left behind: {leftovers}")

    def test_a_truncated_source_is_rejected_and_not_staged(self):
        """Reproduces the reported shape failure: a persistent file whose array is the wrong
        shape must be refused, left untouched, and must not appear locally."""
        self.populate_drive()
        lesion = lfed._cache_path(self.tree.drive, "img_a", "lesion", SIZE)
        np.save(lesion, np.zeros((8, 8, 4), dtype=np.float32))
        with open(lesion, "rb") as handle:
            before = handle.read()

        result = self.mirror()
        self.assertEqual(len(result["corrupt"]), 1)
        self.assertEqual(result["corrupt"][0][1], "lesion")
        self.assertFalse(os.path.exists(
            lfed._cache_path(self.tree.local, "img_a", "lesion", SIZE)))
        with open(lesion, "rb") as handle:
            self.assertEqual(handle.read(), before, "the corrupt persistent file was modified")
        # the other entries still staged -- one bad file must not block the rest
        self.assertGreaterEqual(result["copied"], 4 * (len(self.IDS) - 1))

    def test_a_size_mismatch_during_copy_is_rejected(self):
        self.populate_drive()

        def short_copy(source, destination, *args, **kwargs):
            with open(destination, "wb") as handle:
                handle.write(b"\x00" * 32)

        with mock.patch.object(jcs.shutil, "copyfile", side_effect=short_copy):
            with self.assertRaises(jtd.CorruptCacheFileError) as caught:
                jcs._copy_and_validate(
                    lfed._cache_path(self.tree.drive, "img_a", "vessel", SIZE),
                    lfed._cache_path(self.tree.local, "img_a", "vessel", SIZE),
                    "vessel", "img_a", SIZE)
        self.assertIn("short read", str(caught.exception))

    def test_a_mount_failure_mid_copy_stops_immediately_and_reports(self):
        self.populate_drive()
        real_copyfile = shutil.copyfile
        calls = {"n": 0}

        def flaky(source, destination, *args, **kwargs):
            calls["n"] += 1
            if calls["n"] > 2:
                raise OSError(errno.ENOTCONN, "Transport endpoint is not connected")
            return real_copyfile(source, destination)

        with mock.patch.object(jcs.shutil, "copyfile", side_effect=flaky):
            result = self.mirror()
        self.assertTrue(result["drive_unreachable"])
        self.assertIn("Transport endpoint", result["drive_error"])
        self.assertEqual(calls["n"], 3, "kept copying after the mount failed")
        self.assertEqual(result["copied"], 2)

    def test_enotconn_is_not_recorded_as_a_corrupt_file(self):
        """A mount outage and a bad file are different findings and must not be conflated."""
        self.populate_drive()

        def dead(source, destination, *args, **kwargs):
            raise OSError(errno.ENOTCONN, "Transport endpoint is not connected")

        with mock.patch.object(jcs.shutil, "copyfile", side_effect=dead):
            result = self.mirror()
        self.assertTrue(result["drive_unreachable"])
        self.assertEqual(result["corrupt"], [])


class InsufficientSpaceTests(_StagingTestBase):
    def test_the_mirror_refuses_to_start_when_space_is_insufficient(self):
        self.populate_drive()
        plan = self.plan()
        plan["local_free_bytes"] = 1024          # pretend the disk is nearly full
        plan = jcs._finalize_plan(plan)
        self.assertFalse(plan["fits"])
        with self.assertRaises(jcs.InsufficientLocalSpaceError) as caught:
            self.mirror(plan=plan)
        self.assertIn("Refusing to stage", str(caught.exception))
        self.assertIn("nothing was copied", str(caught.exception))

    def test_nothing_is_copied_when_it_refuses(self):
        self.populate_drive()
        plan = jcs._finalize_plan(dict(self.plan(), local_free_bytes=1024))
        with self.assertRaises(jcs.InsufficientLocalSpaceError):
            self.mirror(plan=plan)
        self.assertEqual(os.listdir(self.tree.local), [])
        self.assertEqual(os.listdir(self.tree.local_racaf), [])


# =====================================================================
# Verification and integrity
# =====================================================================

class VerificationTests(_StagingTestBase):
    def test_verification_reports_zero_drive_fallback_after_a_full_mirror(self):
        self.populate_drive()
        self.mirror()
        report = self.verify()
        self.assertEqual(report["drive_fallback_required"], 0)
        self.assertEqual(report["fully_local"], len(self.IDS))
        for artifact in jcd.ARTIFACTS:
            self.assertEqual(report["artifact_counts"][artifact], len(self.IDS), artifact)
        self.assertGreater(report["local_bytes"], 0)

    def test_verification_separates_drive_fallback_from_missing_everywhere(self):
        """The distinction that decides whether an entry is stageable or a legitimate skip."""
        self.populate_drive(ids=["img_a", "img_b"])   # img_c exists nowhere
        _write_entry(self.tree.local, self.tree.local_racaf, "img_a", seed=0)
        report = self.verify()
        self.assertEqual(report["fully_local"], 1)            # img_a
        self.assertEqual(report["drive_fallback_required"], 1)  # img_b -- stageable
        self.assertEqual(report["missing_everywhere"], 1)       # img_c -- legitimate skip
        self.assertEqual([i for i, _ in report["missing_everywhere_ids"]], ["img_c"])

    def test_verification_flags_a_corrupt_local_file_when_content_is_checked(self):
        self.populate_drive()
        self.mirror()
        np.save(lfed._cache_path(self.tree.local, "img_a", "lesion", SIZE),
                np.zeros((8, 8, 4), dtype=np.float32))
        report = self.verify(validate_contents=True)
        self.assertTrue(any(entry[1] == "lesion" for entry in report["corrupt_local"]))

    def test_verification_default_does_not_load_every_array(self):
        self.populate_drive()
        self.mirror()
        with mock.patch.object(jcs.np, "load", side_effect=AssertionError("should not load")):
            report = self.verify()   # existence + size only
        self.assertEqual(report["fully_local"], len(self.IDS))

    def test_empty_fov_style_entries_are_not_required_to_have_frozen_artifacts(self):
        """An entry with no persistent vessel/lesion/reliability is reported as a legitimate skip,
        not an error, and staging does not invent anything for it."""
        self.populate_drive(ids=["img_a", "img_b"])
        result = self.mirror()
        report = self.verify()
        self.assertEqual(result["corrupt"], [])
        self.assertEqual(report["missing_everywhere"], 1)
        self.assertEqual(report["fully_local"], 2)
        self.assertFalse(os.path.exists(
            lfed._cache_path(self.tree.local, "img_c", "vessel", SIZE)))


class NumericalIntegrityTests(_StagingTestBase):
    def test_staged_copies_are_numerically_identical_to_the_persistent_originals(self):
        self.populate_drive()
        self.mirror()
        comparisons = jcs.sample_numerical_integrity(
            self.tree.entries, self.tree.local, self.tree.local_racaf,
            self.tree.drive, self.tree.drive_racaf, sample=3)
        self.assertEqual(len(comparisons), 4 * 3)
        for entry in comparisons:
            self.assertNotIn("error", entry)
            self.assertTrue(entry["shapes_match"], entry)
            self.assertTrue(entry["dtypes_match"], entry)
            self.assertEqual(entry["max_abs_diff"], 0.0, entry)
            self.assertTrue(entry["array_equal"], entry)
            self.assertTrue(entry["bytes_identical"], entry)

    def test_the_integrity_check_is_bounded_by_the_sample_size(self):
        self.populate_drive()
        self.mirror()
        comparisons = jcs.sample_numerical_integrity(
            self.tree.entries, self.tree.local, self.tree.local_racaf,
            self.tree.drive, self.tree.drive_racaf, sample=1)
        self.assertEqual(len({entry["id_code"] for entry in comparisons}), 1)


class RenderingTests(_StagingTestBase):
    def test_every_renderer_runs_without_raising(self):
        import contextlib
        import io as _io
        self.populate_drive(ids=["img_a", "img_b"])
        buffer = _io.StringIO()
        with contextlib.redirect_stdout(buffer):
            plan = self.plan()
            jcs.print_plan(plan)
            jcs.print_result(self.mirror(plan=plan))
            jcs.print_verification(self.verify())
            jcs.print_integrity(jcs.sample_numerical_integrity(
                self.tree.entries, self.tree.local, self.tree.local_racaf,
                self.tree.drive, self.tree.drive_racaf, sample=1))
        rendered = buffer.getvalue()
        self.assertIn("LOCAL CACHE MIRROR PLAN", rendered)
        self.assertIn("MIRROR RESULT", rendered)
        self.assertIn("LOCAL CACHE VERIFICATION", rendered)
        self.assertIn("NUMERICAL INTEGRITY", rendered)
        self.assertIn("numerically identical", rendered)


if __name__ == "__main__":
    unittest.main()
