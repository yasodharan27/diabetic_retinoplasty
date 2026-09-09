"""
Tests for `joint_cache_archive.py` -- the shard-archive representation of the frozen cache.

Synthetic/temporary data only. The "Drive" side is a second temp directory, which is exactly what
a persistent cache dir is to this code: a path it lists and reads.

The property that makes this design safe is that extraction reproduces the EXISTING
representation exactly -- byte-identical individual `.npy`/`.npz` files at the paths the pipeline
already expects -- so nothing about local-first cache resolution changes. That is asserted
directly, and then the real `_build_joint_sample` is run against an extracted cache with every
persistent path booby-trapped, to prove the extracted cache alone is sufficient.
"""

import errno
import io
import os
import shutil
import sys
import tarfile
import tempfile
import unittest
from unittest import mock

import numpy as np

import joint_cache_archive as jca
import joint_cache_diagnostics as jcd
import joint_training_dataset as jtd
import local_feature_extraction_dataset as lfed
import racaf

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

SIZE = jtd.STAGE5_IMAGE_SIZE


def _write_entry(cache_dir, racaf_cache_dir, id_code, image_size=SIZE, seed=0):
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


class _ArchiveTestBase(unittest.TestCase):
    IDS = ["arc_a", "arc_b", "arc_c", "arc_d"]

    def setUp(self):
        self.root = tempfile.mkdtemp(prefix="jca_")
        self.addCleanup(shutil.rmtree, self.root, True)
        self.drive = os.path.join(self.root, "drive", "LocalFeatureExtraction")
        self.drive_racaf = os.path.join(self.root, "drive", "RACAF")
        self.archive = os.path.join(self.root, "drive", "cache_archive")
        self.local = os.path.join(self.root, "content", "cache", "local_feature_extraction")
        self.local_racaf = os.path.join(self.root, "content", "cache", "racaf")
        for directory in (self.drive, self.drive_racaf, self.local, self.local_racaf):
            os.makedirs(directory, exist_ok=True)
        self.entries = [(id_code, i % 5) for i, id_code in enumerate(self.IDS)]

    def populate_drive(self, ids=None):
        for i, id_code in enumerate(ids if ids is not None else self.IDS):
            _write_entry(self.drive, self.drive_racaf, id_code, seed=i)

    def build(self, **overrides):
        kwargs = dict(entries=self.entries, persistent_cache_dir=self.drive,
                      persistent_racaf_cache_dir=self.drive_racaf, archive_dir=self.archive,
                      shard_entries=2, progress=False)
        kwargs.update(overrides)
        return jca.build_archive(**kwargs)

    def extract(self, **overrides):
        kwargs = dict(archive_dir=self.archive, cache_dir=self.local,
                      racaf_cache_dir=self.local_racaf, progress=False)
        kwargs.update(overrides)
        return jca.extract_archive(**kwargs)

    def snapshot_drive(self):
        state = {}
        for directory in (self.drive, self.drive_racaf):
            for name in sorted(os.listdir(directory)):
                path = os.path.join(directory, name)
                with open(path, "rb") as handle:
                    state[path] = (handle.read(), os.stat(path).st_mtime_ns)
        return state


# =====================================================================
# Derived sizes -- the metadata-cost fix
# =====================================================================

class DerivedSizeTests(unittest.TestCase):
    def test_expected_sizes_match_real_npy_files_exactly(self):
        root = tempfile.mkdtemp()
        self.addCleanup(shutil.rmtree, root, True)
        _write_entry(root, root, "sz")
        for artifact, path in jcd.artifact_paths("sz", root, root, SIZE).items():
            expected = jca.expected_artifact_bytes(artifact, SIZE)
            if artifact == "reliability":
                continue  # nominal by design; not geometry-determined
            self.assertEqual(expected, os.path.getsize(path), artifact)

    def test_reliability_uses_a_documented_nominal_size(self):
        self.assertEqual(jca.expected_artifact_bytes("reliability", SIZE),
                         jca.NOMINAL_RELIABILITY_BYTES)

    def test_an_unknown_artifact_has_no_derived_size(self):
        self.assertIsNone(jca.expected_artifact_bytes("not_an_artifact", SIZE))


class ListingTests(_ArchiveTestBase):
    def test_indexing_uses_two_listings_and_no_per_file_stat(self):
        self.populate_drive()
        stats = {"n": 0}
        real_stat = os.stat

        def counting(path, *args, **kwargs):
            if os.path.abspath(str(path)).startswith(os.path.abspath(self.drive)):
                stats["n"] += 1
            return real_stat(path, *args, **kwargs)

        with mock.patch("os.stat", side_effect=counting):
            complete, incomplete, listings = jca.index_persistent_cache(
                self.entries, self.drive, self.drive_racaf)
        self.assertEqual(len(complete), len(self.IDS))
        self.assertEqual(incomplete, {})
        self.assertEqual(listings["feature_files"], 3 * len(self.IDS))
        self.assertEqual(stats["n"], 0, "indexing stat'ed individual persistent files")

    def test_an_incomplete_entry_is_reported_with_its_missing_artifacts(self):
        self.populate_drive()
        os.remove(lfed._cache_path(self.drive, "arc_b", "lesion", SIZE))
        complete, incomplete, _ = jca.index_persistent_cache(
            self.entries, self.drive, self.drive_racaf)
        self.assertEqual(len(complete), len(self.IDS) - 1)
        self.assertEqual(incomplete, {"arc_b": ["lesion"]})

    def test_a_dead_mount_raises_rather_than_listing_empty(self):
        with mock.patch("os.listdir", side_effect=OSError(errno.ENOTCONN, "Transport endpoint")):
            with self.assertRaises(jtd.PersistentCacheUnavailableError):
                jca.list_cache_dir(self.drive)

    def test_an_absent_directory_is_empty_not_an_error(self):
        self.assertEqual(jca.list_cache_dir(os.path.join(self.root, "nope")), set())


# =====================================================================
# Build
# =====================================================================

class BuildTests(_ArchiveTestBase):
    def test_building_produces_shards_covering_every_complete_entry(self):
        self.populate_drive()
        result = self.build()
        self.assertEqual(result["shards_total"], 2)      # 4 entries / shard_entries=2
        self.assertEqual(result["shards_written"], 2)
        self.assertEqual(result["entries_archived"], 4)
        self.assertGreater(result["bytes_written"], 0)
        self.assertEqual(sorted(os.listdir(self.archive)),
                         ["cache_shard_000.tar", "cache_shard_001.tar"])

    def test_building_never_modifies_the_persistent_source(self):
        self.populate_drive()
        before = self.snapshot_drive()
        self.build()
        self.assertEqual(before, self.snapshot_drive())

    def test_incomplete_entries_are_excluded_and_reported_never_fabricated(self):
        self.populate_drive()
        os.remove(lfed._cache_path(self.drive, "arc_b", "vessel", SIZE))
        result = self.build()
        self.assertIn("arc_b", result["incomplete_entries"])
        self.assertEqual(result["entries_archived"], 3)
        self.extract()
        self.assertFalse(os.path.exists(lfed._cache_path(self.local, "arc_b", "vessel", SIZE)))

    def test_building_is_resumable_and_skips_finished_shards(self):
        self.populate_drive()
        first = self.build(max_shards=1)
        self.assertEqual(first["shards_written"], 1)
        second = self.build()
        self.assertEqual(second["shards_skipped"], 1)
        self.assertEqual(second["shards_written"], 1)

    def test_an_interrupted_shard_leaves_no_file_a_later_run_would_trust(self):
        self.populate_drive()
        real_add = tarfile.TarFile.add

        def dying_add(self_tar, *args, **kwargs):
            real_add(self_tar, *args, **kwargs)
            raise OSError(errno.EACCES, "died mid-shard")

        with mock.patch.object(tarfile.TarFile, "add", dying_add):
            with self.assertRaises(OSError):
                self.build()
        self.assertEqual([f for f in os.listdir(self.archive) if not f.endswith(".building")], [])
        self.assertEqual([f for f in os.listdir(self.archive) if f.endswith(".building")], [])

    def test_a_mount_failure_during_build_stops_and_reports(self):
        self.populate_drive()
        with mock.patch.object(tarfile.TarFile, "add",
                               side_effect=OSError(errno.ENOTCONN, "Transport endpoint")):
            result = self.build()
        self.assertTrue(result["drive_unreachable"])
        self.assertEqual(result["shards_written"], 0)

    def test_shard_grouping_is_deterministic(self):
        groups_a = jca.plan_shards(self.entries, shard_entries=2)
        groups_b = jca.plan_shards(list(reversed(self.entries)), shard_entries=2)
        self.assertEqual(groups_a, groups_b)


# =====================================================================
# Extract -- the property that keeps the pipeline unchanged
# =====================================================================

class ExtractTests(_ArchiveTestBase):
    def test_extraction_reproduces_byte_identical_files_at_the_expected_paths(self):
        self.populate_drive()
        self.build()
        result = self.extract()
        self.assertEqual(result["files_extracted"], 4 * len(self.IDS))
        for id_code in self.IDS:
            drive = jcd.artifact_paths(id_code, self.drive, self.drive_racaf, SIZE)
            local = jcd.artifact_paths(id_code, self.local, self.local_racaf, SIZE)
            for artifact in jcd.ARTIFACTS:
                self.assertTrue(os.path.exists(local[artifact]), f"{id_code}/{artifact}")
                with open(drive[artifact], "rb") as a, open(local[artifact], "rb") as b:
                    self.assertEqual(a.read(), b.read(), f"{id_code}/{artifact}")

    def test_the_extracted_cache_alone_satisfies_the_real_sample_builder(self):
        """The decisive compatibility test: build a sample from the extracted cache with EVERY
        persistent path booby-trapped to raise. If extraction did not fully reproduce the expected
        representation, this fails."""
        self.populate_drive()
        self.build()
        self.extract()

        real_stat, real_load = os.stat, np.load
        drive_root = os.path.abspath(os.path.join(self.root, "drive"))

        def guard_stat(path, *args, **kwargs):
            if os.path.abspath(str(path)).startswith(drive_root):
                raise OSError(errno.ENOTCONN, "Transport endpoint is not connected")
            return real_stat(path, *args, **kwargs)

        def guard_load(path, *args, **kwargs):
            if os.path.abspath(str(path)).startswith(drive_root):
                raise OSError(errno.ENOTCONN, "Transport endpoint is not connected")
            return real_load(path, *args, **kwargs)

        with mock.patch.multiple("os", stat=guard_stat), \
             mock.patch.object(jtd.np, "load", guard_load), \
             mock.patch("joint_training_dataset.predict_vessel_mask") as vessel_spy:
            sample = jtd._build_joint_sample(
                "arc_a", 0, os.path.join(self.root, "no_images"), self.local, self.local_racaf,
                None, None, augment=False, rng=None,
                persistent_cache_dir=self.drive, persistent_racaf_cache_dir=self.drive_racaf,
            )
        self.assertEqual(sample["stage5_input"].shape, (*SIZE, 8))
        self.assertEqual(sample["stage6_input"].shape, (*jtd.STAGE6_IMAGE_SIZE, 3))
        self.assertEqual(vessel_spy.call_count, 0, "extraction left a gap that forced recompute")

    def test_extraction_validates_a_sample_of_artifacts(self):
        self.populate_drive()
        self.build()
        result = self.extract(validate_sample=100)
        self.assertGreater(result["validated"], 0)
        self.assertEqual(result["corrupt"], [])

    def test_extraction_is_idempotent_and_resumable(self):
        self.populate_drive()
        self.build()
        first = self.extract()
        second = self.extract()
        self.assertEqual(first["files_extracted"], 4 * len(self.IDS))
        self.assertEqual(second["files_extracted"], 0)
        self.assertEqual(second["skipped_existing"], 4 * len(self.IDS))

    def test_extraction_never_modifies_the_archive_or_the_source(self):
        self.populate_drive()
        self.build()
        before_drive = self.snapshot_drive()
        before_archive = {name: os.path.getsize(os.path.join(self.archive, name))
                          for name in os.listdir(self.archive)}
        self.extract()
        self.assertEqual(before_drive, self.snapshot_drive())
        self.assertEqual(before_archive,
                         {name: os.path.getsize(os.path.join(self.archive, name))
                          for name in os.listdir(self.archive)})

    def test_a_corrupt_shard_is_reported_and_does_not_abort_the_rest(self):
        self.populate_drive()
        self.build()
        shard = os.path.join(self.archive, "cache_shard_000.tar")
        with open(shard, "r+b") as handle:
            handle.seek(1024)
            handle.write(b"\xff" * 4096)
        result = self.extract()
        self.assertTrue(result["corrupt"] or result["files_extracted"] < 4 * len(self.IDS))
        # the second shard still extracted
        self.assertTrue(os.path.exists(
            lfed._cache_path(self.local, "arc_c", "vessel", SIZE))
            or os.path.exists(lfed._cache_path(self.local, "arc_d", "vessel", SIZE)))

    def test_a_mount_failure_during_extraction_stops_and_keeps_what_landed(self):
        self.populate_drive()
        self.build()
        real_open = tarfile.open
        calls = {"n": 0}

        def flaky(*args, **kwargs):
            calls["n"] += 1
            if calls["n"] > 1:
                raise OSError(errno.ENOTCONN, "Transport endpoint is not connected")
            return real_open(*args, **kwargs)

        with mock.patch.object(jca.tarfile, "open", side_effect=flaky):
            result = self.extract()
        self.assertTrue(result["drive_unreachable"])
        self.assertEqual(result["shards_extracted"], 1)

    def test_extraction_refuses_when_free_space_is_insufficient(self):
        self.populate_drive()
        self.build()
        with self.assertRaises(RuntimeError) as caught:
            self.extract(min_free_bytes=10 ** 15)
        self.assertIn("Refusing to extract", str(caught.exception))
        self.assertEqual(os.listdir(self.local), [])

    def test_streaming_extraction_never_seeks_the_source(self):
        """Why a shard can be read straight off a FUSE mount: `r|` is purely sequential, so no
        local copy of the archive is ever needed (archive + extracted would not fit)."""
        self.populate_drive()
        self.build()
        shard = os.path.join(self.archive, "cache_shard_000.tar")
        seeks = {"n": 0}
        real_open = open

        class NoSeek:
            def __init__(self, path):
                self._f = real_open(path, "rb")
            def read(self, n=-1):
                return self._f.read(n)
            def readable(self):
                return True
            def seekable(self):
                return False
            def seek(self, *args):
                seeks["n"] += 1
                raise OSError("seek attempted")
            def close(self):
                self._f.close()

        handle = NoSeek(shard)
        target = os.path.join(self.root, "seektest")
        try:
            with tarfile.open(fileobj=handle, mode="r|") as tar:
                tar.extractall(target)
        finally:
            handle.close()
        self.assertEqual(seeks["n"], 0)


# =====================================================================
# Disk budget -- refuse before reading a byte
# =====================================================================

class ExtractionPlanTests(_ArchiveTestBase):
    def test_plan_measures_shards_local_state_and_free_space(self):
        self.populate_drive()
        self.build()
        plan = jca.plan_extraction(self.archive, self.local, self.local_racaf)
        shard_bytes = sum(os.path.getsize(os.path.join(self.archive, name))
                          for name in os.listdir(self.archive) if name.startswith("cache_shard_"))
        self.assertEqual(plan["archive_bytes"], shard_bytes)
        self.assertEqual(plan["local_cache_bytes"], 0)
        self.assertEqual(plan["local_files"], 0)
        self.assertEqual(plan["required_bytes"],
                         shard_bytes + jca.DEFAULT_RUNTIME_MARGIN_BYTES)

    def test_an_already_extracted_cache_reduces_what_is_still_required(self):
        self.populate_drive()
        self.build()
        empty = jca.plan_extraction(self.archive, self.local, self.local_racaf)
        self.extract()
        resumed = jca.plan_extraction(self.archive, self.local, self.local_racaf)
        self.assertGreater(resumed["local_files"], 0)
        self.assertLess(resumed["required_bytes"], empty["required_bytes"])

    def test_plan_refuses_when_free_space_is_short_and_says_by_how_much(self):
        self.populate_drive()
        self.build()
        Usage = __import__("collections").namedtuple("Usage", "total used free")
        with mock.patch("shutil.disk_usage", return_value=Usage(100, 100, 1024)):
            plan = jca.plan_extraction(self.archive, self.local, self.local_racaf)
        self.assertFalse(plan["fits"])
        self.assertGreater(plan["shortfall_bytes"], 0)

    def test_extract_refuses_and_writes_nothing_when_min_free_bytes_is_not_met(self):
        self.populate_drive()
        self.build()
        before = self.snapshot_drive()
        Usage = __import__("collections").namedtuple("Usage", "total used free")
        with mock.patch("shutil.disk_usage", return_value=Usage(100, 100, 1024)):
            with self.assertRaises(RuntimeError):
                self.extract(min_free_bytes=10 * 1024 ** 3)
        self.assertEqual(os.listdir(self.local), [])
        self.assertEqual(os.listdir(self.local_racaf), [])
        self.assertEqual(self.snapshot_drive(), before)

    def test_an_unlistable_archive_dir_raises_rather_than_reporting_an_empty_archive(self):
        with mock.patch("os.listdir", side_effect=OSError(errno.ENOTCONN, "transport endpoint")):
            with self.assertRaises(jtd.PersistentCacheUnavailableError):
                jca.plan_extraction(self.archive, self.local, self.local_racaf)

    def test_a_dead_mount_while_measuring_shards_is_reported_not_treated_as_zero_bytes(self):
        self.populate_drive()
        self.build()
        with mock.patch("os.stat", side_effect=OSError(errno.ENOTCONN, "transport endpoint")):
            plan = jca.plan_extraction(self.archive, self.local, self.local_racaf)
        self.assertTrue(plan["drive_unreachable"])
        self.assertFalse(plan["fits"])

    def test_plan_reads_no_shard_content(self):
        self.populate_drive()
        self.build()
        real_open = io.open

        def refuse_shard(path, *args, **kwargs):
            if isinstance(path, str) and "cache_shard_" in path:
                raise AssertionError("plan_extraction opened a shard: %s" % path)
            return real_open(path, *args, **kwargs)

        with mock.patch("io.open", side_effect=refuse_shard):
            plan = jca.plan_extraction(self.archive, self.local, self.local_racaf)
        self.assertGreater(plan["archive_bytes"], 0)

    def test_plan_works_on_a_fresh_runtime_where_the_cache_dir_does_not_exist_yet(self):
        """`/content/cache/local_feature_extraction` does not exist before the first extraction,
        and `shutil.disk_usage` raises on a missing path -- the plan must walk up to the nearest
        existing parent instead of dying before it can report anything."""
        self.populate_drive()
        self.build()
        fresh = os.path.join(self.root, "content", "nothing", "here", "yet")
        fresh_racaf = os.path.join(self.root, "content", "nothing", "here", "racaf")
        self.assertFalse(os.path.exists(fresh))
        plan = jca.plan_extraction(self.archive, fresh, fresh_racaf)
        self.assertGreater(plan["free_bytes"], 0)
        self.assertEqual(plan["local_files"], 0)
        self.assertFalse(os.path.exists(fresh))  # planning creates nothing

    def test_renderer_runs_for_both_verdicts(self):
        self.populate_drive()
        self.build()
        jca.print_extraction_plan(jca.plan_extraction(self.archive, self.local, self.local_racaf))
        Usage = __import__("collections").namedtuple("Usage", "total used free")
        with mock.patch("shutil.disk_usage", return_value=Usage(100, 100, 1024)):
            jca.print_extraction_plan(
                jca.plan_extraction(self.archive, self.local, self.local_racaf))

class CompressedShardTests(_ArchiveTestBase):
    def test_a_compressed_archive_round_trips_byte_identically(self):
        """Compression is off by default on evidence, but must still be correct when chosen."""
        self.populate_drive()
        self.build(compress=True)
        self.assertTrue(all(name.endswith(".tar.gz") for name in os.listdir(self.archive)))
        self.extract()
        for id_code in self.IDS:
            drive = jcd.artifact_paths(id_code, self.drive, self.drive_racaf, SIZE)
            local = jcd.artifact_paths(id_code, self.local, self.local_racaf, SIZE)
            for artifact in jcd.ARTIFACTS:
                with open(drive[artifact], "rb") as a, open(local[artifact], "rb") as b:
                    self.assertEqual(a.read(), b.read())


class RenderingTests(_ArchiveTestBase):
    def test_renderers_run_without_raising(self):
        import contextlib
        import io as _io
        self.populate_drive()
        buffer = _io.StringIO()
        with contextlib.redirect_stdout(buffer):
            jca.print_build(self.build())
            jca.print_extract(self.extract())
        rendered = buffer.getvalue()
        self.assertIn("CACHE ARCHIVE BUILD", rendered)
        self.assertIn("CACHE ARCHIVE EXTRACT", rendered)


if __name__ == "__main__":
    unittest.main()
