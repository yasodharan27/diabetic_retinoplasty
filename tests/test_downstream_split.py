"""
Tests for downstream_split.py -- the authoritative APTOS2019 train/val
split shared by every downstream trainable stage.

Uses a small, synthetic, on-disk train.csv (never the real, local
APTOS2019 dataset) for every test that computes a split from scratch, per
PROJECT_CODE.md's "unit tests use synthetic/temporary data only" rule. The
one exception (`RealManifestTests`) only reads the already-committed
`dataset_splits/aptos2019_train_val_split.csv` manifest and the real
`train.csv`'s row count -- both fast, already-on-disk checks, not a
network/training operation.
"""

import csv
import os
import shutil
import tempfile
import unittest

import downstream_split as ds


class _SyntheticCSV:
    """A temporary train.csv with `id_code,diagnosis` rows."""

    def __init__(self, id_diagnosis_pairs):
        self.root = tempfile.mkdtemp(prefix="downstream_split_synth_")
        self.csv_path = os.path.join(self.root, "train.csv")
        with open(self.csv_path, "w", newline="", encoding="utf-8") as handle:
            writer = csv.writer(handle)
            writer.writerow(["id_code", "diagnosis"])
            for id_code, diagnosis in id_diagnosis_pairs:
                writer.writerow([id_code, diagnosis])

    def cleanup(self):
        shutil.rmtree(self.root, ignore_errors=True)


def _balanced_pairs(per_class=6, num_classes=5):
    return [(f"img_{c}_{i:02d}", c) for c in range(num_classes) for i in range(per_class)]


class ListLabeledImagesTests(unittest.TestCase):
    def setUp(self):
        self.csv = _SyntheticCSV([("a", 0), ("b", 2), ("c", 4)])
        self.addCleanup(self.csv.cleanup)

    def test_reads_id_code_and_diagnosis(self):
        entries = ds.list_labeled_images(self.csv.csv_path)
        self.assertEqual(sorted(entries), [("a", 0), ("b", 2), ("c", 4)])

    def test_diagnosis_is_int(self):
        for _, diagnosis in ds.list_labeled_images(self.csv.csv_path):
            self.assertIsInstance(diagnosis, int)

    def test_missing_csv_raises(self):
        with self.assertRaises(FileNotFoundError):
            ds.list_labeled_images(os.path.join(self.csv.root, "does_not_exist.csv"))


class ComputeSplitTests(unittest.TestCase):
    def setUp(self):
        self.csv = _SyntheticCSV(_balanced_pairs(per_class=6))
        self.addCleanup(self.csv.cleanup)

    def test_deterministic_across_calls(self):
        train_a, val_a = ds.compute_split(self.csv.csv_path, val_split=0.3, seed=42)
        train_b, val_b = ds.compute_split(self.csv.csv_path, val_split=0.3, seed=42)
        self.assertEqual(train_a, train_b)
        self.assertEqual(val_a, val_b)

    def test_no_overlap_and_full_coverage(self):
        train_entries, val_entries = ds.compute_split(self.csv.csv_path, val_split=0.3, seed=42)
        train_ids = {i for i, _ in train_entries}
        val_ids = {i for i, _ in val_entries}
        all_ids = {i for i, _ in _balanced_pairs(per_class=6)}
        self.assertEqual(train_ids & val_ids, set())
        self.assertEqual(train_ids | val_ids, all_ids)

    def test_different_seed_can_produce_different_partition(self):
        train_a, _ = ds.compute_split(self.csv.csv_path, val_split=0.3, seed=42)
        train_b, _ = ds.compute_split(self.csv.csv_path, val_split=0.3, seed=7)
        self.assertNotEqual(train_a, train_b)

    def test_stratified_by_diagnosis(self):
        """Each class (6 samples) contributes proportionally to both
        halves, not just to one -- the property `train_test_split`'s
        `stratify=` guarantees and the prior, non-stratified interim split
        did not."""
        train_entries, val_entries = ds.compute_split(self.csv.csv_path, val_split=1 / 3, seed=42)
        train_counts = {}
        val_counts = {}
        for _, diagnosis in train_entries:
            train_counts[diagnosis] = train_counts.get(diagnosis, 0) + 1
        for _, diagnosis in val_entries:
            val_counts[diagnosis] = val_counts.get(diagnosis, 0) + 1
        for c in range(5):
            self.assertEqual(train_counts.get(c, 0), 4, f"class {c} train count")
            self.assertEqual(val_counts.get(c, 0), 2, f"class {c} val count")

    def test_too_few_members_in_a_class_raises_clear_error(self):
        """A class with only 1 member cannot be stratified into a 2-way
        split -- this should surface as a clear error, not a silent,
        non-stratified fallback that would defeat the point of this
        module's stratification guarantee."""
        csv_obj = _SyntheticCSV([("a", 0), ("b", 0), ("c", 1)])
        self.addCleanup(csv_obj.cleanup)
        with self.assertRaises(ValueError):
            ds.compute_split(csv_obj.csv_path, val_split=0.3, seed=42)


class SaveLoadSplitTests(unittest.TestCase):
    def setUp(self):
        self.tmp_dir = tempfile.mkdtemp(prefix="downstream_split_manifest_")
        self.addCleanup(shutil.rmtree, self.tmp_dir, True)
        self.manifest_path = os.path.join(self.tmp_dir, "nested", "split.csv")

    def test_save_creates_file(self):
        ds.save_split([("a", 0), ("b", 1)], [("c", 2)], self.manifest_path)
        self.assertTrue(os.path.exists(self.manifest_path))

    def test_round_trip_preserves_entries(self):
        train_entries = [("a", 0), ("b", 1), ("d", 3)]
        val_entries = [("c", 2)]
        ds.save_split(train_entries, val_entries, self.manifest_path)
        loaded_train, loaded_val = ds.load_split(self.manifest_path)
        self.assertEqual(loaded_train, sorted(train_entries))
        self.assertEqual(loaded_val, sorted(val_entries))

    def test_load_missing_manifest_raises(self):
        with self.assertRaises(FileNotFoundError):
            ds.load_split(os.path.join(self.tmp_dir, "does_not_exist.csv"))

    def test_load_rejects_unknown_split_label(self):
        os.makedirs(os.path.dirname(self.manifest_path), exist_ok=True)
        with open(self.manifest_path, "w", newline="", encoding="utf-8") as handle:
            writer = csv.writer(handle)
            writer.writerow(["id_code", "diagnosis", "split"])
            writer.writerow(["a", "0", "test"])
        with self.assertRaises(ValueError):
            ds.load_split(self.manifest_path)


class GetAuthoritativeSplitTests(unittest.TestCase):
    def setUp(self):
        self.csv = _SyntheticCSV(_balanced_pairs(per_class=6))
        self.addCleanup(self.csv.cleanup)
        self.tmp_dir = tempfile.mkdtemp(prefix="downstream_split_authoritative_")
        self.addCleanup(shutil.rmtree, self.tmp_dir, True)
        self.manifest_path = os.path.join(self.tmp_dir, "split.csv")

    def test_non_default_csv_path_never_reads_the_real_default_manifest(self):
        """A caller passing its own csv_path and a non-default val_split,
        while leaving manifest_path at its own default
        (DEFAULT_SPLIT_MANIFEST -- the real, committed manifest), must
        compute fresh from the given synthetic CSV rather than silently
        returning the real APTOS2019 manifest's content. Proven directly:
        the returned id_codes are exactly this synthetic tree's own ids,
        never a real APTOS id_code from the committed manifest."""
        train_entries, val_entries = ds.get_authoritative_split(
            self.csv.csv_path, val_split=0.3, seed=42,
        )
        returned_ids = {i for i, _ in train_entries + val_entries}
        synthetic_ids = {i for i, _ in _balanced_pairs(per_class=6)}
        self.assertEqual(returned_ids, synthetic_ids)

    def test_custom_manifest_path_caches_independently_of_csv_path(self):
        """A caller who explicitly provides their own manifest_path (never
        DEFAULT_SPLIT_MANIFEST) may cache there regardless of csv_path --
        there is no shared state at risk. First call computes and saves;
        the file must then exist at that explicit location."""
        ds.get_authoritative_split(
            self.csv.csv_path, val_split=0.3, seed=42, manifest_path=self.manifest_path,
        )
        self.assertTrue(os.path.exists(self.manifest_path))

    def test_default_params_create_and_reuse_manifest(self):
        """Simulates the default-parameter path using a temporary manifest
        location (never the real, committed one) -- first call computes
        and saves; second call loads the saved manifest and returns the
        identical partition."""
        first_train, first_val = ds.get_authoritative_split(
            self.csv.csv_path, val_split=ds.DEFAULT_VAL_SPLIT, seed=ds.DEFAULT_SEED,
            manifest_path=self.manifest_path,
        )
        self.assertTrue(os.path.exists(self.manifest_path))

        # Mutate the manifest file itself, then call again -- if the second
        # call is truly reading from disk (not recomputing), it must return
        # what's on disk now, not a freshly recomputed value.
        train_entries, val_entries = ds.load_split(self.manifest_path)
        val_entries = val_entries + [train_entries.pop()]
        ds.save_split(train_entries, val_entries, self.manifest_path)

        second_train, second_val = ds.get_authoritative_split(
            self.csv.csv_path, val_split=ds.DEFAULT_VAL_SPLIT, seed=ds.DEFAULT_SEED,
            manifest_path=self.manifest_path,
        )
        self.assertEqual(second_train, train_entries)
        self.assertEqual(second_val, sorted(val_entries))
        self.assertNotEqual(second_train, first_train)


class RealManifestTests(unittest.TestCase):
    """Checks the real, committed dataset_splits/aptos2019_train_val_split.csv
    manifest against the real, local APTOS2019 train.csv -- both already on
    disk, no computation or network access performed here."""

    def test_manifest_exists(self):
        self.assertTrue(
            os.path.exists(ds.DEFAULT_SPLIT_MANIFEST),
            f"Expected the committed split manifest at {ds.DEFAULT_SPLIT_MANIFEST}.",
        )

    def test_manifest_covers_every_labeled_id_exactly_once(self):
        if not os.path.exists(ds.DEFAULT_TRAIN_CSV):
            self.skipTest("Real APTOS2019 train.csv not present in this environment.")
        real_entries = ds.list_labeled_images(ds.DEFAULT_TRAIN_CSV)
        train_entries, val_entries = ds.load_split(ds.DEFAULT_SPLIT_MANIFEST)
        manifest_entries = sorted(train_entries + val_entries)
        self.assertEqual(manifest_entries, real_entries)

    def test_manifest_train_val_ratio_is_approximately_80_20(self):
        train_entries, val_entries = ds.load_split(ds.DEFAULT_SPLIT_MANIFEST)
        total = len(train_entries) + len(val_entries)
        self.assertAlmostEqual(len(val_entries) / total, ds.DEFAULT_VAL_SPLIT, delta=0.01)

    def test_manifest_no_overlap(self):
        train_entries, val_entries = ds.load_split(ds.DEFAULT_SPLIT_MANIFEST)
        train_ids = {i for i, _ in train_entries}
        val_ids = {i for i, _ in val_entries}
        self.assertEqual(train_ids & val_ids, set())


if __name__ == "__main__":
    unittest.main()
