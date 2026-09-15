"""Fast, CPU-only unit tests for improved_training_data.py.

Determinism/order tests patch `load_cached_sample`; the data-path tests build a REAL tiny local
cache and make every upstream entry point (raw image load, Stage 02, Stage 03, Stage 04, Drive
read) raise, proving training reads the cache and nothing else.
"""
import os
import shutil
import tempfile
import unittest
from unittest import mock

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "-1")

import numpy as np

import improved_training_data as itd
import joint_cache_diagnostics as jcd
import joint_training_dataset as jtd
import local_feature_extraction_dataset as lfed
import racaf

SMALL = (16, 16)


def _write_entry(id_code, cache_dir, racaf_cache_dir, artifacts=jcd.ARTIFACTS, image_size=SMALL):
    paths = jcd.artifact_paths(id_code, cache_dir, racaf_cache_dir, image_size)
    rng = np.random.default_rng(abs(hash(id_code)) % 2 ** 32)
    for artifact in artifacts:
        os.makedirs(os.path.dirname(paths[artifact]), exist_ok=True)
        if artifact == "reliability":
            np.savez(paths[artifact], kappa=rng.random(4).astype(np.float32), r=np.float32(0.7))
        else:
            channels = {"vessel": 1, "lesion": 4, "rgb": 3}[artifact]
            np.save(paths[artifact], rng.random((*image_size, channels)).astype(np.float32))


class TempDirTestCase(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.mkdtemp()
        self.addCleanup(shutil.rmtree, self.tmp, ignore_errors=True)
        self.cache = os.path.join(self.tmp, "local", "features")
        self.racaf_cache = os.path.join(self.tmp, "local", "racaf")
        self.drive_cache = os.path.join(self.tmp, "drive", "features")
        self.drive_racaf_cache = os.path.join(self.tmp, "drive", "racaf")


class SeedKeyTests(unittest.TestCase):
    def test_deterministic_for_the_same_key(self):
        self.assertEqual(itd._seed_from_key("tag", 42, 3, "abc123"),
                         itd._seed_from_key("tag", 42, 3, "abc123"))

    def test_differs_for_a_different_tag_epoch_or_seed(self):
        base = itd._seed_from_key("aug", 42, 3, "abc")
        self.assertNotEqual(base, itd._seed_from_key("order", 42, 3, "abc"))
        self.assertNotEqual(base, itd._seed_from_key("aug", 42, 4, "abc"))
        self.assertNotEqual(base, itd._seed_from_key("aug", 123, 3, "abc"))


class PerImageAugmentationRngTests(unittest.TestCase):
    def test_same_key_gives_the_same_draws(self):
        np.testing.assert_array_equal(itd.per_image_augmentation_rng(42, 5, "x").random(10),
                                      itd.per_image_augmentation_rng(42, 5, "x").random(10))

    def test_epoch_seed_or_image_change_the_draws(self):
        base = itd.per_image_augmentation_rng(42, 5, "x").random(10)
        for other in (itd.per_image_augmentation_rng(42, 6, "x"),
                      itd.per_image_augmentation_rng(123, 5, "x"),
                      itd.per_image_augmentation_rng(42, 5, "y")):
            self.assertFalse(np.array_equal(base, other.random(10)))


class EpochTrainingOrderTests(unittest.TestCase):
    ENTRIES = [(f"id{i:03d}", i % 5) for i in range(20)]

    def test_is_a_permutation_of_the_same_entries(self):
        ordered = itd.epoch_training_order(self.ENTRIES, run_seed=42, epoch=0)
        self.assertEqual(sorted(ordered), sorted(self.ENTRIES))

    def test_deterministic_and_epoch_and_seed_dependent(self):
        a = itd.epoch_training_order(self.ENTRIES, run_seed=42, epoch=3)
        self.assertEqual(a, itd.epoch_training_order(self.ENTRIES, run_seed=42, epoch=3))
        self.assertNotEqual(a, itd.epoch_training_order(self.ENTRIES, run_seed=42, epoch=4))
        self.assertNotEqual(a, itd.epoch_training_order(self.ENTRIES, run_seed=123, epoch=3))

    def test_independent_of_the_starting_list_order(self):
        self.assertEqual(itd.epoch_training_order(self.ENTRIES, run_seed=42, epoch=2),
                         itd.epoch_training_order(list(reversed(self.ENTRIES)), run_seed=42, epoch=2))


class MakeEpochDatasetDeterminismTests(unittest.TestCase):
    ENTRIES = [(f"id{i:03d}", i % 5) for i in range(6)]

    def _consume(self, epoch, augment, run_seed=42):
        calls = []

        def fake(id_code, diagnosis, cache_dir, racaf_cache_dir, augment, rng, image_size):
            calls.append((id_code, augment, float(rng.random()) if rng is not None else None))
            return {"stage5_input": np.zeros((*image_size, 8), np.float32),
                    "stage6_input": np.zeros((*jtd.STAGE6_IMAGE_SIZE, 3), np.float32),
                    "reliability": np.float32(0.5), "grade": int(diagnosis)}

        with mock.patch.object(itd, "load_cached_sample", side_effect=fake):
            list(itd.make_epoch_dataset(self.ENTRIES, epoch=epoch, run_seed=run_seed,
                                        cache_dir="c", racaf_cache_dir="r", batch_size=2,
                                        augment=augment, image_size=SMALL))
        return calls

    def test_rebuilt_epoch_reproduces_the_stream_exactly(self):
        self.assertEqual(self._consume(epoch=7, augment=True), self._consume(epoch=7, augment=True))

    def test_different_epochs_give_a_different_order_and_different_draws(self):
        epoch0, epoch1 = self._consume(0, True), self._consume(1, True)
        self.assertNotEqual([c[0] for c in epoch0], [c[0] for c in epoch1])
        by_id_0, by_id_1 = {c[0]: c[2] for c in epoch0}, {c[0]: c[2] for c in epoch1}
        self.assertTrue(any(by_id_0[k] != by_id_1[k] for k in by_id_0))

    def test_validation_uses_given_order_and_no_rng(self):
        calls = self._consume(epoch=0, augment=False)
        self.assertEqual([c[0] for c in calls], [e[0] for e in self.ENTRIES])
        self.assertTrue(all(draw is None for _i, _a, draw in calls))


def _forbid(name):
    def _raise(*_args, **_kwargs):
        raise AssertionError(f"upstream entry point {name} was called during cached training")
    return _raise


class CacheOnlyDataPathTests(TempDirTestCase):
    """The real `_build_joint_sample` over a real local cache, with every upstream stage and every
    Drive read made to raise."""

    def setUp(self):
        super().setUp()
        self.entries = [("a1", 0), ("b2", 3), ("c3", 1)]
        for id_code, _ in self.entries:
            _write_entry(id_code, self.cache, self.racaf_cache)
        for target, name in ((lfed, "_load_raw_bgr"), (lfed, "_resolve_processed_rgb"),
                             (jtd, "predict_vessel_mask"), (racaf, "tta_views"),
                             (racaf, "prepare_stage4_input"), (jtd, "_load_persistent_array"),
                             (jtd, "_persistent_exists")):
            patcher = mock.patch.object(target, name, side_effect=_forbid(name))
            patcher.start()
            self.addCleanup(patcher.stop)

    def test_training_and_validation_epochs_read_only_the_local_cache(self):
        for augment in (True, False):
            batches = list(itd.make_epoch_dataset(self.entries, epoch=2, run_seed=42,
                                                  cache_dir=self.cache,
                                                  racaf_cache_dir=self.racaf_cache,
                                                  batch_size=2, augment=augment, image_size=SMALL))
            self.assertEqual(sum(int(b[1].shape[0]) for b in batches), len(self.entries))
            stage5, stage6, reliability = batches[0][0]
            self.assertEqual(tuple(stage5.shape[1:]), (*SMALL, 8))
            self.assertEqual(tuple(stage6.shape[1:]), (*jtd.STAGE6_IMAGE_SIZE, 3))
            np.testing.assert_allclose(reliability.numpy(), 0.7)

    def test_unaugmented_sample_is_exactly_the_cached_representation(self):
        sample = itd.load_cached_sample("a1", 0, self.cache, self.racaf_cache, False, None, SMALL)
        paths = jcd.artifact_paths("a1", self.cache, self.racaf_cache, SMALL)
        expected = np.concatenate([np.load(paths["rgb"]), np.load(paths["vessel"]),
                                   np.load(paths["lesion"])], axis=-1)
        np.testing.assert_array_equal(sample["stage5_input"], expected)

    def test_an_incomplete_entry_raises_instead_of_recomputing(self):
        os.remove(jcd.artifact_paths("b2", self.cache, self.racaf_cache, SMALL)["lesion"])
        with self.assertRaises(itd.UncachedEntryError):
            itd.load_cached_sample("b2", 3, self.cache, self.racaf_cache, True,
                                   itd.per_image_augmentation_rng(42, 0, "b2"), SMALL)

    def test_locally_cached_entries_excludes_incomplete_ones_in_order(self):
        os.remove(jcd.artifact_paths("b2", self.cache, self.racaf_cache, SMALL)["rgb"])
        self.assertEqual(itd.locally_cached_entries(self.entries, self.cache, self.racaf_cache, SMALL),
                         [("a1", 0), ("c3", 1)])


class CompleteLocalCacheTests(TempDirTestCase):
    def setUp(self):
        super().setUp()
        self.mirror = self._patch("mirror_persistent_cache_to_local",
                                  return_value={"drive_unreachable": False, "corrupt": [], "copied": 0})
        self.stage = self._patch("stage_raw_images_for_uncached_entries",
                                 return_value={"missing_at_source": [], "drive_unreachable": False,
                                               "copied": 1})
        self.precompute = mock.patch.object(jtd, "precompute_joint_frozen_caches").start()
        self.addCleanup(mock.patch.stopall)

    def _patch(self, name, **kwargs):
        return mock.patch.object(itd.jcs, name, **kwargs).start()

    def _run(self, entries, known=None):
        return itd.complete_local_cache(entries, self.cache, self.racaf_cache, self.drive_cache,
                                        self.drive_racaf_cache, source_image_dir="raw",
                                        local_image_dir="staged", known_empty_fov_ids=known,
                                        image_size=SMALL)

    def test_a_complete_cache_is_reused_without_drive_raw_images_or_models(self):
        _write_entry("a1", self.cache, self.racaf_cache)
        report = self._run([("a1", 0), ("e1", 2)], known=["e1"])
        self.mirror.assert_not_called()
        self.stage.assert_not_called()
        self.precompute.assert_not_called()
        self.assertEqual(report["empty_fov_ids"], ["e1"])
        self.assertEqual(report["already_local"], 1)

    def test_a_drive_only_entry_is_mirrored_not_generated(self):
        def mirror(candidates, *args, **kwargs):
            for id_code, _ in candidates:
                _write_entry(id_code, self.cache, self.racaf_cache)
            return {"drive_unreachable": False, "corrupt": [], "copied": 4}
        self.mirror.side_effect = mirror
        report = self._run([("m1", 1)])
        self.assertEqual(report["mirrored_files"], 4)
        self.stage.assert_not_called()
        self.precompute.assert_not_called()

    def test_a_missing_entry_is_generated_once_and_persisted(self):
        def generate(entries, **kwargs):
            for id_code, _ in entries:
                _write_entry(id_code, self.cache, self.racaf_cache)
            return {"skipped_empty_fov": []}
        self.precompute.side_effect = generate
        _write_entry("a1", self.cache, self.racaf_cache)

        report = self._run([("a1", 0), ("g1", 4)])
        self.assertEqual(report["generated_ids"], ["g1"])
        self.assertEqual([e for e in self.stage.call_args[0][0]], [("g1", 4)])
        self.assertEqual(self.precompute.call_args[0][0], [("g1", 4)])
        drive_paths = jcd.artifact_paths("g1", self.drive_cache, self.drive_racaf_cache, SMALL)
        self.assertTrue(all(os.path.exists(p) for p in drive_paths.values()))
        self.assertEqual(report["persisted_files"], 4)

        self.mirror.reset_mock(); self.stage.reset_mock(); self.precompute.reset_mock()
        again = self._run([("a1", 0), ("g1", 4)])
        self.precompute.assert_not_called()
        self.stage.assert_not_called()
        self.mirror.assert_not_called()
        self.assertEqual(again["generated_ids"], [])

    def test_empty_fov_is_detected_once_then_never_regenerated_once_pinned(self):
        self.precompute.return_value = {"skipped_empty_fov": ["e1"]}
        first = self._run([("e1", 2)])
        self.assertEqual(first["empty_fov_ids"], ["e1"])
        self.assertEqual(first["persisted_files"], 0)
        self.precompute.reset_mock(); self.stage.reset_mock()
        self._run([("e1", 2)], known=first["empty_fov_ids"])
        self.precompute.assert_not_called()
        self.stage.assert_not_called()

    def test_an_unexplained_missing_entry_refuses(self):
        self.precompute.return_value = {"skipped_empty_fov": []}
        with self.assertRaises(RuntimeError):
            self._run([("x1", 0)])


if __name__ == "__main__":
    unittest.main()
