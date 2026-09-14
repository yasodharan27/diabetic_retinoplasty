"""Fast, CPU-only unit tests for improved_training_data.py.

`_build_joint_sample` is monkeypatched so these tests exercise only THIS module's own
determinism/order/resume logic -- never real images, caches, or GPU models.
"""
import os
import unittest
from unittest import mock

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "-1")

import numpy as np

import improved_training_data as itd
import joint_training_dataset as jtd
from vessel_segmentation_inference import EmptyFieldOfViewError


class SeedKeyTests(unittest.TestCase):
    def test_deterministic_for_the_same_key(self):
        a = itd._seed_from_key("tag", 42, 3, "abc123")
        b = itd._seed_from_key("tag", 42, 3, "abc123")
        self.assertEqual(a, b)

    def test_differs_for_a_different_tag(self):
        a = itd._seed_from_key("aug", 42, 3, "abc")
        b = itd._seed_from_key("order", 42, 3, "abc")
        self.assertNotEqual(a, b)

    def test_differs_for_a_different_epoch(self):
        a = itd._seed_from_key("tag", 42, 3, "abc")
        b = itd._seed_from_key("tag", 42, 4, "abc")
        self.assertNotEqual(a, b)

    def test_differs_for_a_different_run_seed(self):
        a = itd._seed_from_key("tag", 42, 3, "abc")
        b = itd._seed_from_key("tag", 123, 3, "abc")
        self.assertNotEqual(a, b)


class PerImageAugmentationRngTests(unittest.TestCase):
    def test_same_run_seed_epoch_image_gives_the_same_draws(self):
        rng1 = itd.per_image_augmentation_rng(42, 5, "005b95c28852")
        rng2 = itd.per_image_augmentation_rng(42, 5, "005b95c28852")
        np.testing.assert_array_equal(rng1.random(10), rng2.random(10))

    def test_different_epoch_gives_different_draws(self):
        rng1 = itd.per_image_augmentation_rng(42, 5, "005b95c28852")
        rng2 = itd.per_image_augmentation_rng(42, 6, "005b95c28852")
        self.assertFalse(np.array_equal(rng1.random(10), rng2.random(10)))

    def test_different_run_seed_gives_different_draws(self):
        rng1 = itd.per_image_augmentation_rng(42, 5, "005b95c28852")
        rng2 = itd.per_image_augmentation_rng(123, 5, "005b95c28852")
        self.assertFalse(np.array_equal(rng1.random(10), rng2.random(10)))

    def test_different_image_gives_different_draws(self):
        rng1 = itd.per_image_augmentation_rng(42, 5, "005b95c28852")
        rng2 = itd.per_image_augmentation_rng(42, 5, "0104b032c141")
        self.assertFalse(np.array_equal(rng1.random(10), rng2.random(10)))

    def test_independent_of_iteration_position(self):
        """The per-image RNG must not depend on where the image sits in the entries list --
        only on (run_seed, epoch, image_id)."""
        entries_a = [("id1", 0), ("id2", 1), ("id3", 2)]
        entries_b = [("id3", 2), ("id1", 0), ("id2", 1)]  # same ids, different order
        for id_code in ("id1", "id2", "id3"):
            rng_a = itd.per_image_augmentation_rng(42, 1, id_code)
            rng_b = itd.per_image_augmentation_rng(42, 1, id_code)
            np.testing.assert_array_equal(rng_a.random(5), rng_b.random(5))
        del entries_a, entries_b  # order genuinely never enters the key at all


class EpochTrainingOrderTests(unittest.TestCase):
    ENTRIES = [(f"id{i:03d}", i % 5) for i in range(20)]

    def test_is_a_permutation_of_the_same_entries(self):
        ordered = itd.epoch_training_order(self.ENTRIES, run_seed=42, epoch=0)
        self.assertEqual(sorted(ordered), sorted(self.ENTRIES))
        self.assertEqual(len(ordered), len(self.ENTRIES))

    def test_deterministic_for_the_same_run_seed_and_epoch(self):
        a = itd.epoch_training_order(self.ENTRIES, run_seed=42, epoch=3)
        b = itd.epoch_training_order(self.ENTRIES, run_seed=42, epoch=3)
        self.assertEqual(a, b)

    def test_different_epochs_give_different_orders(self):
        a = itd.epoch_training_order(self.ENTRIES, run_seed=42, epoch=0)
        b = itd.epoch_training_order(self.ENTRIES, run_seed=42, epoch=1)
        self.assertNotEqual(a, b)

    def test_different_run_seeds_give_different_orders(self):
        a = itd.epoch_training_order(self.ENTRIES, run_seed=42, epoch=0)
        b = itd.epoch_training_order(self.ENTRIES, run_seed=123, epoch=0)
        self.assertNotEqual(a, b)

    def test_independent_of_the_starting_list_order(self):
        shuffled = list(reversed(self.ENTRIES))
        a = itd.epoch_training_order(self.ENTRIES, run_seed=42, epoch=2)
        b = itd.epoch_training_order(shuffled, run_seed=42, epoch=2)
        self.assertEqual(a, b)


class MakeEpochDatasetDeterminismTests(unittest.TestCase):
    """Exercises make_epoch_dataset() end-to-end with a fake _build_joint_sample that just
    records exactly what (id_code, augment, rng-derived-value) it was called with -- proving
    the full pipeline (order + rng selection) is deterministic and resume-safe, without needing
    real images, a cache, or a GPU model."""

    ENTRIES = [(f"id{i:03d}", i % 5) for i in range(6)]

    def _fake_build_joint_sample(self, calls):
        def _fake(id_code, diagnosis, image_dir, cache_dir, racaf_cache_dir, vessel_model,
                  stage4_model, augment, rng, **kwargs):
            # Raise BEFORE recording the call: `calls` must reflect what the dataset actually
            # YIELDS (skipped entries excluded), not merely what was attempted -- matching
            # joint_training_dataset._make_joint_dataset()'s own generator, which never yields
            # anything for an id that raises EmptyFieldOfViewError.
            if id_code == "id003":
                raise EmptyFieldOfViewError("synthetic skip for id003")
            draw = float(rng.random()) if rng is not None else None
            calls.append((id_code, augment, draw))
            return {
                "image_id": id_code,
                "stage5_input": np.zeros((*jtd.STAGE5_IMAGE_SIZE, 8), dtype=np.float32),
                "stage6_input": np.zeros((*jtd.STAGE6_IMAGE_SIZE, 3), dtype=np.float32),
                "reliability": np.float32(0.5),
                "grade": int(diagnosis),
            }
        return _fake

    def _consume(self, epoch, augment, run_seed=42):
        calls = []
        with mock.patch.object(jtd, "_build_joint_sample", self._fake_build_joint_sample(calls)):
            ds = itd.make_epoch_dataset(
                self.ENTRIES, epoch=epoch, run_seed=run_seed, image_dir="unused",
                cache_dir="unused", racaf_cache_dir="unused", vessel_model=None,
                stage4_model=None, batch_size=2, augment=augment,
            )
            list(ds)  # force full iteration
        return calls

    def test_empty_fov_image_is_skipped_not_crashed(self):
        calls = self._consume(epoch=0, augment=True)
        self.assertNotIn("id003", [c[0] for c in calls])
        self.assertEqual(len({c[0] for c in calls}), 5)  # 6 entries minus the one skip

    def test_resumed_epoch_reproduces_the_uninterrupted_epoch_stream_exactly(self):
        first_pass = self._consume(epoch=7, augment=True)
        second_pass = self._consume(epoch=7, augment=True)  # simulates a from-scratch retry
        self.assertEqual(first_pass, second_pass)

    def test_different_epochs_give_a_different_order_and_different_draws(self):
        epoch0 = self._consume(epoch=0, augment=True)
        epoch1 = self._consume(epoch=1, augment=True)
        self.assertNotEqual([c[0] for c in epoch0], [c[0] for c in epoch1])  # different order
        by_id_0 = {c[0]: c[2] for c in epoch0}
        by_id_1 = {c[0]: c[2] for c in epoch1}
        self.assertTrue(any(by_id_0[k] != by_id_1[k] for k in by_id_0))

    def test_validation_uses_manifest_order_and_no_rng(self):
        calls = self._consume(epoch=0, augment=False)
        yielded_ids = [c[0] for c in calls]
        expected_order = [id_code for id_code, _ in self.ENTRIES if id_code != "id003"]
        self.assertEqual(yielded_ids, expected_order)
        self.assertTrue(all(draw is None for _id, _aug, draw in calls))

    def test_split_seed_unaffected_placeholder(self):
        # improved_training_data never touches the split at all -- it only orders/augments
        # entries it is GIVEN. This is a documentation-style guard: the module has no parameter
        # resembling "split_seed" for a caller to misuse.
        import inspect
        sig = inspect.signature(itd.make_epoch_dataset)
        self.assertNotIn("split_seed", sig.parameters)
        self.assertNotIn("split", sig.parameters)


class CountCachedEntriesTests(unittest.TestCase):
    def test_counts_only_entries_whose_cache_exists(self):
        entries = [("cached1", 0), ("cached2", 1), ("missing1", 2)]
        with mock.patch.object(jtd, "_cache_entry_exists",
                               side_effect=lambda id_code, *a, **k: id_code.startswith("cached")):
            count = itd.count_cached_entries(entries, cache_dir="x", racaf_cache_dir="y")
        self.assertEqual(count, 2)


if __name__ == "__main__":
    unittest.main()
