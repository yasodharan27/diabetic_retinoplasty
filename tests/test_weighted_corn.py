"""Fast, CPU-only unit tests for weighted_corn.py."""
import os
import unittest

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "-1")

import numpy as np
import tensorflow as tf

import corn
import weighted_corn as wc


class ClassWeightsFromCountsTests(unittest.TestCase):
    def test_uniform_counts_give_uniform_weights(self):
        weights = wc.class_weights_from_counts([100, 100, 100, 100, 100])
        np.testing.assert_allclose(weights, np.ones(5), atol=1e-9)

    def test_normalized_so_expected_weight_is_one(self):
        counts = [1444, 296, 799, 154, 236]
        weights = wc.class_weights_from_counts(counts, power=0.5)
        proportions = np.array(counts, dtype=np.float64) / sum(counts)
        self.assertAlmostEqual(float(np.sum(proportions * weights)), 1.0, places=9)

    def test_rare_class_gets_larger_weight(self):
        weights = wc.class_weights_from_counts([1444, 296, 799, 154, 236], power=0.5)
        self.assertGreater(weights[3], weights[0])   # grade 3 (154) rarer than grade 0 (1444)
        self.assertGreater(weights[3], weights[2])

    def test_power_zero_is_uniform_regardless_of_counts(self):
        weights = wc.class_weights_from_counts([1444, 296, 799, 154, 236], power=0.0)
        np.testing.assert_allclose(weights, np.ones(5), atol=1e-9)

    def test_rejects_nonpositive_counts(self):
        with self.assertRaises(ValueError):
            wc.class_weights_from_counts([100, 0, 50, 10, 10])

    def test_rejects_too_few_classes(self):
        with self.assertRaises(ValueError):
            wc.class_weights_from_counts([100])


class PreregisteredConstantsTests(unittest.TestCase):
    def test_preregistered_weights_match_expected_rounded_constants(self):
        expected = (0.6929, 1.5304, 0.9315, 2.1217, 1.7139)
        rounded = tuple(round(w, 4) for w in wc.PREREGISTERED_CLASS_WEIGHTS)
        for got, want in zip(rounded, expected):
            self.assertAlmostEqual(got, want, places=4)

    def test_pinned_train_counts_match_the_committed_split_manifest(self):
        import downstream_split
        train_entries, _val_entries = downstream_split.get_authoritative_split()
        counts = [0] * 5
        for _id, diagnosis in train_entries:
            counts[int(diagnosis)] += 1
        self.assertEqual(tuple(counts), wc.PREREGISTERED_TRAIN_COUNTS)


def _random_logits_and_grades(batch=8, num_thresholds=corn.NUM_THRESHOLDS, seed=0):
    rng = np.random.default_rng(seed)
    logits = rng.normal(size=(batch, num_thresholds)).astype(np.float32)
    grades = rng.integers(0, corn.NUM_GRADES, size=batch).astype(np.int32)
    return logits, grades


class WeightedCornLossIdentityTests(unittest.TestCase):
    """At uniform weights, weighted_corn_loss must equal corn.corn_loss exactly."""

    def test_unit_weights_match_corn_loss_float32(self):
        logits, grades = _random_logits_and_grades()
        expected = float(corn.corn_loss(logits, grades))
        loss_fn = wc.make_weighted_corn_loss([1.0] * corn.NUM_GRADES)
        got = float(loss_fn(grades, logits))
        self.assertAlmostEqual(got, expected, places=6)

    def test_unit_weights_match_corn_loss_multiple_batches(self):
        for seed in range(5):
            logits, grades = _random_logits_and_grades(batch=16, seed=seed)
            expected = float(corn.corn_loss(logits, grades))
            loss_fn = wc.make_weighted_corn_loss([1.0] * corn.NUM_GRADES)
            got = float(loss_fn(grades, logits))
            self.assertAlmostEqual(got, expected, places=5, msg=f"seed={seed}")

    def test_mixed_float16_logits_are_handled_like_corn_loss(self):
        logits, grades = _random_logits_and_grades()
        logits16 = tf.cast(logits, tf.float16)
        expected = float(corn.corn_loss(logits16, grades))
        loss_fn = wc.make_weighted_corn_loss([1.0] * corn.NUM_GRADES)
        got = float(loss_fn(grades, logits16))
        self.assertAlmostEqual(got, expected, places=3)  # float16 logits -> looser tolerance


class WeightedCornLossWeightingTests(unittest.TestCase):
    def test_denominator_is_unweighted_included_pair_count(self):
        # Two samples, both grade 0 (only task 0 included -> 1 pair each -> denominator 2
        # regardless of the class weight applied to grade 0's numerator).
        logits = np.zeros((2, corn.NUM_THRESHOLDS), dtype=np.float32)
        grades = np.array([0, 0], dtype=np.int32)
        weights = [10.0, 1.0, 1.0, 1.0, 1.0]
        value = float(wc.weighted_corn_loss_value(logits, grades, weights))
        # logits are all zero -> sigmoid_cross_entropy_with_logits(target=0, logit=0) = log(2)
        # per included pair; only task 0 is included for grade 0 -> 2 pairs total.
        expected_unweighted_pair_loss = np.log(2.0)
        expected = 10.0 * expected_unweighted_pair_loss  # weight applied to numerator only
        self.assertAlmostEqual(value, expected, places=5)

    def test_higher_weight_on_true_class_increases_loss_proportionally_isolated(self):
        logits, grades = _random_logits_and_grades(batch=1, seed=1)
        grades = np.array([2], dtype=np.int32)
        base = float(wc.weighted_corn_loss_value(logits, grades, [1.0] * 5))
        doubled = float(wc.weighted_corn_loss_value(logits, grades, [1.0, 1.0, 2.0, 1.0, 1.0]))
        self.assertAlmostEqual(doubled, 2.0 * base, places=5)

    def test_gradients_are_finite(self):
        logits, grades = _random_logits_and_grades(batch=32, seed=2)
        logits_var = tf.Variable(logits)
        loss_fn = wc.make_weighted_corn_loss(wc.PREREGISTERED_CLASS_WEIGHTS)
        with tf.GradientTape() as tape:
            loss = loss_fn(grades, logits_var)
        grads = tape.gradient(loss, logits_var)
        self.assertTrue(np.all(np.isfinite(grads.numpy())))
        self.assertGreater(float(tf.reduce_sum(tf.abs(grads))), 0.0)


class UnweightedCORNLossMetricTests(unittest.TestCase):
    def test_matches_corn_loss_after_one_batch(self):
        logits, grades = _random_logits_and_grades(batch=16, seed=3)
        metric = wc.UnweightedCORNLoss()
        metric.update_state(grades, logits)
        expected = float(corn.corn_loss(logits, grades))
        self.assertAlmostEqual(float(metric.result()), expected, places=5)

    def test_pools_across_batches_not_mean_of_means(self):
        logits1, grades1 = _random_logits_and_grades(batch=4, seed=4)
        logits2, grades2 = _random_logits_and_grades(batch=20, seed=5)
        metric = wc.UnweightedCORNLoss()
        metric.update_state(grades1, logits1)
        metric.update_state(grades2, logits2)
        pooled = float(metric.result())

        all_logits = np.concatenate([logits1, logits2], axis=0)
        all_grades = np.concatenate([grades1, grades2], axis=0)
        expected = float(corn.corn_loss(all_logits, all_grades))
        self.assertAlmostEqual(pooled, expected, places=5)

    def test_reset_state_clears_accumulation(self):
        logits, grades = _random_logits_and_grades(batch=8, seed=6)
        metric = wc.UnweightedCORNLoss()
        metric.update_state(grades, logits)
        metric.reset_state()
        self.assertEqual(float(metric.total_count.numpy()), 0.0)
        self.assertEqual(float(metric.total_loss.numpy()), 0.0)


if __name__ == "__main__":
    unittest.main()
