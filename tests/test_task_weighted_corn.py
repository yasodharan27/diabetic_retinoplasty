"""Fast, CPU-only unit tests for task_weighted_corn.py (task-3 weighting vs mass-matched placebo)."""
import csv
import os
import unittest

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "-1")

import numpy as np
import tensorflow as tf

import corn
import downstream_split
import task_weighted_corn as twc
import weighted_corn as wc

CLASS_WEIGHTS = wc.PREREGISTERED_CLASS_WEIGHTS


def _batch(seed, size=16, grades=None):
    rng = np.random.default_rng(seed)
    logits = rng.normal(0.0, 2.0, size=(size, corn.NUM_THRESHOLDS)).astype(np.float32)
    if grades is None:
        grades = rng.integers(0, corn.NUM_GRADES, size=size)
    return logits, np.asarray(grades, dtype=np.int32)


def _numpy_task_terms(logits, grades, class_weights):
    """Independent NumPy reference: per-task weighted numerator terms and the unweighted denominator."""
    logits = logits.astype(np.float64)
    k = np.arange(corn.NUM_THRESHOLDS)[None, :]
    g = grades[:, None].astype(np.float64)
    mask = (g >= k).astype(np.float64)
    target = (g > k).astype(np.float64)
    bce = np.maximum(logits, 0) - logits * target + np.log1p(np.exp(-np.abs(logits)))
    w = np.asarray(class_weights, dtype=np.float64)[grades][:, None]
    return (bce * mask * w).sum(axis=0), mask.sum()


class IdentityWithWeightedCornTests(unittest.TestCase):
    def test_unit_task_weights_reproduce_weighted_corn_bit_for_bit(self):
        for seed in range(20):
            logits, grades = _batch(seed, size=int(np.random.default_rng(seed).integers(1, 9)))
            ours = twc.task_weighted_corn_loss_value(logits, grades, CLASS_WEIGHTS,
                                                     twc.UNIT_TASK_WEIGHTS)
            theirs = wc.weighted_corn_loss_value(logits, grades, CLASS_WEIGHTS)
            self.assertEqual(float(ours.numpy()), float(theirs.numpy()), f"seed {seed}")

    def test_unit_task_weights_identical_under_float16_logits(self):
        logits, grades = _batch(7)
        half = tf.cast(tf.constant(logits), tf.float16)
        self.assertEqual(
            float(twc.task_weighted_corn_loss_value(half, grades, CLASS_WEIGHTS,
                                                    twc.UNIT_TASK_WEIGHTS).numpy()),
            float(wc.weighted_corn_loss_value(half, grades, CLASS_WEIGHTS).numpy()))

    def test_keras_adapter_matches_weighted_corn_adapter(self):
        logits, grades = _batch(3)
        ours = twc.make_task_weighted_corn_loss(CLASS_WEIGHTS, twc.UNIT_TASK_WEIGHTS)(grades, logits)
        theirs = wc.make_weighted_corn_loss(CLASS_WEIGHTS)(grades, logits)
        self.assertEqual(float(ours.numpy()), float(theirs.numpy()))

    def test_batch_size_two_like_training(self):
        for grades in ([0, 0], [4, 4], [3, 1], [2, 4], [0, 3]):
            logits, g = _batch(11, size=2, grades=grades)
            self.assertEqual(
                float(twc.task_weighted_corn_loss_value(logits, g, CLASS_WEIGHTS,
                                                        twc.UNIT_TASK_WEIGHTS).numpy()),
                float(wc.weighted_corn_loss_value(logits, g, CLASS_WEIGHTS).numpy()))


class TaskSpecificityTests(unittest.TestCase):
    def _check_only_task_changes(self, task, beta):
        weights = twc.task_weights_for(task, beta)
        for seed in range(10):
            logits, grades = _batch(100 + seed)
            numerators, denominator = _numpy_task_terms(logits, grades, CLASS_WEIGHTS)
            base = float(twc.task_weighted_corn_loss_value(logits, grades, CLASS_WEIGHTS,
                                                           twc.UNIT_TASK_WEIGHTS).numpy())
            scaled = float(twc.task_weighted_corn_loss_value(logits, grades, CLASS_WEIGHTS,
                                                             weights).numpy())
            # The change is exactly (beta - 1) * task-k numerator / the SAME unweighted denominator.
            self.assertAlmostEqual(scaled - base, (beta - 1.0) * numerators[task] / denominator,
                                   places=5)

            variable = tf.Variable(logits)
            with tf.GradientTape(persistent=True) as tape:
                loss_base = twc.task_weighted_corn_loss_value(variable, grades, CLASS_WEIGHTS,
                                                              twc.UNIT_TASK_WEIGHTS)
                loss_scaled = twc.task_weighted_corn_loss_value(variable, grades, CLASS_WEIGHTS,
                                                                weights)
            g_base = tape.gradient(loss_base, variable).numpy()
            g_scaled = tape.gradient(loss_scaled, variable).numpy()
            others = [k for k in range(corn.NUM_THRESHOLDS) if k != task]
            np.testing.assert_array_equal(g_scaled[:, others], g_base[:, others])
            np.testing.assert_allclose(g_scaled[:, task], beta * g_base[:, task], rtol=1e-6,
                                       atol=1e-9)

    def test_beta3_changes_only_task3(self):
        self._check_only_task_changes(twc.TREATMENT_TASK, twc.TREATMENT_BETA)

    def test_beta1_changes_only_task1(self):
        self._check_only_task_changes(twc.PLACEBO_TASK, twc.PLACEBO_BETA)

    def test_batch_without_grade3or4_is_unaffected_by_beta3(self):
        logits, grades = _batch(5, size=8, grades=[0, 1, 2, 2, 1, 0, 2, 1])
        self.assertEqual(
            float(twc.task_weighted_corn_loss_value(logits, grades, CLASS_WEIGHTS,
                                                    twc.TREATMENT_TASK_WEIGHTS).numpy()),
            float(wc.weighted_corn_loss_value(logits, grades, CLASS_WEIGHTS).numpy()))

    def test_denominator_is_the_unweighted_pair_count(self):
        logits, grades = _batch(9)
        numerators, denominator = _numpy_task_terms(logits, grades, CLASS_WEIGHTS)
        for weights in (twc.TREATMENT_TASK_WEIGHTS, twc.PLACEBO_TASK_WEIGHTS):
            expected = float(np.dot(numerators, weights) / denominator)
            got = float(twc.task_weighted_corn_loss_value(logits, grades, CLASS_WEIGHTS,
                                                          weights).numpy())
            self.assertAlmostEqual(got, expected, places=5)


class MassMatchingTests(unittest.TestCase):
    def test_masses_match_the_stated_values(self):
        masses = twc.task_loss_masses()
        self.assertAlmostEqual(masses[0], 2929.0, places=6)  # weights normalised: sum(n_c w_c) = N
        self.assertEqual(round(masses[1], 1), 1928.5)
        self.assertEqual(round(masses[3], 1), 731.2)
        self.assertEqual(round(twc.PLACEBO_BETA, 4), 1.3792)

    def test_treatment_and_placebo_add_equal_mass(self):
        masses = twc.task_loss_masses()
        added_t = (twc.TREATMENT_BETA - 1.0) * masses[twc.TREATMENT_TASK]
        added_p = (twc.PLACEBO_BETA - 1.0) * masses[twc.PLACEBO_TASK]
        self.assertAlmostEqual(added_t, added_p, places=9)
        self.assertAlmostEqual(sum(np.multiply(masses, twc.TREATMENT_TASK_WEIGHTS)),
                               sum(np.multiply(masses, twc.PLACEBO_TASK_WEIGHTS)), places=9)

    def test_task_weights_only_touch_their_own_task(self):
        self.assertEqual(twc.TREATMENT_TASK_WEIGHTS, (1.0, 1.0, 1.0, 2.0))
        self.assertEqual(twc.PLACEBO_TASK_WEIGHTS[0], 1.0)
        self.assertEqual(twc.PLACEBO_TASK_WEIGHTS[2:], (1.0, 1.0))

    def test_rejects_bad_task_weights(self):
        logits, grades = _batch(1)
        for bad in ((1.0, 1.0, 1.0), (1.0, 0.0, 1.0, 1.0), (1.0, float("nan"), 1.0, 1.0)):
            with self.assertRaises(ValueError):
                twc.task_weighted_corn_loss_value(logits, grades, CLASS_WEIGHTS, bad)


class NoValidationDataTests(unittest.TestCase):
    """Every data-derived constant of this loss (class weights, masses, placebo beta) comes from the
    TRAINING split only; the loss itself holds no fitted state."""

    def _split_counts(self, split):
        counts = [0] * corn.NUM_GRADES
        with open(downstream_split.DEFAULT_SPLIT_MANIFEST, newline="") as fh:
            for row in csv.DictReader(fh):
                if row["split"] == split:
                    counts[int(row["diagnosis"])] += 1
        return tuple(counts)

    def test_pinned_counts_are_the_training_split(self):
        self.assertEqual(self._split_counts("train"), tuple(wc.PREREGISTERED_TRAIN_COUNTS))

    def test_placebo_beta_derives_from_training_counts_only(self):
        from_train = twc.mass_matched_beta(
            twc.PLACEBO_TASK, twc.TREATMENT_TASK, twc.TREATMENT_BETA,
            masses=twc.task_loss_masses(self._split_counts("train")))
        self.assertAlmostEqual(from_train, twc.PLACEBO_BETA, places=12)
        from_val = twc.mass_matched_beta(
            twc.PLACEBO_TASK, twc.TREATMENT_TASK, twc.TREATMENT_BETA,
            masses=twc.task_loss_masses(self._split_counts("val")))
        self.assertNotAlmostEqual(from_val, twc.PLACEBO_BETA, places=6)  # a leak would be visible

    def test_loss_has_no_trainable_or_fitted_state(self):
        loss = twc.make_task_weighted_corn_loss(CLASS_WEIGHTS, twc.TREATMENT_TASK_WEIGHTS)
        logits, grades = _batch(2)
        first = float(loss(grades, logits).numpy())
        _ = loss(*_batch(3)[::-1])  # evaluating another batch must not change it
        self.assertEqual(first, float(loss(grades, logits).numpy()))
        self.assertFalse(getattr(loss, "trainable_variables", []))


if __name__ == "__main__":
    unittest.main()
