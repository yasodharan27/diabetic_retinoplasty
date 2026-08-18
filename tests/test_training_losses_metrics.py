"""
Regression tests for the Stage 04 Experiment 2A fix: `training.losses.
dice_loss`/`bce_dice_loss` and `training.metrics.dice_coefficient`/
`iou_score` must compute Dice/IoU independently per channel for a
multi-channel `(batch, H, W, C)` tensor and then average across channels,
rather than pooling every channel's pixels together first. Pooling let a
channel with a much larger foreground area dominate a much smaller
channel's contribution to both the loss and the reported metric -- the
confirmed root cause of Stage 04's severe per-class Dice imbalance
(Microaneurysm/Haemorrhage/SoftExudate near zero while HardExudate alone
was learned).

All data here is small and synthetic, hand-computed by hand-worked Dice/IoU
arithmetic -- no project performance metric (baseline or otherwise) is
asserted or reproduced.
"""

import unittest

import numpy as np
import tensorflow as tf

from training.losses import bce_dice_loss, dice_loss, weighted_bce_dice_loss
from training.metrics import dice_coefficient, iou_score


def _two_channel_example():
    """(1, 4, 4, 2) y_true/y_pred where channel 0 is a perfect prediction
    (16/16 foreground pixels, fully overlapping) and channel 1 is a
    completely disjoint single-pixel prediction (1 true, 1 predicted, zero
    overlap). Hand-computed (smooth=1.0):

      channel 0: intersection=16, union=16+16=32 -> dice=(2*16+1)/33=1.0
                                                   -> iou=(16+1)/(16+1)=1.0
      channel 1: intersection=0,  union=1+1=2     -> dice=(0+1)/3=1/3
                                                   -> iou=(0+1)/(2-0+1)=1/3

      correct per-channel-averaged dice/iou = mean(1.0, 1/3) = 2/3

    A pooled (channel-mixing) computation would instead give, over all 32
    combined elements: intersection=16, union=17+17=34 ->
    dice=(2*16+1)/35=33/35 (~0.943) -- a very different, wrong number this
    test must NOT match.
    """
    y_true = np.zeros((1, 4, 4, 2), dtype=np.float32)
    y_pred = np.zeros((1, 4, 4, 2), dtype=np.float32)
    y_true[0, :, :, 0] = 1.0
    y_pred[0, :, :, 0] = 1.0
    y_true[0, 0, 0, 1] = 1.0
    y_pred[0, 3, 3, 1] = 1.0
    return y_true, y_pred


class DiceCoefficientChannelIndependenceTests(unittest.TestCase):
    def test_multichannel_result_is_mean_of_per_channel_scores(self):
        y_true, y_pred = _two_channel_example()
        result = dice_coefficient(y_true, y_pred).numpy()
        expected_per_channel_mean = (1.0 + 1.0 / 3.0) / 2.0  # 2/3
        pooled_wrong_value = 33.0 / 35.0

        self.assertEqual(result.shape, (1,))
        self.assertAlmostEqual(float(result[0]), expected_per_channel_mean, places=5)
        self.assertNotAlmostEqual(float(result[0]), pooled_wrong_value, places=2)

    def test_matches_manually_sliced_per_channel_average(self):
        """The multi-channel result must equal the mean of calling
        dice_coefficient independently on each already-sliced channel --
        exactly the definition of "per channel, then averaged"."""
        y_true, y_pred = _two_channel_example()
        channel_0 = float(dice_coefficient(y_true[..., 0], y_pred[..., 0]).numpy())
        channel_1 = float(dice_coefficient(y_true[..., 1], y_pred[..., 1]).numpy())
        manual_mean = (channel_0 + channel_1) / 2.0

        result = float(dice_coefficient(y_true, y_pred).numpy()[0])
        self.assertAlmostEqual(result, manual_mean, places=5)

    def test_per_batch_element_shape_and_independence(self):
        """A second batch element where both channels are perfect must
        score independently of the first (imbalanced) example."""
        y_true_1, y_pred_1 = _two_channel_example()
        y_true_2 = np.ones((1, 4, 4, 2), dtype=np.float32)
        y_pred_2 = np.ones((1, 4, 4, 2), dtype=np.float32)

        y_true = np.concatenate([y_true_1, y_true_2], axis=0)
        y_pred = np.concatenate([y_pred_1, y_pred_2], axis=0)

        result = dice_coefficient(y_true, y_pred).numpy()
        self.assertEqual(result.shape, (2,))
        self.assertAlmostEqual(float(result[0]), 2.0 / 3.0, places=5)
        self.assertAlmostEqual(float(result[1]), 1.0, places=5)


class IouScoreChannelIndependenceTests(unittest.TestCase):
    def test_multichannel_result_is_mean_of_per_channel_scores(self):
        y_true, y_pred = _two_channel_example()
        result = iou_score(y_true, y_pred).numpy()
        expected_per_channel_mean = (1.0 + 1.0 / 3.0) / 2.0  # 2/3

        self.assertEqual(result.shape, (1,))
        self.assertAlmostEqual(float(result[0]), expected_per_channel_mean, places=5)

    def test_matches_manually_sliced_per_channel_average(self):
        y_true, y_pred = _two_channel_example()
        channel_0 = float(iou_score(y_true[..., 0], y_pred[..., 0]).numpy())
        channel_1 = float(iou_score(y_true[..., 1], y_pred[..., 1]).numpy())
        manual_mean = (channel_0 + channel_1) / 2.0

        result = float(iou_score(y_true, y_pred).numpy()[0])
        self.assertAlmostEqual(result, manual_mean, places=5)


class SingleChannelBehaviorUnchangedTests(unittest.TestCase):
    """Rank < 4 tensors (already single-channel, e.g. a pre-sliced (batch,
    H, W) mask, or a plain 2D binary mask) have no channel axis to average
    over -- both metrics must fall back to a single fully-pooled scalar,
    exactly as before this change."""

    def test_dice_coefficient_rank3_matches_hand_computed_pooled_value(self):
        y_true = np.zeros((1, 4, 4), dtype=np.float32)
        y_pred = np.zeros((1, 4, 4), dtype=np.float32)
        y_true[0, 0:2, 0:2] = 1.0  # 4 foreground pixels
        y_pred[0, 0:2, 0:2] = 1.0  # identical -> perfect overlap

        result = float(dice_coefficient(y_true, y_pred).numpy())
        expected = (2.0 * 4 + 1.0) / (4 + 4 + 1.0)  # 9/9 = 1.0
        self.assertAlmostEqual(result, expected, places=5)
        self.assertAlmostEqual(result, 1.0, places=5)

    def test_iou_score_rank2_matches_hand_computed_pooled_value(self):
        y_true = np.array([[1.0, 0.0], [0.0, 1.0]], dtype=np.float32)
        y_pred = np.array([[1.0, 1.0], [0.0, 0.0]], dtype=np.float32)
        # intersection=1 (top-left), union=2+2-1=3
        result = float(iou_score(y_true, y_pred).numpy())
        expected = (1.0 + 1.0) / (3.0 + 1.0)  # 2/4 = 0.5
        self.assertAlmostEqual(result, expected, places=5)

    def test_dice_loss_rank3_is_one_minus_dice_coefficient(self):
        y_true = np.zeros((1, 4, 4), dtype=np.float32)
        y_pred = np.zeros((1, 4, 4), dtype=np.float32)
        y_true[0, 0, 0] = 1.0
        y_pred[0, 3, 3] = 1.0  # completely disjoint

        loss_fn = dice_loss()
        loss_value = float(loss_fn(y_true, y_pred).numpy()[0])
        dice_value = float(dice_coefficient(y_true, y_pred).numpy())
        self.assertAlmostEqual(loss_value, 1.0 - dice_value, places=5)


class DiceLossChannelIndependenceTests(unittest.TestCase):
    def test_multichannel_loss_is_one_minus_mean_of_per_channel_dice(self):
        y_true, y_pred = _two_channel_example()
        loss_fn = dice_loss()
        loss_value = loss_fn(y_true, y_pred).numpy()

        expected_dice_mean = (1.0 + 1.0 / 3.0) / 2.0
        self.assertEqual(loss_value.shape, (1,))
        self.assertAlmostEqual(float(loss_value[0]), 1.0 - expected_dice_mean, places=5)


class BceDiceLossKerasCompatibilityTests(unittest.TestCase):
    """bce_dice_loss() must remain usable exactly as a Keras `model.compile
    (loss=...)` argument -- finite, broadcasts correctly against BCE's
    scalar reduction, and drives a real (tiny, synthetic) training step
    without error."""

    def test_returns_finite_value_for_multichannel_input(self):
        y_true, y_pred = _two_channel_example()
        loss_fn = bce_dice_loss()
        value = loss_fn(y_true, y_pred).numpy()
        self.assertTrue(np.all(np.isfinite(value)))

    def test_compiles_and_fits_a_tiny_multichannel_model(self):
        inputs = tf.keras.Input(shape=(8, 8, 4))
        outputs = tf.keras.layers.Conv2D(4, 1, activation="sigmoid")(inputs)
        model = tf.keras.Model(inputs, outputs)
        model.compile(optimizer="adam", loss=bce_dice_loss(),
                      metrics=[dice_coefficient, iou_score])

        rng = np.random.RandomState(0)
        x = rng.rand(4, 8, 8, 4).astype("float32")
        y = (rng.rand(4, 8, 8, 4) > 0.5).astype("float32")

        history = model.fit(x, y, epochs=1, batch_size=2, verbose=0)
        self.assertIn("loss", history.history)
        self.assertTrue(np.isfinite(history.history["loss"][0]))


def _four_channel_weighting_example():
    """(1, 4, 4, 4) y_true/y_pred with four deliberately distinct per-channel
    prediction qualities, soft (non-binary) probabilities throughout so both
    the BCE and Dice components vary meaningfully per channel:

      channel 0: y_true all-foreground, y_pred=0.9 everywhere  (best)
      channel 1: y_true a 2x2 foreground block, y_pred=0.5 everywhere
      channel 2: y_true all-background,        y_pred=0.1 everywhere
      channel 3: y_true a single foreground pixel, y_pred=0.01 everywhere (worst)

    Distinct per-channel quality means a channel-to-weight mapping bug
    (e.g. weights applied to the wrong channel, or misaligned after a
    reshape) is very likely to change which channel dominates the result.
    """
    y_true = np.zeros((1, 4, 4, 4), dtype=np.float32)
    y_pred = np.zeros((1, 4, 4, 4), dtype=np.float32)

    y_true[0, :, :, 0] = 1.0
    y_pred[0, :, :, 0] = 0.9

    y_true[0, 0:2, 0:2, 1] = 1.0
    y_pred[0, :, :, 1] = 0.5

    y_pred[0, :, :, 2] = 0.1  # y_true channel 2 stays all-background

    y_true[0, 0, 0, 3] = 1.0
    y_pred[0, :, :, 3] = 0.01

    return y_true, y_pred


def _independent_weighted_reference(y_true, y_pred, weights):
    """Ground truth for `weighted_bce_dice_loss`, computed independently of
    the implementation under test: per channel, a plain manual BCE mean
    (its own formula, not `weighted_bce_dice_loss`'s) and the already-
    verified `dice_coefficient` (from `DiceCoefficientChannelIndependenceTests`
    above) sliced to that single channel, then combined via the spec's own
    weighted-average formula: sum(w_i * loss_i) / sum(w_i)."""
    weights = np.asarray(weights, dtype=np.float64)
    num_channels = y_true.shape[-1]
    epsilon = 1e-7

    bce_per_channel = []
    dice_loss_per_channel = []
    for c in range(num_channels):
        yt = y_true[..., c].astype(np.float64)
        yp = np.clip(y_pred[..., c].astype(np.float64), epsilon, 1 - epsilon)
        bce = -(yt * np.log(yp) + (1 - yt) * np.log(1 - yp))
        bce_per_channel.append(bce.mean())

        dice = float(dice_coefficient(y_true[..., c], y_pred[..., c]).numpy())
        dice_loss_per_channel.append(1.0 - dice)

    weighted_bce = float(np.sum(np.array(bce_per_channel) * weights) / weights.sum())
    weighted_dice_loss = float(np.sum(np.array(dice_loss_per_channel) * weights) / weights.sum())
    return 0.5 * weighted_bce + 0.5 * weighted_dice_loss


class WeightedBceDiceLossTests(unittest.TestCase):
    """training.losses.weighted_bce_dice_loss() -- Stage 04 Experiment 2B."""

    def test_multichannel_input_returns_finite_value(self):
        y_true, y_pred = _four_channel_weighting_example()
        loss_fn = weighted_bce_dice_loss([2.0, 1.0, 1.1, 1.8])
        value = loss_fn(y_true, y_pred).numpy()
        self.assertEqual(value.shape, (1,))
        self.assertTrue(np.all(np.isfinite(value)))

    def test_matches_independent_reference_implementation(self):
        """Correct channel-to-weight mapping: the implementation's output
        must match a from-scratch, independently-written reference that
        applies the same weights to the same channels via the spec's
        weighted-average formula."""
        y_true, y_pred = _four_channel_weighting_example()
        weights = [2.0, 1.0, 1.1, 1.8]
        loss_fn = weighted_bce_dice_loss(weights)
        actual = float(loss_fn(y_true, y_pred).numpy()[0])
        expected = _independent_weighted_reference(y_true, y_pred, weights)
        self.assertAlmostEqual(actual, expected, places=4)

    def test_uniform_weights_reduce_to_plain_bce_dice_loss(self):
        """Normalization sanity check: sum(w_i*loss_i)/sum(w_i) with all
        weights equal must reproduce the plain, unweighted per-channel
        average that bce_dice_loss() already computes (Experiment 2A)."""
        y_true, y_pred = _four_channel_weighting_example()
        weighted = float(weighted_bce_dice_loss([1.0, 1.0, 1.0, 1.0])(y_true, y_pred).numpy()[0])
        unweighted = float(bce_dice_loss()(y_true, y_pred).numpy()[0])
        self.assertAlmostEqual(weighted, unweighted, places=5)

    def test_uniformly_scaling_all_weights_leaves_loss_unchanged(self):
        """Normalization of weights: multiplying every weight by the same
        constant must not change the result (dividing by sum(weights)
        cancels the scale factor out)."""
        y_true, y_pred = _four_channel_weighting_example()
        base_weights = [2.0, 1.0, 1.1, 1.8]
        scaled_weights = [10.0 * w for w in base_weights]
        base = float(weighted_bce_dice_loss(base_weights)(y_true, y_pred).numpy()[0])
        scaled = float(weighted_bce_dice_loss(scaled_weights)(y_true, y_pred).numpy()[0])
        self.assertAlmostEqual(base, scaled, places=5)

    def test_changing_one_channels_weight_changes_the_loss(self):
        y_true, y_pred = _four_channel_weighting_example()
        original = float(weighted_bce_dice_loss([1.0, 1.0, 1.0, 1.0])(y_true, y_pred).numpy()[0])
        channel_3_upweighted = float(
            weighted_bce_dice_loss([1.0, 1.0, 1.0, 20.0])(y_true, y_pred).numpy()[0]
        )
        self.assertNotAlmostEqual(original, channel_3_upweighted, places=3)

    def test_upweighting_the_worst_channel_increases_loss_more_than_the_best_channel(self):
        """A second, direction-sensitive check on channel-to-weight mapping:
        channel 3 (worst prediction quality) and channel 0 (best) must not
        be interchangeable -- concentrating weight on the worst channel must
        raise the total loss above concentrating the same weight on the
        best channel."""
        y_true, y_pred = _four_channel_weighting_example()
        weight_on_best = float(
            weighted_bce_dice_loss([20.0, 1.0, 1.0, 1.0])(y_true, y_pred).numpy()[0]
        )
        weight_on_worst = float(
            weighted_bce_dice_loss([1.0, 1.0, 1.0, 20.0])(y_true, y_pred).numpy()[0]
        )
        self.assertGreater(weight_on_worst, weight_on_best)

    def test_keras_model_compiles_and_fits_one_step(self):
        inputs = tf.keras.Input(shape=(8, 8, 4))
        outputs = tf.keras.layers.Conv2D(4, 1, activation="sigmoid")(inputs)
        model = tf.keras.Model(inputs, outputs)
        model.compile(optimizer="adam", loss=weighted_bce_dice_loss([2.0, 1.0, 1.1, 1.8]),
                      metrics=[dice_coefficient, iou_score])

        rng = np.random.RandomState(1)
        x = rng.rand(4, 8, 8, 4).astype("float32")
        y = (rng.rand(4, 8, 8, 4) > 0.5).astype("float32")

        history = model.fit(x, y, epochs=1, batch_size=2, verbose=0)
        self.assertIn("loss", history.history)
        self.assertTrue(np.isfinite(history.history["loss"][0]))

    def test_existing_unweighted_losses_are_unaffected(self):
        """dice_loss()/bce_dice_loss() must still behave exactly as
        Experiment 2A left them -- adding weighted_bce_dice_loss must not
        have perturbed either."""
        y_true, y_pred = _two_channel_example()
        expected_dice_mean = (1.0 + 1.0 / 3.0) / 2.0

        loss_value = float(dice_loss()(y_true, y_pred).numpy()[0])
        self.assertAlmostEqual(loss_value, 1.0 - expected_dice_mean, places=5)

        combined = bce_dice_loss()(y_true, y_pred).numpy()
        self.assertTrue(np.all(np.isfinite(combined)))

    def test_existing_single_channel_dice_iou_are_unaffected(self):
        """dice_coefficient()/iou_score() single-channel (rank < 4) behavior
        must still be the fully-pooled scalar from before Experiment 2A --
        unaffected by adding weighted_bce_dice_loss."""
        y_true = np.zeros((1, 4, 4), dtype=np.float32)
        y_pred = np.zeros((1, 4, 4), dtype=np.float32)
        y_true[0, 0:2, 0:2] = 1.0
        y_pred[0, 0:2, 0:2] = 1.0

        self.assertAlmostEqual(float(dice_coefficient(y_true, y_pred).numpy()), 1.0, places=5)
        self.assertAlmostEqual(float(iou_score(y_true, y_pred).numpy()), 1.0, places=5)


if __name__ == "__main__":
    unittest.main()
