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

from training.losses import bce_dice_loss, dice_loss
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


if __name__ == "__main__":
    unittest.main()
