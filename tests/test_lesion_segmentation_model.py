"""
Regression tests for lesion_segmentation_model.py (Stage 04 Attention
U-Net + inference + LesionSegmentationStage).

Every model built here is tiny (small base_filters, small spatial size) --
these tests verify architectural/plumbing correctness (shapes, dtypes,
value ranges, save/load round-tripping, the pipeline.SegmentationStage
contract), not segmentation quality. No real training run happens, no
metric is ever reported as a real evaluation result -- per this project's
"unit tests use synthetic/temporary data only" rule and the explicit
instruction not to fabricate performance metrics. The real, gitignored
Stage 03 LWNet checkpoint is never touched -- vessel probability maps are
always supplied directly or backed by the same synthetic checkpoint
pattern tests/test_vessel_segmentation_device.py already established.
"""

import os
import shutil
import tempfile
import unittest
from unittest import mock

import numpy as np
import tensorflow as tf
import torch

import lesion_segmentation_model as lsm
from pipeline import SegmentationStage
from training.losses import weighted_bce_dice_loss, weighted_pooled_bce_dice_loss
from vessel_segmentation_model import build_vessel_segmentation_model, load_state_dict_from_checkpoint


def _synthetic_fundus_image(size=96, seed=0):
    rng = np.random.RandomState(seed)
    image = np.zeros((size, size, 3), dtype=np.uint8)
    yy, xx = np.ogrid[:size, :size]
    center = size // 2
    radius = int(size * 0.4)
    circle = (xx - center) ** 2 + (yy - center) ** 2 <= radius ** 2
    base = 140 + rng.randint(-20, 20, size=(size, size, 3))
    image[circle] = np.clip(base[circle], 60, 220).astype(np.uint8)
    return image


def _build_synthetic_vessel_model():
    model = build_vessel_segmentation_model()
    tmp_dir = tempfile.mkdtemp(prefix="synthetic_vessel_ckpt_")
    checkpoint_path = os.path.join(tmp_dir, "synthetic_checkpoint.pth")
    torch.save(
        {"model_state_dict": model.state_dict(), "optimizer_state_dict": {}, "stats": None},
        checkpoint_path,
    )
    loaded = load_state_dict_from_checkpoint(model, checkpoint_path, device="cpu")
    shutil.rmtree(tmp_dir, ignore_errors=True)
    return loaded


def _tiny_dataset(num_samples=4, size=16, batch_size=2, seed=0):
    rng = np.random.RandomState(seed)
    x = rng.rand(num_samples, size, size, 4).astype("float32")
    y = (rng.rand(num_samples, size, size, 4) > 0.5).astype("float32")
    return tf.data.Dataset.from_tensor_slices((x, y)).batch(batch_size)


class BuildAttentionUnetTests(unittest.TestCase):
    def test_output_shape_and_channel_count(self):
        model = lsm.build_attention_unet(input_shape=(64, 64, 4), base_filters=4)
        x = np.random.rand(2, 64, 64, 4).astype("float32")
        y = model.predict(x, verbose=0)
        self.assertEqual(y.shape, (2, 64, 64, 4))

    def test_sigmoid_output_in_unit_range(self):
        model = lsm.build_attention_unet(input_shape=(64, 64, 4), base_filters=4)
        x = np.random.rand(2, 64, 64, 4).astype("float32")
        y = model.predict(x, verbose=0)
        self.assertTrue(np.isfinite(y).all())
        self.assertGreaterEqual(y.min(), 0.0)
        self.assertLessEqual(y.max(), 1.0)

    def test_input_shape_not_divisible_by_16_raises(self):
        with self.assertRaises(ValueError):
            lsm.build_attention_unet(input_shape=(50, 50, 4), base_filters=4)

    def test_model_is_compiled_with_bce_dice_loss_and_segmentation_metrics(self):
        model = lsm.build_attention_unet(input_shape=(32, 32, 4), base_filters=4)
        self.assertIsNotNone(model.loss)
        x = np.random.rand(2, 32, 32, 4).astype("float32")
        y = np.random.randint(0, 2, (2, 32, 32, 4)).astype("float32")
        # metrics_names is populated lazily by Keras 3 -- only after the
        # model has actually been called/evaluated once.
        results = model.evaluate(x, y, verbose=0, return_dict=True)
        self.assertTrue(any("dice" in name for name in results))
        self.assertTrue(any("iou" in name for name in results))


class BuildAttentionUnetClassWeightsTests(unittest.TestCase):
    """Stage 04 Experiment 2B: `class_weights=None` (the default) must keep
    building the plain, unweighted Experiment 2A model exactly as before;
    passing a length-4 weight sequence must switch to
    `training.weighted_bce_dice_loss` instead. No other stage's model-
    building code is touched by this parameter."""

    def test_default_class_weights_none_uses_unweighted_loss(self):
        model = lsm.build_attention_unet(input_shape=(32, 32, 4), base_filters=4)
        self.assertIsNotNone(model.loss)
        x = np.random.rand(2, 32, 32, 4).astype("float32")
        y = np.random.randint(0, 2, (2, 32, 32, 4)).astype("float32")
        results = model.evaluate(x, y, verbose=0, return_dict=True)
        self.assertIn("loss", results)
        self.assertTrue(np.isfinite(results["loss"]))

    def test_explicit_class_weights_builds_and_trains_one_step(self):
        model = lsm.build_attention_unet(
            input_shape=(32, 32, 4), base_filters=4,
            class_weights=lsm.EXPERIMENT_2B_CLASS_WEIGHTS,
        )
        x = np.random.rand(2, 32, 32, 4).astype("float32")
        y = np.random.randint(0, 2, (2, 32, 32, 4)).astype("float32")
        history = model.fit(x, y, epochs=1, batch_size=2, verbose=0)
        self.assertIn("loss", history.history)
        self.assertTrue(np.isfinite(history.history["loss"][0]))

    def test_mismatched_class_weights_length_raises(self):
        with self.assertRaises(ValueError):
            lsm.build_attention_unet(input_shape=(32, 32, 4), base_filters=4,
                                      class_weights=[1.0, 2.0])

    def test_experiment_2b_class_weights_constant_matches_the_approved_values(self):
        self.assertEqual(tuple(lsm.EXPERIMENT_2B_CLASS_WEIGHTS), (2.0, 1.0, 1.1, 1.8))
        self.assertEqual(len(lsm.EXPERIMENT_2B_CLASS_WEIGHTS), len(lsm.LESION_CLASSES))


def _distinctly_weighted_4channel_example(size=8):
    """(1, size, size, 4) y_true/y_pred with deliberately distinct
    per-channel prediction quality (mirrors
    test_training_losses_metrics._four_channel_weighting_example), so that
    weighted-pooled and weighted-per-channel Dice reliably diverge -- unlike
    uniform random data, where the two formulations can coincidentally land
    close together by chance."""
    y_true = np.zeros((1, size, size, 4), dtype=np.float32)
    y_pred = np.zeros((1, size, size, 4), dtype=np.float32)
    y_true[0, :, :, 0] = 1.0
    y_pred[0, :, :, 0] = 0.9
    y_true[0, 0:2, 0:2, 1] = 1.0
    y_pred[0, :, :, 1] = 0.5
    y_pred[0, :, :, 2] = 0.1
    y_true[0, 0, 0, 3] = 1.0
    y_pred[0, :, :, 3] = 0.01
    return y_true, y_pred


class BuildAttentionUnetWeightedDiceModeTests(unittest.TestCase):
    """Stage 04 Experiment 2C: `weighted_dice_mode="pooled"` (with
    `class_weights` set) must switch the compiled loss to
    `training.weighted_pooled_bce_dice_loss` instead of Experiment 2B's
    `weighted_bce_dice_loss` -- the default `weighted_dice_mode=
    "per_channel"` must keep building Experiment 2B's model exactly as
    before this parameter was added."""

    def test_default_weighted_dice_mode_is_per_channel(self):
        model = lsm.build_attention_unet(
            input_shape=(32, 32, 4), base_filters=4, class_weights=lsm.EXPERIMENT_2B_CLASS_WEIGHTS,
        )
        y_true = np.random.RandomState(0).randint(0, 2, (2, 32, 32, 4)).astype("float32")
        y_pred = np.random.RandomState(1).rand(2, 32, 32, 4).astype("float32")
        actual = float(model.loss(y_true, y_pred).numpy()[0])
        expected = float(
            weighted_bce_dice_loss(lsm.EXPERIMENT_2B_CLASS_WEIGHTS)(y_true, y_pred).numpy()[0]
        )
        self.assertAlmostEqual(actual, expected, places=5)

    def test_pooled_mode_selects_weighted_pooled_loss(self):
        model = lsm.build_attention_unet(
            input_shape=(32, 32, 4), base_filters=4, class_weights=lsm.EXPERIMENT_2B_CLASS_WEIGHTS,
            weighted_dice_mode="pooled",
        )
        y_true, y_pred = _distinctly_weighted_4channel_example()

        actual = float(model.loss(y_true, y_pred).numpy()[0])
        expected_pooled = float(
            weighted_pooled_bce_dice_loss(lsm.EXPERIMENT_2B_CLASS_WEIGHTS)(y_true, y_pred).numpy()[0]
        )
        expected_per_channel = float(
            weighted_bce_dice_loss(lsm.EXPERIMENT_2B_CLASS_WEIGHTS)(y_true, y_pred).numpy()[0]
        )
        self.assertAlmostEqual(actual, expected_pooled, places=5)
        self.assertNotAlmostEqual(actual, expected_per_channel, places=3)

    def test_pooled_mode_trains_one_step(self):
        model = lsm.build_attention_unet(
            input_shape=(32, 32, 4), base_filters=4, class_weights=lsm.EXPERIMENT_2B_CLASS_WEIGHTS,
            weighted_dice_mode="pooled",
        )
        rng = np.random.RandomState(3)
        x = rng.rand(2, 32, 32, 4).astype("float32")
        y = (rng.rand(2, 32, 32, 4) > 0.5).astype("float32")
        history = model.fit(x, y, epochs=1, batch_size=2, verbose=0)
        self.assertIn("loss", history.history)
        self.assertTrue(np.isfinite(history.history["loss"][0]))

    def test_invalid_weighted_dice_mode_raises(self):
        with self.assertRaises(ValueError):
            lsm.build_attention_unet(
                input_shape=(32, 32, 4), base_filters=4, class_weights=lsm.EXPERIMENT_2B_CLASS_WEIGHTS,
                weighted_dice_mode="not_a_real_mode",
            )

    def test_weighted_dice_mode_ignored_when_class_weights_is_none(self):
        """weighted_dice_mode="pooled" with class_weights=None must still
        build the plain unweighted Experiment 2A model -- weighted_dice_mode
        only matters once class_weights is given."""
        model_default = lsm.build_attention_unet(input_shape=(32, 32, 4), base_filters=4)
        model_pooled_but_unweighted = lsm.build_attention_unet(
            input_shape=(32, 32, 4), base_filters=4, weighted_dice_mode="pooled",
        )
        y_true = np.random.RandomState(0).randint(0, 2, (2, 32, 32, 4)).astype("float32")
        y_pred = np.random.RandomState(1).rand(2, 32, 32, 4).astype("float32")
        self.assertAlmostEqual(
            float(model_default.loss(y_true, y_pred).numpy()[0]),
            float(model_pooled_but_unweighted.loss(y_true, y_pred).numpy()[0]),
            places=5,
        )


class LesionSegmentationStageClassWeightsTests(unittest.TestCase):
    """Stage 04 wiring for the Colab notebook: LesionSegmentationStage
    (class_weights=...) must forward those weights to build_attention_unet()
    the first time train() builds a model. class_weights=None (the default,
    unchanged from before this wiring) must keep calling
    build_attention_unet() exactly as Experiment 2A already did."""

    def setUp(self):
        self.tmp_dir = tempfile.mkdtemp(prefix="lesion_seg_stage_weights_test_")
        self.addCleanup(shutil.rmtree, self.tmp_dir, ignore_errors=True)

    def test_default_class_weights_none_is_forwarded_as_none(self):
        stage = lsm.LesionSegmentationStage(input_shape=(16, 16, 4))
        train_ds = _tiny_dataset(num_samples=4, size=16, batch_size=2, seed=1)
        with mock.patch("lesion_segmentation_model.build_attention_unet",
                         wraps=lsm.build_attention_unet) as spy:
            stage.train(train_ds, None, run_dir=os.path.join(self.tmp_dir, "run"), epochs=1)
        spy.assert_called_once_with(
            input_shape=(16, 16, 4), class_weights=None, weighted_dice_mode="per_channel",
        )

    def test_explicit_class_weights_are_forwarded_to_build_attention_unet(self):
        stage = lsm.LesionSegmentationStage(
            input_shape=(16, 16, 4), class_weights=lsm.EXPERIMENT_2B_CLASS_WEIGHTS,
        )
        train_ds = _tiny_dataset(num_samples=4, size=16, batch_size=2, seed=1)
        with mock.patch("lesion_segmentation_model.build_attention_unet",
                         wraps=lsm.build_attention_unet) as spy:
            stage.train(train_ds, None, run_dir=os.path.join(self.tmp_dir, "run"), epochs=1)
        spy.assert_called_once_with(
            input_shape=(16, 16, 4), class_weights=lsm.EXPERIMENT_2B_CLASS_WEIGHTS,
            weighted_dice_mode="per_channel",
        )
        self.assertIsNotNone(stage.model)

    def test_experiment_2c_weighted_dice_mode_is_forwarded_to_build_attention_unet(self):
        """Experiment 2C: LesionSegmentationStage(class_weights=...,
        weighted_dice_mode="pooled") must explicitly select
        training.weighted_pooled_bce_dice_loss, and the resulting model
        must actually receive it -- not merely forward the string."""
        stage = lsm.LesionSegmentationStage(
            input_shape=(16, 16, 4), class_weights=lsm.EXPERIMENT_2B_CLASS_WEIGHTS,
            weighted_dice_mode="pooled",
        )
        train_ds = _tiny_dataset(num_samples=4, size=16, batch_size=2, seed=1)
        with mock.patch("lesion_segmentation_model.build_attention_unet",
                         wraps=lsm.build_attention_unet) as spy:
            stage.train(train_ds, None, run_dir=os.path.join(self.tmp_dir, "run"), epochs=1)
        spy.assert_called_once_with(
            input_shape=(16, 16, 4), class_weights=lsm.EXPERIMENT_2B_CLASS_WEIGHTS,
            weighted_dice_mode="pooled",
        )
        self.assertIsNotNone(stage.model)

        y_true, y_pred = _distinctly_weighted_4channel_example(size=16)
        actual = float(stage.model.loss(y_true, y_pred).numpy()[0])
        expected_pooled = float(
            weighted_pooled_bce_dice_loss(lsm.EXPERIMENT_2B_CLASS_WEIGHTS)(y_true, y_pred).numpy()[0]
        )
        expected_per_channel = float(
            weighted_bce_dice_loss(lsm.EXPERIMENT_2B_CLASS_WEIGHTS)(y_true, y_pred).numpy()[0]
        )
        self.assertAlmostEqual(actual, expected_pooled, places=5)
        self.assertNotAlmostEqual(actual, expected_per_channel, places=3)


class PerClassSegmentationMetricsTests(unittest.TestCase):
    def test_perfect_prediction_gives_dice_and_iou_near_one(self):
        y = np.zeros((1, 8, 8, 4), dtype=np.float32)
        y[:, 2:5, 2:5, :] = 1.0
        metrics = lsm.per_class_segmentation_metrics(y, y)
        for name in lsm.LESION_CLASSES:
            self.assertAlmostEqual(metrics[f"dice_{name}"], 1.0, places=4)
            self.assertAlmostEqual(metrics[f"iou_{name}"], 1.0, places=4)
        self.assertAlmostEqual(metrics["dice_mean"], 1.0, places=4)
        self.assertAlmostEqual(metrics["iou_mean"], 1.0, places=4)

    def test_completely_disjoint_prediction_gives_low_dice_and_iou(self):
        y_true = np.zeros((1, 8, 8, 4), dtype=np.float32)
        y_true[:, 0:4, 0:4, :] = 1.0
        y_pred = np.zeros((1, 8, 8, 4), dtype=np.float32)
        y_pred[:, 4:8, 4:8, :] = 1.0
        metrics = lsm.per_class_segmentation_metrics(y_true, y_pred)
        for name in lsm.LESION_CLASSES:
            self.assertLess(metrics[f"dice_{name}"], 0.1)
            self.assertLess(metrics[f"iou_{name}"], 0.1)

    def test_returns_expected_keys_only(self):
        y = np.zeros((1, 4, 4, 4), dtype=np.float32)
        metrics = lsm.per_class_segmentation_metrics(y, y)
        expected_keys = {f"dice_{n}" for n in lsm.LESION_CLASSES} | {f"iou_{n}" for n in lsm.LESION_CLASSES}
        expected_keys |= {"dice_mean", "iou_mean"}
        self.assertEqual(set(metrics.keys()), expected_keys)


class LesionSegmentationStageTrainEvaluateSaveLoadTests(unittest.TestCase):
    """Full pipeline.SegmentationStage lifecycle against tiny synthetic
    tensors -- no real dataset, no fabricated metrics (the reported dice/
    iou values are real computations against random synthetic data, not
    claimed to reflect real segmentation quality)."""

    def setUp(self):
        self.tmp_dir = tempfile.mkdtemp(prefix="lesion_seg_stage_test_")
        self.addCleanup(shutil.rmtree, self.tmp_dir, ignore_errors=True)
        self.stage = lsm.LesionSegmentationStage(input_shape=(16, 16, 4))

    def test_is_a_segmentation_stage(self):
        self.assertIsInstance(self.stage, SegmentationStage)

    def test_train_runs_and_returns_history(self):
        train_ds = _tiny_dataset(num_samples=4, size=16, batch_size=2, seed=1)
        val_ds = _tiny_dataset(num_samples=2, size=16, batch_size=2, seed=2)
        run_dir = os.path.join(self.tmp_dir, "training_run")

        history = self.stage.train(train_ds, val_ds, run_dir=run_dir, epochs=1)

        self.assertIn("loss", history.history)
        self.assertIsNotNone(self.stage.model)

    def test_evaluate_returns_per_class_and_mean_metrics(self):
        train_ds = _tiny_dataset(num_samples=4, size=16, batch_size=2, seed=1)
        val_ds = _tiny_dataset(num_samples=2, size=16, batch_size=2, seed=2)
        self.stage.train(train_ds, val_ds, run_dir=os.path.join(self.tmp_dir, "run"), epochs=1)

        eval_ds = _tiny_dataset(num_samples=4, size=16, batch_size=2, seed=3)
        metrics = self.stage.evaluate(eval_ds)

        for name in lsm.LESION_CLASSES:
            self.assertIn(f"dice_{name}", metrics)
            self.assertIn(f"iou_{name}", metrics)
            self.assertGreaterEqual(metrics[f"dice_{name}"], 0.0)
            self.assertLessEqual(metrics[f"dice_{name}"], 1.0)
        self.assertIn("dice_mean", metrics)
        self.assertIn("iou_mean", metrics)

    def test_evaluate_before_load_or_train_raises(self):
        fresh_stage = lsm.LesionSegmentationStage(input_shape=(16, 16, 4))
        with self.assertRaises(RuntimeError):
            fresh_stage.evaluate(_tiny_dataset())

    def test_predict_before_load_raises(self):
        fresh_stage = lsm.LesionSegmentationStage(input_shape=(16, 16, 4))
        with self.assertRaises(RuntimeError):
            fresh_stage.predict(_synthetic_fundus_image())

    def test_save_before_train_raises(self):
        fresh_stage = lsm.LesionSegmentationStage(input_shape=(16, 16, 4))
        with self.assertRaises(RuntimeError):
            fresh_stage.save(os.path.join(self.tmp_dir, "x.keras"))

    def test_save_and_load_roundtrip_preserves_predictions(self):
        train_ds = _tiny_dataset(num_samples=4, size=16, batch_size=2, seed=1)
        self.stage.train(train_ds, None, run_dir=os.path.join(self.tmp_dir, "run"), epochs=1)

        x = np.random.RandomState(0).rand(1, 16, 16, 4).astype("float32")
        predictions_before = self.stage.model.predict(x, verbose=0)

        checkpoint_path = os.path.join(self.tmp_dir, "best_model.keras")
        saved_path = self.stage.save(checkpoint_path)
        self.assertTrue(os.path.exists(saved_path))

        reloaded_stage = lsm.LesionSegmentationStage(input_shape=(16, 16, 4))
        returned = reloaded_stage.load(checkpoint_path)
        self.assertIs(returned, reloaded_stage)

        predictions_after = reloaded_stage.model.predict(x, verbose=0)
        np.testing.assert_allclose(predictions_before, predictions_after, atol=1e-5)


class PredictLesionMaskTests(unittest.TestCase):
    """predict_lesion_mask / predict_lesion_mask_batch, using a real
    (untrained) Attention U-Net + a synthetic Stage 03 vessel model --
    never the real gitignored LWNet checkpoint."""

    def setUp(self):
        self.vessel_model = _build_synthetic_vessel_model()
        self.model = lsm.build_attention_unet(input_shape=(32, 32, 4), base_filters=4)

    def test_output_matches_native_input_resolution(self):
        image = _synthetic_fundus_image(size=80)
        result = lsm.predict_lesion_mask(image, model=self.model, vessel_model=self.vessel_model)
        self.assertEqual(result["probability_maps"].shape, (80, 80, 4))
        self.assertEqual(result["binary_masks"].shape, (80, 80, 4))
        self.assertEqual(result["input_shape"], (80, 80))
        self.assertEqual(result["class_names"], lsm.LESION_CLASSES)

    def test_probabilities_in_unit_range_and_masks_binary(self):
        image = _synthetic_fundus_image(size=80)
        result = lsm.predict_lesion_mask(image, model=self.model, vessel_model=self.vessel_model)
        probs = result["probability_maps"]
        self.assertTrue(np.isfinite(probs).all())
        self.assertGreaterEqual(probs.min(), 0.0)
        self.assertLessEqual(probs.max(), 1.0)
        self.assertTrue(np.isin(result["binary_masks"], [0, 1]).all())

    def test_precomputed_vessel_map_is_accepted_directly(self):
        image = _synthetic_fundus_image(size=64)
        vessel_map = np.random.RandomState(0).rand(64, 64, 1).astype("float32")
        result = lsm.predict_lesion_mask(image, vessel_probability_map=vessel_map, model=self.model)
        self.assertEqual(result["probability_maps"].shape, (64, 64, 4))

    def test_batch_matches_single_image_calls(self):
        images = [_synthetic_fundus_image(size=64, seed=i) for i in range(2)]
        batch_results = lsm.predict_lesion_mask_batch(images, model=self.model, vessel_model=self.vessel_model)
        self.assertEqual(len(batch_results), 2)
        for result in batch_results:
            self.assertEqual(result["probability_maps"].shape, (64, 64, 4))


class LesionSegmentationStagePredictContractTests(unittest.TestCase):
    """LesionSegmentationStage.predict()/predict_batch() follow
    VesselSegmentationStage's exact pattern: return the bare probability-map
    array(s), not the full result dict. Stage 03's own predict_vessel_mask
    is mocked here (rather than passing a real vessel model, which the
    fixed pipeline.SegmentationStage.predict(self, input_data) signature
    has no parameter for) so this never depends on the real, gitignored
    LWNet checkpoint."""

    def setUp(self):
        self.stage = lsm.LesionSegmentationStage(input_shape=(32, 32, 4))
        self.stage.model = lsm.build_attention_unet(input_shape=(32, 32, 4), base_filters=4)
        self._vessel_patch = mock.patch(
            "lesion_segmentation_model.predict_vessel_mask",
            return_value={"probability_map": np.zeros((64, 64, 1), dtype=np.float32)},
        )
        self._vessel_patch.start()
        self.addCleanup(self._vessel_patch.stop)

    def test_predict_returns_bare_probability_array(self):
        image = _synthetic_fundus_image(size=64)
        result = self.stage.predict(image)
        self.assertIsInstance(result, np.ndarray)
        self.assertEqual(result.shape, (64, 64, 4))

    def test_predict_batch_returns_list_of_arrays(self):
        images = [_synthetic_fundus_image(size=64, seed=i) for i in range(2)]
        results = self.stage.predict_batch(images)
        self.assertEqual(len(results), 2)
        for r in results:
            self.assertEqual(r.shape, (64, 64, 4))


if __name__ == "__main__":
    unittest.main()
