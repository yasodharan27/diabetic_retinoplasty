"""
Regression tests for local_feature_extraction_model.py (Stage 05 Adaptive
Multi-Kernel CNN + LocalFeatureExtractionStage).

Most models built here use small, custom `stage_filters`/`input_shape` for
speed (mirroring test_lesion_segmentation_model.py's "every model built
here is tiny" convention) -- except for one explicit test that builds the
real default (512, 512, 8) -> (32, 32, 256) contract, since that exact
shape is the requirement this module exists to satisfy. No training run
happens anywhere in this file, and no metric is ever reported as a real
evaluation result -- these are architectural/plumbing checks only.
"""

import os
import shutil
import tempfile
import unittest

import numpy as np
import tensorflow as tf

import local_feature_extraction_model as lfem
from pipeline import FeatureExtractionStage


class BuildLocalFeatureExtractorShapeTests(unittest.TestCase):
    def test_small_custom_configuration_output_shape(self):
        model = lfem.build_local_feature_extractor(
            input_shape=(64, 64, 8), stage_filters=(4, 8, 16, 32),
        )
        x = np.random.rand(2, 64, 64, 8).astype("float32")
        y = model.predict(x, verbose=0)
        self.assertEqual(y.shape, (2, 4, 4, 32))

    def test_default_configuration_matches_approved_output_contract(self):
        """The literal (B, 512, 512, 8) -> (B, 32, 32, 256) contract from
        the approved Stage 05 design -- built and run once, at batch size
        1, with the real default input_shape/stage_filters."""
        model = lfem.build_local_feature_extractor()
        x = np.random.rand(1, 512, 512, 8).astype("float32")
        y = model.predict(x, verbose=0)
        self.assertEqual(x.shape, (1, 512, 512, 8))
        self.assertEqual(y.shape, (1, 32, 32, 256))

    def test_output_constants_match_default_configuration(self):
        self.assertEqual(lfem.OUTPUT_SPATIAL_SIZE, 32)
        self.assertEqual(lfem.OUTPUT_CHANNELS, 256)

    def test_input_shape_not_divisible_by_pooling_depth_raises(self):
        with self.assertRaises(ValueError):
            lfem.build_local_feature_extractor(input_shape=(50, 50, 8), stage_filters=(4, 8, 16, 32))

    def test_output_is_finite(self):
        model = lfem.build_local_feature_extractor(input_shape=(32, 32, 8), stage_filters=(4, 8))
        x = np.random.rand(2, 32, 32, 8).astype("float32")
        y = model.predict(x, verbose=0)
        self.assertTrue(np.isfinite(y).all())


class OutputIsSpatialNotPooledTests(unittest.TestCase):
    def setUp(self):
        self.model = lfem.build_local_feature_extractor(input_shape=(32, 32, 8), stage_filters=(4, 8))

    def test_output_rank_is_four(self):
        x = np.random.rand(1, 32, 32, 8).astype("float32")
        y = self.model.predict(x, verbose=0)
        self.assertEqual(y.ndim, 4)

    def test_output_spatial_dimensions_exceed_one(self):
        x = np.random.rand(1, 32, 32, 8).astype("float32")
        y = self.model.predict(x, verbose=0)
        self.assertGreater(y.shape[1], 1)
        self.assertGreater(y.shape[2], 1)

    def test_no_global_pooling_layer_exists(self):
        pooling_types = (tf.keras.layers.GlobalAveragePooling2D, tf.keras.layers.GlobalMaxPooling2D)
        self.assertFalse(any(isinstance(layer, pooling_types) for layer in self.model.layers))


class NoUnintendedClassificationHeadTests(unittest.TestCase):
    def test_no_dense_layer_exists(self):
        model = lfem.build_local_feature_extractor(input_shape=(32, 32, 8), stage_filters=(4, 8))
        self.assertFalse(any(isinstance(layer, tf.keras.layers.Dense) for layer in model.layers))

    def test_model_is_not_compiled_with_a_loss(self):
        """Unlike build_attention_unet, this model must NOT be compiled --
        Stage 05 has no standalone ground truth/loss (see this module's
        docstring)."""
        model = lfem.build_local_feature_extractor(input_shape=(32, 32, 8), stage_filters=(4, 8))
        self.assertIsNone(model.loss)


class MultiKernelBranchesPresentTests(unittest.TestCase):
    """Verifies the multi-kernel design principle is genuinely implemented
    -- not merely claimed -- by inspecting the actual layer objects for
    distinct receptive-field configurations within a single block."""

    def setUp(self):
        self.model = lfem.build_local_feature_extractor(input_shape=(32, 32, 8), stage_filters=(4, 8))

    def test_stage_one_has_three_distinct_receptive_field_branches(self):
        k3 = self.model.get_layer("stage1_k3_conv")
        k5 = self.model.get_layer("stage1_k5_conv")
        dilated = self.model.get_layer("stage1_dilated_conv")

        self.assertEqual(k3.kernel_size, (3, 3))
        self.assertEqual(k3.dilation_rate, (1, 1))

        self.assertEqual(k5.kernel_size, (5, 5))

        self.assertEqual(dilated.kernel_size, (3, 3))
        self.assertEqual(dilated.dilation_rate, (3, 3))

    def test_every_stage_has_a_multi_kernel_block(self):
        # self.model was built with stage_filters=(4, 8) -- 2 stages.
        for i in (1, 2):
            for suffix in ("k3_conv", "k5_conv", "dilated_conv"):
                self.assertIsNotNone(self.model.get_layer(f"stage{i}_{suffix}"))

    def test_four_downsampling_stages_in_default_configuration(self):
        model = lfem.build_local_feature_extractor()
        pool_layers = [layer for layer in model.layers if isinstance(layer, tf.keras.layers.MaxPooling2D)]
        self.assertEqual(len(pool_layers), 4)


class StopGradientBoundaryTests(unittest.TestCase):
    """The frozen Stage 03/04 outputs entering this model must not receive
    gradient from anything downstream of this model -- verified directly
    via GradientTape, not merely asserted in a docstring."""

    def test_gradient_with_respect_to_input_is_none(self):
        model = lfem.build_local_feature_extractor(input_shape=(16, 16, 8), stage_filters=(4,))
        x = tf.Variable(np.random.rand(1, 16, 16, 8).astype("float32"))
        with tf.GradientTape() as tape:
            y = model(x)
            loss = tf.reduce_sum(y)
        grad = tape.gradient(loss, x)
        self.assertIsNone(grad)

    def test_gradient_with_respect_to_model_weights_is_not_none(self):
        """The stop-gradient boundary applies only to the input tensor --
        the model's own parameters must remain fully trainable."""
        model = lfem.build_local_feature_extractor(input_shape=(16, 16, 8), stage_filters=(4,))
        x = np.random.rand(1, 16, 16, 8).astype("float32")
        with tf.GradientTape() as tape:
            y = model(x)
            loss = tf.reduce_sum(y)
        grads = tape.gradient(loss, model.trainable_variables)
        self.assertTrue(len(grads) > 0)
        self.assertTrue(all(g is not None for g in grads))


class ModelParametersAreTrainableTests(unittest.TestCase):
    def test_trainable_variables_are_nonempty(self):
        model = lfem.build_local_feature_extractor(input_shape=(32, 32, 8), stage_filters=(4, 8))
        self.assertGreater(len(model.trainable_variables), 0)
        self.assertGreater(model.count_params(), 0)

    def test_one_gradient_step_changes_weights(self):
        model = lfem.build_local_feature_extractor(input_shape=(16, 16, 8), stage_filters=(4,))
        x = np.random.rand(2, 16, 16, 8).astype("float32")
        before = [w.numpy().copy() for w in model.trainable_variables]

        optimizer = tf.keras.optimizers.SGD(learning_rate=0.1)
        with tf.GradientTape() as tape:
            y = model(x, training=True)
            loss = tf.reduce_mean(tf.square(y))
        grads = tape.gradient(loss, model.trainable_variables)
        optimizer.apply_gradients(zip(grads, model.trainable_variables))

        after = [w.numpy() for w in model.trainable_variables]
        changed = any(not np.allclose(b, a) for b, a in zip(before, after))
        self.assertTrue(changed)


class LocalFeatureExtractionStageTests(unittest.TestCase):
    def setUp(self):
        self.tmp_dir = tempfile.mkdtemp(prefix="local_feature_stage_test_")
        self.addCleanup(shutil.rmtree, self.tmp_dir, ignore_errors=True)
        self.stage = lfem.LocalFeatureExtractionStage(input_shape=(32, 32, 8), stage_filters=(4, 8))

    def test_is_a_feature_extraction_stage(self):
        self.assertIsInstance(self.stage, FeatureExtractionStage)

    def test_build_assigns_uncompiled_model(self):
        model = self.stage.build()
        self.assertIsNotNone(self.stage.model)
        self.assertIs(model, self.stage.model)
        self.assertIsNone(model.loss)

    def test_train_raises_not_implemented(self):
        with self.assertRaises(NotImplementedError):
            self.stage.train(train_data=None)

    def test_evaluate_raises_not_implemented(self):
        with self.assertRaises(NotImplementedError):
            self.stage.evaluate(eval_data=None)

    def test_predict_before_build_or_load_raises(self):
        with self.assertRaises(RuntimeError):
            self.stage.predict(np.zeros((32, 32, 8), dtype="float32"))

    def test_save_before_build_raises(self):
        with self.assertRaises(RuntimeError):
            self.stage.save(os.path.join(self.tmp_dir, "x.keras"))

    def test_predict_single_image_returns_spatial_feature_map(self):
        self.stage.build()
        x = np.random.rand(32, 32, 8).astype("float32")
        features = self.stage.predict(x)
        self.assertEqual(features.shape, (8, 8, 8))

    def test_predict_batch_returns_list_of_feature_maps(self):
        self.stage.build()
        images = [np.random.rand(32, 32, 8).astype("float32") for _ in range(3)]
        results = self.stage.predict_batch(images)
        self.assertEqual(len(results), 3)
        for r in results:
            self.assertEqual(r.shape, (8, 8, 8))

    def test_save_and_load_roundtrip_preserves_predictions(self):
        self.stage.build()
        x = np.random.rand(1, 32, 32, 8).astype("float32")
        predictions_before = self.stage.model.predict(x, verbose=0)

        checkpoint_path = os.path.join(self.tmp_dir, "best_model.keras")
        saved_path = self.stage.save(checkpoint_path)
        self.assertTrue(os.path.exists(saved_path))

        reloaded_stage = lfem.LocalFeatureExtractionStage(input_shape=(32, 32, 8), stage_filters=(4, 8))
        returned = reloaded_stage.load(checkpoint_path)
        self.assertIs(returned, reloaded_stage)

        predictions_after = reloaded_stage.model.predict(x, verbose=0)
        np.testing.assert_allclose(predictions_before, predictions_after, atol=1e-5)


if __name__ == "__main__":
    unittest.main()
