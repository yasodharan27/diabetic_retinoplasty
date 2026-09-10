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


class AdaptiveBranchFusionTests(unittest.TestCase):
    """The "Adaptive" half of "Adaptive Multi-Kernel CNN": the branch mixture
    must depend on the image, not be one fixed learned vector reused for every
    sample. These tests inspect the real layer, not a docstring claim."""

    def make_branches(self, batch=2, size=8, channels=4, seed=0):
        rng = np.random.RandomState(seed)
        return [tf.constant(rng.rand(batch, size, size, channels).astype("float32"))
                for _ in range(3)]

    def test_layer_builds_and_reports_its_shape(self):
        layer = lfem.AdaptiveBranchFusion()
        branches = self.make_branches(channels=64)
        outputs = layer(branches)
        self.assertTrue(layer.built)
        self.assertEqual(layer.num_branches, 3)
        self.assertEqual(layer.channels, 64)
        self.assertEqual(layer.reduced_units, 8)  # max(64 // 8, 8)
        self.assertEqual(len(outputs), 3)
        for branch, output in zip(branches, outputs):
            self.assertEqual(tuple(output.shape), tuple(branch.shape))

    def test_reduced_units_never_collapse_below_the_floor(self):
        layer = lfem.AdaptiveBranchFusion(reduction_ratio=8, min_reduced_units=8)
        layer(self.make_branches(channels=4))
        self.assertEqual(layer.reduced_units, 8)  # 4 // 8 == 0 would be degenerate

    def test_branch_weights_are_normalized_across_branches(self):
        layer = lfem.AdaptiveBranchFusion()
        branches = self.make_branches(batch=3, channels=16)
        layer(branches)
        weights = np.asarray(layer.branch_weights(branches))

        self.assertEqual(weights.shape, (3, 3, 16))  # (batch, branches, channels)
        np.testing.assert_allclose(weights.sum(axis=1), np.ones((3, 16)), atol=1e-5)
        self.assertTrue((weights >= 0).all())
        self.assertTrue((weights <= 1).all())

    def test_weights_vary_between_different_input_samples(self):
        layer = lfem.AdaptiveBranchFusion()
        first = self.make_branches(batch=1, channels=16, seed=1)
        second = [tf.constant(np.asarray(b) * 4.0 + 2.0) for b in self.make_branches(
            batch=1, channels=16, seed=99)]
        layer(first)
        w1 = np.asarray(layer.branch_weights(first))
        w2 = np.asarray(layer.branch_weights(second))
        self.assertGreater(float(np.abs(w1 - w2).max()), 1e-4,
                           "branch weights did not respond to a different image")

    def test_not_merely_one_globally_fixed_learned_fusion_vector(self):
        """Regression guard for the defect this layer exists to fix.

        A plain concat + 1x1 convolution mixes the branches with a kernel that
        is the SAME for every image. Two checks together rule that out: the
        weights differ across a batch of deliberately different samples, and the
        gradient of the weights with respect to the branch activations is
        non-zero (i.e. the weights are genuinely a function of the input, not a
        constant that merely happens to differ)."""
        layer = lfem.AdaptiveBranchFusion()
        rng = np.random.RandomState(3)
        base = rng.rand(4, 8, 8, 16).astype("float32")
        # Four clearly distinct samples in one batch.
        base[1] *= 5.0
        base[2] = 1.0 - base[2]
        base[3] += 3.0
        branches = [tf.constant(base), tf.constant(base * 0.5), tf.constant(base * 2.0)]
        layer(branches)

        weights = np.asarray(layer.branch_weights(branches))
        spread = np.abs(weights - weights.mean(axis=0, keepdims=True)).max()
        self.assertGreater(spread, 1e-4,
                           "every sample in the batch received the same branch weights -- the "
                           "fusion is still globally fixed, not image-dependent")

        variables = [tf.Variable(b) for b in branches]
        with tf.GradientTape() as tape:
            w = layer.branch_weights(variables)
            probe = tf.reduce_sum(w[:, 0, :])  # depends ONLY on the weights
        grads = tape.gradient(probe, variables)
        self.assertTrue(all(g is not None for g in grads))
        self.assertGreater(float(max(tf.reduce_max(tf.abs(g)) for g in grads)), 0.0,
                           "the branch weights do not depend on the branch activations at all")

    def test_a_sample_gets_the_same_weights_alone_as_inside_a_batch(self):
        """The weights must come from the sample's own features only -- no batch
        statistic, no cross-sample term."""
        layer = lfem.AdaptiveBranchFusion()
        rng = np.random.RandomState(11)
        batch = rng.rand(4, 8, 8, 16).astype("float32")
        branches = [tf.constant(batch), tf.constant(batch * 0.5), tf.constant(batch * 2.0)]
        layer(branches)
        batched = np.asarray(layer.branch_weights(branches))

        single = [tf.constant(np.asarray(b)[2:3]) for b in branches]
        alone = np.asarray(layer.branch_weights(single))
        np.testing.assert_allclose(batched[2:3], alone, atol=1e-5)

    def test_gradients_reach_the_adaptive_weighting_parameters(self):
        model = lfem.build_local_feature_extractor(input_shape=(16, 16, 8), stage_filters=(4,))
        fusion = model.get_layer("stage1_adaptive_fusion")
        self.assertGreater(len(fusion.trainable_weights), 0)

        x = np.random.RandomState(5).rand(2, 16, 16, 8).astype("float32")
        with tf.GradientTape() as tape:
            loss = tf.reduce_mean(tf.square(model(x, training=True)))
        grads = tape.gradient(loss, fusion.trainable_weights)
        self.assertTrue(all(g is not None for g in grads),
                        "the adaptive weighting parameters received no gradient")
        self.assertGreater(float(max(tf.reduce_max(tf.abs(g)) for g in grads)), 0.0)

    def test_batch_size_two_matches_the_training_configuration(self):
        model = lfem.build_local_feature_extractor(input_shape=(32, 32, 8), stage_filters=(4, 8))
        x = np.random.RandomState(6).rand(2, 32, 32, 8).astype("float32")
        y = model.predict(x, verbose=0)
        self.assertEqual(y.shape, (2, 8, 8, 8))
        self.assertTrue(np.isfinite(y).all())

    def test_batch_size_one_still_works(self):
        model = lfem.build_local_feature_extractor(input_shape=(32, 32, 8), stage_filters=(4, 8))
        y = model.predict(np.random.RandomState(7).rand(1, 32, 32, 8).astype("float32"), verbose=0)
        self.assertEqual(y.shape, (1, 8, 8, 8))

    def test_deterministic_input_produces_reproducible_output(self):
        model = lfem.build_local_feature_extractor(input_shape=(32, 32, 8), stage_filters=(4, 8))
        x = np.random.RandomState(8).rand(2, 32, 32, 8).astype("float32")
        first = model.predict(x, verbose=0)
        second = model.predict(x, verbose=0)
        np.testing.assert_allclose(first, second, atol=0)

    def test_runs_under_mixed_float16(self):
        """The block's convolutions run in float16 while the branch weights are
        computed in float32 (see `AdaptiveBranchFusion`'s docstring); both the
        forward pass and the gradients must stay finite."""
        original = tf.keras.mixed_precision.global_policy()
        try:
            tf.keras.mixed_precision.set_global_policy("mixed_float16")
            model = lfem.build_local_feature_extractor(input_shape=(32, 32, 8),
                                                       stage_filters=(4, 8))
            fusion = model.get_layer("stage1_adaptive_fusion")
            self.assertEqual(fusion.dtype_policy.name, "mixed_float16")
            self.assertEqual(fusion.reduce.dtype_policy.name, "float32")
            self.assertEqual(fusion.expand.dtype_policy.name, "float32")

            x = np.random.RandomState(9).rand(2, 32, 32, 8).astype("float32")
            with tf.GradientTape() as tape:
                loss = tf.reduce_mean(tf.square(tf.cast(model(x, training=True), tf.float32)))
            grads = tape.gradient(loss, model.trainable_variables)
            self.assertTrue(np.isfinite(float(loss)))
            self.assertTrue(all(g is not None for g in grads))
            self.assertTrue(all(bool(tf.reduce_all(tf.math.is_finite(tf.cast(g, tf.float32))))
                                for g in grads))
        finally:
            tf.keras.mixed_precision.set_global_policy(original)

    def test_runs_under_xla_jit_compile(self):
        """Requirement 10: the adaptive path must be traceable by the XLA/JIT
        setting the joint training configuration uses (`jit_compile` resolves to
        True on GPU). Everything in `AdaptiveBranchFusion` is shape-static --
        pooling, two dense projections, a reshape, a softmax and a broadcast
        multiply -- so a compiled train step must build and produce finite
        gradients. Compiled here on whatever device this host has; XLA tracing,
        not device throughput, is what is under test."""
        model = lfem.build_local_feature_extractor(input_shape=(32, 32, 8), stage_filters=(4, 8))
        wrapped = tf.keras.Model(model.input, tf.keras.layers.GlobalAveragePooling2D()(model.output))
        wrapped.compile(optimizer=tf.keras.optimizers.Adam(1e-3), loss="mse", jit_compile=True)
        self.assertTrue(wrapped.jit_compile)

        x = np.random.RandomState(12).rand(2, 32, 32, 8).astype("float32")
        y = np.zeros((2, 8), dtype="float32")
        loss = wrapped.train_on_batch(x, y)
        self.assertTrue(np.isfinite(float(loss)))

    def test_layer_configuration_round_trips(self):
        layer = lfem.AdaptiveBranchFusion(reduction_ratio=4, min_reduced_units=2,
                                          name="fusion_probe")
        config_dict = layer.get_config()
        self.assertEqual(config_dict["reduction_ratio"], 4)
        self.assertEqual(config_dict["min_reduced_units"], 2)

        clone = lfem.AdaptiveBranchFusion.from_config(config_dict)
        self.assertEqual(clone.reduction_ratio, 4)
        self.assertEqual(clone.min_reduced_units, 2)
        self.assertEqual(clone.name, "fusion_probe")

    def test_layer_is_registered_for_keras_serialization(self):
        import keras
        registered = keras.saving.get_registered_object(
            "local_feature_extraction>AdaptiveBranchFusion")
        self.assertIs(registered, lfem.AdaptiveBranchFusion)

    def test_a_single_tensor_input_is_rejected(self):
        layer = lfem.AdaptiveBranchFusion()
        with self.assertRaises(ValueError):
            layer(tf.zeros((2, 8, 8, 4)))

    def test_mismatched_branch_channel_counts_are_rejected(self):
        layer = lfem.AdaptiveBranchFusion()
        with self.assertRaises(ValueError):
            layer([tf.zeros((2, 8, 8, 4)), tf.zeros((2, 8, 8, 6)), tf.zeros((2, 8, 8, 4))])

    def test_every_stage_has_an_adaptive_fusion_layer(self):
        model = lfem.build_local_feature_extractor()
        fusion = [l for l in model.layers if isinstance(l, lfem.AdaptiveBranchFusion)]
        self.assertEqual(len(fusion), len(lfem.DEFAULT_STAGE_FILTERS))
        for index, filters in enumerate(lfem.DEFAULT_STAGE_FILTERS, start=1):
            layer = model.get_layer(f"stage{index}_adaptive_fusion")
            self.assertEqual(layer.channels, filters)
            self.assertEqual(layer.num_branches, 3)

    def test_parameter_growth_is_small_and_intentional(self):
        """Adaptivity must not smuggle in a second backbone: the whole mechanism
        is a per-stage (C -> C/8 -> 3C) projection."""
        model = lfem.build_local_feature_extractor()
        added = sum(
            int(np.prod(w.shape))
            for layer in model.layers if isinstance(layer, lfem.AdaptiveBranchFusion)
            for w in layer.weights
        )
        self.assertEqual(added, 45_536)
        self.assertLess(added / model.count_params(), 0.03)

    def test_output_contract_is_unchanged_by_adaptive_fusion(self):
        """The one shape Stage 06/07/RACAF depend on."""
        model = lfem.build_local_feature_extractor()
        self.assertEqual(model.output_shape, (None, 32, 32, 256))
        self.assertEqual(lfem.OUTPUT_SPATIAL_SIZE, 32)
        self.assertEqual(lfem.OUTPUT_CHANNELS, 256)


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
