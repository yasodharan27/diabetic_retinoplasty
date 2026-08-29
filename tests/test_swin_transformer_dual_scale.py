"""
Regression tests for swin_transformer.py's Stage 06 addition
(create_dual_scale_swin_model / GlobalFeatureExtractionStage).

The approved Stage 06 architecture has no small/cheap variant -- unlike
Stage 05's build_local_feature_extractor, branch configuration is a fixed
module-level constant, not a function parameter (the architecture is
final, per the Stage 06 design resolution), so these tests build the real,
full-size model. Model *construction* (defining the Keras functional
graph) is cheap regardless of parameter count -- only actual forward/
backward passes cost real compute -- so each TestCase below builds its
model once in setUpClass and reuses it across every test method, to keep
the number of real forward/backward passes in this file to a minimum.

create_hybrid_model() is deliberately never called in this file -- it
loads EfficientNetB0 with weights='imagenet' as PRE-EXISTING behavior,
which would trigger a real network download as a side effect of running
this test suite. Its unchanged status is verified via `git diff` instead
(see the implementation report), not by executing it here.

No training happens anywhere in this file. Every model built here is
untrained (random initialization, exactly as the approved design
specifies) -- no metric is ever reported as a real evaluation result.
"""

import os
import shutil
import tempfile
import unittest

import numpy as np
import tensorflow as tf

import swin_transformer as swin
from pipeline import FeatureExtractionStage


class SharedSwinInfrastructureRegressionTests(unittest.TestCase):
    """Regression tests for the three pre-existing-bug fixes applied to
    SwinTransformerBlock.build() and PatchMerging.call() -- deliberately
    using small, Stage-06-INDEPENDENT dimensions (not DUAL_SCALE_BRANCH_A/
    B_CONFIG), so these tests exercise the shared infrastructure generically
    rather than only re-testing Stage 06's own specific numbers. Every
    class here (PatchEmbed, WindowAttention, SwinTransformerBlock,
    PatchMerging, BasicLayer) is used by both Stage 06 and (transitively,
    if it ever becomes buildable again -- see KnownLegacyIssueTests below)
    create_swin_tiny_model(); create_hybrid_model() is unaffected by any
    of these three fixes (see its own module docstring note) and is never
    called here (no pretrained-weight download)."""

    def test_shift_size_zero_block_builds_and_runs(self):
        """Control case: a block with shift_size=0 (create_hybrid_model()'s
        exact configuration) never enters the fixed code path at all, and
        must keep working exactly as before."""
        inputs = tf.keras.Input(shape=(7, 7, 16))
        block = swin.SwinTransformerBlock(dim=16, num_heads=2, window_size=7, shift_size=0)
        outputs = block(inputs)
        model = tf.keras.Model(inputs=inputs, outputs=outputs)
        x = np.random.RandomState(0).rand(2, 7, 7, 16).astype("float32")
        y = model.predict(x, verbose=0)
        self.assertEqual(y.shape, (2, 7, 7, 16))
        self.assertEqual(y.dtype, np.float32)
        self.assertTrue(np.isfinite(y).all())

    def test_shifted_window_block_builds_and_runs(self):
        """The actual fixed code path: shift_size>0, small dimensions
        unrelated to Stage 06's own branch configs."""
        inputs = tf.keras.Input(shape=(12, 12, 24))
        block = swin.SwinTransformerBlock(dim=24, num_heads=4, window_size=6, shift_size=3)
        outputs = block(inputs)
        model = tf.keras.Model(inputs=inputs, outputs=outputs)
        x = np.random.RandomState(1).rand(2, 12, 12, 24).astype("float32")
        y = model.predict(x, verbose=0)
        self.assertEqual(y.shape, (2, 12, 12, 24))
        self.assertEqual(y.dtype, np.float32)
        self.assertTrue(np.isfinite(y).all())
        self.assertEqual(block.attn_mask.dtype, tf.float32)

    def test_basic_layer_with_downsample_builds_and_runs(self):
        """depth=2 (so both shift_size=0 and shift_size>0 blocks are
        exercised, matching BasicLayer's own alternating rule) plus a
        PatchMerging downsample -- exercises all three fixes together at
        once, at dimensions independent of Stage 06's own configs."""
        inputs = tf.keras.Input(shape=(12, 12, 24))
        layer = swin.BasicLayer(dim=24, depth=2, num_heads=4, window_size=6, downsample=swin.PatchMerging)
        outputs = layer(inputs)
        model = tf.keras.Model(inputs=inputs, outputs=outputs)
        x = np.random.RandomState(2).rand(2, 12, 12, 24).astype("float32")
        y = model.predict(x, verbose=0)
        self.assertEqual(y.shape, (2, 6, 6, 48))  # PatchMerging halves H/W, doubles C
        self.assertTrue(np.isfinite(y).all())

    def test_patch_merging_gradient_flows(self):
        inputs = tf.keras.Input(shape=(8, 8, 16))
        outputs = swin.PatchMerging(dim=16)(inputs)
        model = tf.keras.Model(inputs=inputs, outputs=outputs)
        x = np.random.RandomState(3).rand(1, 8, 8, 16).astype("float32")
        with tf.GradientTape() as tape:
            y = model(x, training=True)
            loss = tf.reduce_mean(tf.square(y))
        grads = tape.gradient(loss, model.trainable_variables)
        self.assertTrue(len(grads) > 0)
        self.assertTrue(all(g is not None for g in grads))

    def test_shifted_window_block_gradient_flows(self):
        inputs = tf.keras.Input(shape=(12, 12, 24))
        outputs = swin.SwinTransformerBlock(dim=24, num_heads=4, window_size=6, shift_size=3)(inputs)
        model = tf.keras.Model(inputs=inputs, outputs=outputs)
        x = np.random.RandomState(4).rand(1, 12, 12, 24).astype("float32")
        with tf.GradientTape() as tape:
            y = model(x, training=True)
            loss = tf.reduce_mean(tf.square(y))
        grads = tape.gradient(loss, model.trainable_variables)
        self.assertTrue(len(grads) > 0)
        self.assertTrue(all(g is not None for g in grads))


class RelativePositionIndexDeviceRegressionTests(unittest.TestCase):
    """Regression tests for a device-placement bug found on a real T4 GPU smoke test:
    `WindowAttention.relative_position_index` used to be a bare `tf.Variable(trainable=False)`
    -- an int32 RESOURCE variable. TensorFlow's own placement policy pins int32 resource
    variables to CPU regardless of GPU availability, and reading a resource variable requires
    the reading op to run on the SAME device it lives on -- so once a model containing this
    layer executed end-to-end on GPU (e.g. the joint model's `model.predict()`), the GPU-placed
    `tf.gather` inside `WindowAttention.call()` could not read the CPU-pinned resource, raising
    `InvalidArgumentError: Trying to access resource relative_position_index ... located on
    device CPU:0 from device GPU:0`. Fixed by making it a plain `tf.constant` instead -- not a
    resource, so it carries no persistent device pinning and TensorFlow copies it to whatever
    device the consuming op actually runs on, CPU or GPU alike."""

    def test_relative_position_index_is_not_a_resource_variable(self):
        """The exact mechanism of the fix -- a `tf.constant`, never a `tf.Variable` -- verified
        directly. This needs no GPU: it is about the OBJECT TYPE, which is what determines
        whether the cross-device resource-access error is even possible in the first place."""
        wa = swin.WindowAttention(dim=32, window_size=6, num_heads=2, name="rpi_type_check")
        self.assertNotIsInstance(wa.relative_position_index, tf.Variable)
        self.assertIsInstance(wa.relative_position_index, tf.Tensor)

    def test_relative_position_index_not_tracked_as_a_layer_weight(self):
        """Preserves the exact pre-fix behavior: this buffer was never part of
        `layer.weights`/`model.count_params()` (a bare `tf.Variable` attribute was not tracked
        by Keras's own weight system either) -- only `relative_position_bias_table` (the real,
        trainable parameter) is."""
        wa = swin.WindowAttention(dim=32, window_size=6, num_heads=2, name="rpi_tracking_check")
        weight_names = [w.name for w in wa.weights]
        self.assertNotIn("relative_position_index", " ".join(weight_names))
        self.assertEqual(len(wa.weights), 1)  # only relative_position_bias_table

    def test_relative_position_index_values_are_unchanged(self):
        """The fix changes ONLY how the buffer is stored (tf.constant vs. tf.Variable), never
        the relative-position math itself -- cross-checked against an independent NumPy
        re-derivation of the exact same formula."""
        window_h = window_w = 8
        wa = swin.WindowAttention(dim=96, window_size=8, num_heads=3, name="rpi_values_check")

        coords_h = np.arange(window_h)
        coords_w = np.arange(window_w)
        coords = np.stack(np.meshgrid(coords_h, coords_w, indexing="ij")).reshape(2, -1).T
        relative_coords = (coords[:, None, :] - coords[None, :, :]).astype(np.int32)
        relative_coords[..., 0] += window_h - 1
        relative_coords[..., 1] += window_w - 1
        expected = (relative_coords[..., 0] * (2 * window_w - 1) + relative_coords[..., 1]).astype(np.int32)

        np.testing.assert_array_equal(wa.relative_position_index.numpy(), expected)

    def test_stage6_parameter_count_and_output_shape_unaffected_by_the_fix(self):
        """Regression guard for this fix's own explicit constraint: it must not change Stage
        06's parameter count or output shape."""
        model = swin.create_dual_scale_swin_model()
        x = np.random.RandomState(5).rand(1, 256, 256, 3).astype("float32")
        y = model.predict(x, verbose=0)
        self.assertEqual(y.shape, (1, 64, 1152))
        self.assertEqual(model.count_params(), 39_697_956)

    @unittest.skipUnless(
        tf.config.list_physical_devices("GPU"), "requires a GPU to reproduce the original failure",
    )
    def test_window_attention_runs_under_explicit_gpu_placement(self):
        """Reproduces the exact original failure condition on a real GPU: forcing execution
        under `/GPU:0` used to raise `InvalidArgumentError: Trying to access resource
        relative_position_index ... located on device CPU:0 from device GPU:0`. Skipped when no
        GPU is present (this project's local/CI environment is CPU-only) -- the CPU-only tests
        above already cover the buffer's type, tracking, and numerical correctness portably."""
        with tf.device("/GPU:0"):
            wa = swin.WindowAttention(dim=32, window_size=6, num_heads=2, name="rpi_gpu_check")
            x = tf.random.normal((2, 36, 32))
            y = wa(x)
        self.assertEqual(y.shape, (2, 36, 32))
        self.assertTrue(np.all(np.isfinite(y.numpy())))

    @unittest.skipUnless(
        tf.config.list_physical_devices("GPU"), "requires a GPU to reproduce the original failure",
    )
    def test_joint_model_forward_and_gradient_pass_on_gpu(self):
        """The actual reported failure: the FULL joint model (Stage 05-08 + RACAF, with Stage 06
        nested inside it) executing end-to-end on GPU via `model.predict()`, then a
        `GradientTape` step -- mirrors `colab/notebooks/stage08_corn_classifier.ipynb`'s own
        smoke-test cell exactly. No optimizer step; this is still a smoke test, not training."""
        import joint_training_model as jtm

        with tf.device("/GPU:0"):
            model = jtm.build_joint_model()
            jtm.compile_joint_model(model)

            s5 = np.random.RandomState(6).rand(2, 512, 512, 8).astype("float32")
            s6 = np.random.RandomState(7).rand(2, 256, 256, 3).astype("float32")
            r = np.random.RandomState(8).rand(2, 1).astype("float32")
            grades = tf.constant([1, 3], dtype=tf.int32)

            logits = model.predict([s5, s6, r], verbose=0)
            self.assertEqual(logits.shape, (2, 4))

            with tf.GradientTape() as tape:
                out = model([s5, s6, r], training=True)
                loss = jtm.joint_corn_loss(grades, out)
            grads = tape.gradient(loss, model.trainable_variables)
            self.assertEqual(sum(1 for g in grads if g is None), 0)


class KnownLegacyIssueTests(unittest.TestCase):
    """SwinTransformer.__init__'s `self.layers = []` shadows Keras 3's
    reserved `Model.layers` property -- a pre-existing defect, independent
    of the three fixes above, deliberately NOT fixed (see the
    implementation report for why: create_swin_tiny_model()/SwinTransformer
    have no active caller anywhere in this project -- verified by grep;
    every real consumer of swin_transformer.py imports create_hybrid_model
    or the low-level building blocks directly, never SwinTransformer).
    This test pins the exact expected failure so a future change that
    accidentally fixes (or further breaks) this code path is caught rather
    than silently drifting -- if this test starts failing because
    SwinTransformer now builds successfully, that's good news; update this
    test, don't just delete it."""

    def test_swin_transformer_class_still_fails_with_the_known_error(self):
        with self.assertRaises(AttributeError) as ctx:
            swin.SwinTransformer(
                img_size=32, patch_size=4, in_chans=3, num_classes=5,
                embed_dim=16, depths=[1, 1], num_heads=[2, 2], window_size=4,
            )
        self.assertIn("Model.layers", str(ctx.exception))

    def test_create_swin_tiny_model_has_no_active_caller_in_this_project(self):
        """Structural evidence, not just a claim: grep every .py file for
        an actual call site (not merely an import)."""
        import subprocess
        repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        result = subprocess.run(
            ["git", "grep", "-l", "-e", "create_swin_tiny_model(", "-e", "SwinTransformer(",
             "--", "*.py"],
            cwd=repo_root, capture_output=True, text=True,
        )
        matching_files = {
            line for line in result.stdout.splitlines()
            if line and line != "swin_transformer.py" and not line.startswith("tests/")
        }
        self.assertEqual(
            matching_files, set(),
            f"create_swin_tiny_model()/SwinTransformer() now has an active caller in "
            f"{matching_files} -- KnownLegacyIssueTests' rationale for leaving self.layers "
            "unfixed no longer holds; re-evaluate.",
        )


class DualScaleSwinArchitectureTests(unittest.TestCase):
    """Structural checks against the real, full-size model -- no forward
    pass needed, so these stay cheap even at full scale."""

    @classmethod
    def setUpClass(cls):
        cls.model = swin.create_dual_scale_swin_model()

    def test_builder_exists_and_returns_a_keras_model(self):
        self.assertTrue(callable(swin.create_dual_scale_swin_model))
        self.assertIsInstance(self.model, tf.keras.Model)

    def test_input_shape_accepted(self):
        self.assertEqual(tuple(self.model.input_shape[1:]), (256, 256, 3))

    def test_branch_a_configuration(self):
        cfg = swin.DUAL_SCALE_BRANCH_A_CONFIG
        self.assertEqual(cfg["patch_size"], 4)
        self.assertEqual(cfg["embed_dim"], 96)
        self.assertEqual(tuple(cfg["depths"]), (2, 2, 6, 2))
        self.assertEqual(tuple(cfg["num_heads"]), (3, 6, 12, 24))
        self.assertEqual(cfg["window_size"], 8)

    def test_branch_b_configuration(self):
        cfg = swin.DUAL_SCALE_BRANCH_B_CONFIG
        self.assertEqual(cfg["patch_size"], 8)
        self.assertEqual(cfg["embed_dim"], 96)
        self.assertEqual(tuple(cfg["depths"]), (2, 2, 6))
        self.assertEqual(tuple(cfg["num_heads"]), (3, 6, 12))
        self.assertEqual(cfg["window_size"], 8)

    def test_branch_a_final_grid_and_channels(self):
        layer = self.model.get_layer("branch_a_final_norm")
        self.assertEqual(tuple(layer.output.shape[1:3]), (8, 8))
        self.assertEqual(layer.output.shape[-1], 768)

    def test_branch_b_final_grid_and_channels(self):
        layer = self.model.get_layer("branch_b_final_norm")
        self.assertEqual(tuple(layer.output.shape[1:3]), (8, 8))
        self.assertEqual(layer.output.shape[-1], 384)

    def test_final_output_shape(self):
        self.assertEqual(tuple(self.model.output_shape[1:]), (64, 1152))

    def test_output_constants_match_model(self):
        self.assertEqual(swin.DUAL_SCALE_OUTPUT_GRID, 8)
        self.assertEqual(swin.DUAL_SCALE_OUTPUT_TOKENS, 64)
        self.assertEqual(swin.DUAL_SCALE_OUTPUT_CHANNELS, 1152)

    def test_no_pooling_layer_exists(self):
        pooling_types = (
            tf.keras.layers.GlobalAveragePooling2D, tf.keras.layers.GlobalMaxPooling2D,
            tf.keras.layers.AveragePooling2D, tf.keras.layers.MaxPooling2D,
        )
        self.assertFalse(any(isinstance(layer, pooling_types) for layer in self.model.layers))

    def test_no_classification_head(self):
        for layer in self.model.layers:
            activation = getattr(layer, "activation", None)
            if activation is not None:
                name = getattr(activation, "__name__", str(activation))
                self.assertNotIn("softmax", name)

    def test_no_projection_between_concat_and_output(self):
        """The Reshape layer must consume the concatenation's output
        directly -- no Dense/Conv layer inserted between them, per the
        approved design's explicit "no post-concatenation projection"
        requirement (RACAF's own W_r already handles this)."""
        concat_layer = self.model.get_layer("dual_scale_concat")
        flatten_layer = self.model.get_layer("dual_scale_flatten")
        self.assertIs(flatten_layer.input, concat_layer.output)

    def test_model_is_not_compiled_with_a_loss(self):
        self.assertIsNone(self.model.loss)

    def test_rejects_non_rgb_channel_count(self):
        """Stage 06 must not consume Stage 5/vessel/lesion channels --
        enforced structurally by rejecting any input_shape whose channel
        count isn't 3."""
        with self.assertRaises(ValueError):
            swin.create_dual_scale_swin_model(input_shape=(256, 256, 8))

    def test_rejects_misaligned_resolution(self):
        """A resolution that breaks the branches' grid-alignment property
        must be rejected, not silently produce misaligned output."""
        with self.assertRaises(ValueError):
            swin.create_dual_scale_swin_model(input_shape=(100, 100, 3))


class DualScaleSwinRandomInitializationTests(unittest.TestCase):
    def test_two_instances_have_different_weights(self):
        """Compares every weight tensor, not a single fixed index -- some
        layers (e.g. LayerNormalization's gamma/beta) are deterministically
        initialized to ones/zeros regardless of random seed, so a single
        arbitrarily-chosen weight is not a reliable randomness probe."""
        model_1 = swin.create_dual_scale_swin_model()
        model_2 = swin.create_dual_scale_swin_model()
        w1_list = model_1.get_weights()
        w2_list = model_2.get_weights()
        self.assertTrue(any(not np.allclose(a, b) for a, b in zip(w1_list, w2_list)))


class DualScaleSwinForwardPassTests(unittest.TestCase):
    """The only tests in this file that run a real forward (and, in one
    case, backward) pass -- built once and reused across methods to
    minimize total compute."""

    @classmethod
    def setUpClass(cls):
        cls.model = swin.create_dual_scale_swin_model()
        cls.x = np.random.RandomState(0).rand(1, 256, 256, 3).astype("float32")
        cls.y = cls.model.predict(cls.x, verbose=0)

    def test_forward_pass_on_synthetic_batch(self):
        self.assertEqual(self.x.shape, (1, 256, 256, 3))
        self.assertEqual(self.y.shape, (1, 64, 1152))
        self.assertTrue(np.isfinite(self.y).all())

    def test_deterministic_inference(self):
        y_again = self.model.predict(self.x, verbose=0)
        np.testing.assert_array_equal(self.y, y_again)

    def test_trainable_and_gradient_flows(self):
        self.assertGreater(len(self.model.trainable_variables), 0)
        with tf.GradientTape() as tape:
            y = self.model(self.x, training=True)
            loss = tf.reduce_mean(tf.square(y))
        grads = tape.gradient(loss, self.model.trainable_variables)
        self.assertTrue(len(grads) > 0)
        self.assertTrue(all(g is not None for g in grads))

    def test_save_and_load_roundtrip_preserves_predictions(self):
        """Weights-only save/load (see GlobalFeatureExtractionStage.save()'s
        docstring for why) -- rebuild via create_dual_scale_swin_model(),
        then load_weights()."""
        tmp_dir = tempfile.mkdtemp(prefix="dual_scale_swin_test_")
        try:
            checkpoint_path = os.path.join(tmp_dir, "best_model.weights.h5")
            self.model.save_weights(checkpoint_path)
            reloaded = swin.create_dual_scale_swin_model()
            reloaded.load_weights(checkpoint_path)
            y_reloaded = reloaded.predict(self.x, verbose=0)
            np.testing.assert_allclose(self.y, y_reloaded, atol=1e-5)
        finally:
            shutil.rmtree(tmp_dir, ignore_errors=True)

    def test_parameter_count(self):
        total = self.model.count_params()
        trainable = int(sum(np.prod(v.shape) for v in self.model.trainable_variables))
        self.assertGreater(total, 0)
        self.assertEqual(total, trainable)  # nothing frozen inside Stage 06 itself


class GlobalFeatureExtractionStageTests(unittest.TestCase):
    def setUp(self):
        self.tmp_dir = tempfile.mkdtemp(prefix="global_feature_stage_test_")
        self.addCleanup(shutil.rmtree, self.tmp_dir, ignore_errors=True)
        self.stage = swin.GlobalFeatureExtractionStage()

    def test_is_a_feature_extraction_stage(self):
        self.assertIsInstance(self.stage, FeatureExtractionStage)

    def test_train_raises_not_implemented(self):
        with self.assertRaises(NotImplementedError):
            self.stage.train(train_data=None)

    def test_evaluate_raises_not_implemented(self):
        with self.assertRaises(NotImplementedError):
            self.stage.evaluate(eval_data=None)

    def test_predict_before_build_or_load_raises(self):
        with self.assertRaises(RuntimeError):
            self.stage.predict(np.zeros((256, 256, 3), dtype="float32"))

    def test_save_before_build_raises(self):
        with self.assertRaises(RuntimeError):
            self.stage.save(os.path.join(self.tmp_dir, "x.weights.h5"))

    def test_build_predict_save_load_roundtrip(self):
        """Exercises the actual public Stage interface (not the raw
        create_dual_scale_swin_model() function) end to end: build, predict,
        save, load into a fresh Stage instance, predict again, compare."""
        self.stage.build()
        x = np.random.RandomState(0).rand(256, 256, 3).astype("float32")
        features_before = self.stage.predict(x)
        self.assertEqual(features_before.shape, (64, 1152))

        checkpoint_path = os.path.join(self.tmp_dir, "best_model.weights.h5")
        saved_path = self.stage.save(checkpoint_path)
        self.assertTrue(os.path.exists(saved_path))

        reloaded_stage = swin.GlobalFeatureExtractionStage()
        returned = reloaded_stage.load(checkpoint_path)
        self.assertIs(returned, reloaded_stage)

        features_after = reloaded_stage.predict(x)
        np.testing.assert_allclose(features_before, features_after, atol=1e-5)


class ExistingSwinBuildersUntouchedTests(unittest.TestCase):
    """create_swin_tiny_model() independently fails to build under the
    currently-installed Keras version (SwinTransformer.__init__'s
    `self.layers = []` shadows Keras 3's reserved `Model.layers` property
    -- a pre-existing defect, unrelated to Stage 06 and not fixed here,
    since Stage 06 never instantiates SwinTransformer and this task
    explicitly forbids modifying create_swin_tiny_model()). That means it
    cannot be exercised as a live regression test right now -- instead,
    this checks the one thing that actually matters for this task: that
    create_swin_tiny_model() and create_hybrid_model()'s source code is
    still present, still exported, and Stage 06's additions did not
    replace or remove them."""

    def test_create_swin_tiny_model_still_defined(self):
        self.assertTrue(hasattr(swin, "create_swin_tiny_model"))
        self.assertTrue(callable(swin.create_swin_tiny_model))

    def test_create_hybrid_model_still_defined(self):
        self.assertTrue(hasattr(swin, "create_hybrid_model"))
        self.assertTrue(callable(swin.create_hybrid_model))

    def test_create_hybrid_model_swin_refine_block_unaffected_by_shift_fix(self):
        """create_hybrid_model()'s one SwinTransformerBlock passes
        shift_size=0 explicitly, so it never enters the code path this
        session's dtype fix touches -- confirmed here by inspecting the
        call site's argument directly, not by running the (pretrained-
        weight-downloading) function itself."""
        import inspect
        source = inspect.getsource(swin.create_hybrid_model)
        self.assertIn("shift_size=0", source)


if __name__ == "__main__":
    unittest.main()
