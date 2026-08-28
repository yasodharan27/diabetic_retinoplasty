"""
Regression tests for racaf.py's RACAF implementation (tta_views /
compute_reliability / get_or_compute_reliability / build_racaf_fusion /
RACAFStage).

Uses a tiny, synthetic stand-in for the frozen Stage 04 model in all
TTA-mechanism tests below, rather than the real
`lesion_segmentation_model.load_lesion_model()` -- that function requires
a real, gitignored, locally-trained `.keras` checkpoint
(`models/lesion_segmentation/best_model.keras`) that is NOT guaranteed to
exist in every environment (a fresh clone of this repository will not
have it). The stand-in matches Stage 04's exact `(512,512,4)->(512,512,4)`
sigmoid signature, so every test here verifies the TTA/reliability
MACHINERY, which depends only on that input/output contract, not on
Stage 04's actual trained weights. `lesion_segmentation_model.py` itself
is never modified, retrained, or fine-tuned anywhere in this file.

"Real" integration tests are reserved for Stage 06/07, per the approved
design's own instruction -- both are checkpoint-free (randomly
initialized, built directly from code) and already used the same way in
`tests/test_feature_fusion.py`.

No training happens anywhere in this file. All save/load tests use
`tempfile` and clean up after themselves; no real checkpoint is left
behind.
"""

import inspect
import os
import shutil
import tempfile
import unittest

import numpy as np
import tensorflow as tf
from keras import Input, Model, layers

import racaf
from pipeline.inference import InferenceStage
from pipeline.trainable import TrainableStage


def _build_dummy_stage4_model():
    """Tiny stand-in for the real, checkpoint-dependent frozen Stage 04
    model -- matches its exact (512,512,4)->(512,512,4) sigmoid signature.
    See this module's docstring for why the real model isn't used here."""
    inputs = Input(shape=(512, 512, 4))
    x = layers.Conv2D(4, 3, padding="same", activation="sigmoid")(inputs)
    return Model(inputs, x)


def _random_aligned_predictions(batch=1, height=16, width=16, num_classes=4, rng=None):
    rng = rng or np.random.RandomState(0)
    return rng.rand(batch, 4, height, width, num_classes).astype("float32")


class TTATransformTests(unittest.TestCase):
    """Verifies the four TTA transforms exactly, and that each is its own
    exact inverse (no interpolation is introduced by any of them)."""

    def test_exactly_four_transforms_in_the_documented_order(self):
        self.assertEqual(
            racaf.TTA_TRANSFORMS,
            ("identity", "horizontal_flip", "vertical_flip", "rotate_180"),
        )

    def test_identity_is_a_no_op(self):
        x = tf.random.normal((1, 4, 4, 2))
        np.testing.assert_array_equal(racaf._apply_transform(x, "identity").numpy(), x.numpy())

    def test_horizontal_flip_flips_width_axis(self):
        x = tf.reshape(tf.range(1.0 * 1 * 1 * 4 * 1), (1, 1, 4, 1))
        flipped = racaf._apply_transform(x, "horizontal_flip").numpy()
        np.testing.assert_array_equal(flipped[0, 0, :, 0], [3, 2, 1, 0])

    def test_vertical_flip_flips_height_axis(self):
        x = tf.reshape(tf.range(1.0 * 1 * 4 * 1 * 1), (1, 4, 1, 1))
        flipped = racaf._apply_transform(x, "vertical_flip").numpy()
        np.testing.assert_array_equal(flipped[0, :, 0, 0], [3, 2, 1, 0])

    def test_rotate_180_flips_both_axes(self):
        x = tf.reshape(tf.range(1.0 * 1 * 2 * 2 * 1), (1, 2, 2, 1))
        rotated = racaf._apply_transform(x, "rotate_180").numpy()
        expected = x.numpy()[:, ::-1, ::-1, :]
        np.testing.assert_array_equal(rotated, expected)

    def test_unknown_transform_raises(self):
        with self.assertRaises(ValueError):
            racaf._apply_transform(tf.zeros((1, 2, 2, 1)), "diagonal_flip")

    def test_every_transform_is_its_own_exact_inverse(self):
        """Applying the same transform twice must exactly reconstruct the
        original tensor -- this is what makes re-applying it a valid
        'inverse transform' step in tta_views(), and what guarantees zero
        interpolation across the round trip."""
        x = tf.random.normal((2, 6, 6, 3))
        for transform in racaf.TTA_TRANSFORMS:
            once = racaf._apply_transform(x, transform)
            twice = racaf._apply_transform(once, transform)
            np.testing.assert_array_equal(twice.numpy(), x.numpy())


class TTAViewsTests(unittest.TestCase):
    """Verifies tta_views() calls the frozen Stage 04 model directly,
    produces correctly-shaped, aligned predictions, and never touches
    predict_lesion_mask()."""

    @classmethod
    def setUpClass(cls):
        cls.stage4 = _build_dummy_stage4_model()

    def test_output_shape_is_batch_four_views_512_512_4(self):
        prepared = np.random.rand(1, 512, 512, 4).astype("float32")
        views = racaf.tta_views(self.stage4, prepared)
        self.assertEqual(tuple(views.shape), (1, 4, 512, 512, 4))

    def test_each_individual_view_matches_stage4_output_contract(self):
        prepared = np.random.rand(2, 512, 512, 4).astype("float32")
        views = racaf.tta_views(self.stage4, prepared)
        for k in range(4):
            self.assertEqual(tuple(views[:, k].shape), (2, 512, 512, 4))

    def test_calls_frozen_model_exactly_four_times(self):
        """Dunder methods like `__call__` are resolved on the type, not
        the instance, so a plain instance-level monkeypatch of
        `__call__` would silently never fire -- a subclassed model with
        its own internal counter avoids that pitfall entirely."""

        class _CountingModel(tf.keras.Model):
            def __init__(self):
                super().__init__()
                self.conv = layers.Conv2D(4, 3, padding="same", activation="sigmoid")
                self.call_count = 0

            def call(self, inputs, training=False):
                self.call_count += 1
                return self.conv(inputs)

        counting_model = _CountingModel()
        prepared = np.random.rand(1, 512, 512, 4).astype("float32")
        counting_model(prepared, training=False)  # warm up: subclassed models trace `call()`
        counting_model.call_count = 0  # once on first invocation, before this counts anything
        racaf.tta_views(counting_model, prepared)
        self.assertEqual(counting_model.call_count, 4)

    def test_identity_view_equals_a_direct_model_call(self):
        prepared = np.random.rand(1, 512, 512, 4).astype("float32")
        views = racaf.tta_views(self.stage4, prepared)
        direct = self.stage4(prepared, training=False).numpy()
        np.testing.assert_allclose(views[:, 0].numpy(), direct, atol=1e-6)

    def test_source_never_calls_predict_lesion_mask(self):
        """Structural evidence, not just a claim: the TTA loop's own
        CODE (excluding its docstring, which legitimately explains what
        it deliberately avoids) must never call predict_lesion_mask /
        predict_lesion_mask_batch."""
        source = inspect.getsource(racaf.tta_views)
        body_only = source[source.index('"""', source.index('"""') + 3) + 3:]
        self.assertNotIn("predict_lesion_mask", body_only)


class Stage4FreezingTests(unittest.TestCase):
    """Verifies Stage 04 is treated as strictly inference-only: explicitly
    non-trainable, and no gradient reaches it through either path."""

    def test_load_frozen_stage4_model_sets_trainable_false(self):
        source = inspect.getsource(racaf.load_frozen_stage4_model)
        self.assertIn("trainable = False", source.replace("trainable=False", "trainable = False"))

    def test_trainable_false_empties_trainable_variables(self):
        model = _build_dummy_stage4_model()
        self.assertGreater(len(model.trainable_variables), 0)
        model.trainable = False
        self.assertEqual(len(model.trainable_variables), 0)

    def test_no_gradient_reaches_stage4_even_when_trainable_true(self):
        """Defense-in-depth check: even if trainable=False were somehow
        skipped, tta_views()'s own stop_gradient must independently block
        all gradient flow into Stage 04's parameters."""
        model = _build_dummy_stage4_model()  # trainable=True (default)
        prepared = tf.Variable(np.random.rand(1, 512, 512, 4).astype("float32"))
        with tf.GradientTape() as tape:
            views = racaf.tta_views(model, prepared)
            loss = tf.reduce_sum(views)
        grads = tape.gradient(loss, model.trainable_variables)
        self.assertTrue(all(g is None for g in grads))

    def test_no_gradient_reaches_stage4_with_trainable_false(self):
        model = _build_dummy_stage4_model()
        model.trainable = False
        prepared = tf.Variable(np.random.rand(1, 512, 512, 4).astype("float32"))
        with tf.GradientTape() as tape:
            views = racaf.tta_views(model, prepared)
            loss = tf.reduce_sum(views)
        grads = tape.gradient(loss, model.trainable_variables)
        self.assertEqual(len(model.trainable_variables), 0)
        self.assertEqual(len(grads), 0)


class DisagreementTests(unittest.TestCase):
    """Verifies the population-variance disagreement definition exactly,
    including the load-bearing DELTA_MAX=0.25 relationship."""

    def test_population_variance_of_maximal_disagreement_is_exactly_0_25(self):
        aligned = np.zeros((1, 4, 1, 1, 1), dtype="float32")
        aligned[0, :, 0, 0, 0] = [0.0, 0.0, 1.0, 1.0]
        result = racaf.compute_reliability(aligned)
        self.assertAlmostEqual(float(result["D"][0, 0, 0, 0]), 0.25, places=6)

    def test_population_variance_matches_numpy_ddof_0(self):
        rng = np.random.RandomState(1)
        aligned = rng.rand(1, 4, 3, 3, 4).astype("float32")
        result = racaf.compute_reliability(aligned)
        expected_D = aligned[0].var(axis=0, ddof=0)
        np.testing.assert_allclose(result["D"][0], expected_D, atol=1e-6)

    def test_population_variance_never_exceeds_delta_max(self):
        rng = np.random.RandomState(2)
        aligned = rng.rand(3, 4, 5, 5, 4).astype("float32")
        result = racaf.compute_reliability(aligned)
        self.assertLessEqual(result["D"].max(), racaf.DELTA_MAX + 1e-6)

    def test_identical_views_give_zero_disagreement(self):
        single = np.random.rand(1, 1, 4, 4, 4).astype("float32")
        aligned = np.repeat(single, 4, axis=1)
        result = racaf.compute_reliability(aligned)
        np.testing.assert_allclose(result["D"], 0.0, atol=1e-7)


class ForegroundMaskTests(unittest.TestCase):
    def test_empty_foreground_gives_delta_zero(self):
        # All predictions well below threshold everywhere -> U_c empty for every class.
        aligned = np.full((1, 4, 4, 4, 4), 0.1, dtype="float32")
        result = racaf.compute_reliability(aligned)
        np.testing.assert_allclose(result["delta"], 0.0, atol=1e-7)

    def test_foreground_threshold_matches_project_default(self):
        self.assertEqual(racaf.FOREGROUND_THRESHOLD, 0.5)

    def test_delta_only_pools_over_foreground_pixels(self):
        """A single, high-disagreement background pixel (mean <= 0.5) must
        not influence delta; only foreground pixels should."""
        aligned = np.full((1, 4, 2, 2, 1), 0.9, dtype="float32")  # foreground, no disagreement
        aligned[0, :, 0, 0, 0] = [0.9, 0.9, 0.9, 0.9]
        aligned[0, :, 1, 1, 0] = [0.0, 0.0, 0.4, 0.4]  # mean=0.2, background, high disagreement
        result = racaf.compute_reliability(aligned)
        self.assertAlmostEqual(float(result["delta"][0, 0]), 0.0, places=6)


class KappaTests(unittest.TestCase):
    def test_kappa_bounded_in_0_1_for_random_inputs(self):
        rng = np.random.RandomState(3)
        for _ in range(5):
            aligned = _random_aligned_predictions(batch=2, rng=rng)
            result = racaf.compute_reliability(aligned)
            self.assertTrue(np.all(result["kappa"] >= 0.0))
            self.assertTrue(np.all(result["kappa"] <= 1.0))

    def test_kappa_is_one_when_delta_is_zero(self):
        aligned = np.full((1, 4, 4, 4, 4), 0.9, dtype="float32")  # no disagreement anywhere
        result = racaf.compute_reliability(aligned)
        np.testing.assert_allclose(result["kappa"], 1.0, atol=1e-6)

    def test_kappa_shape_is_four_per_class(self):
        aligned = _random_aligned_predictions(batch=3)
        result = racaf.compute_reliability(aligned)
        self.assertEqual(result["kappa"].shape, (3, 4))


class BurdenWeightTests(unittest.TestCase):
    def test_burden_weights_sum_to_one(self):
        rng = np.random.RandomState(4)
        aligned = _random_aligned_predictions(batch=4, rng=rng)
        result = racaf.compute_reliability(aligned)
        np.testing.assert_allclose(result["burden_weight"].sum(axis=1), 1.0, atol=1e-6)

    def test_zero_burden_fallback_is_exactly_one_quarter_each(self):
        aligned = np.zeros((2, 4, 4, 4, 4), dtype="float32")
        result = racaf.compute_reliability(aligned)
        np.testing.assert_allclose(result["burden_weight"], 0.25, atol=1e-7)

    def test_burden_dominant_class_gets_largest_weight(self):
        aligned = np.zeros((1, 4, 2, 2, 4), dtype="float32")
        aligned[0, :, :, :, 0] = 0.9  # class 0 has by far the highest burden
        aligned[0, :, :, :, 1:] = 0.05
        result = racaf.compute_reliability(aligned)
        self.assertEqual(np.argmax(result["burden_weight"][0]), 0)


class ImageLevelReliabilityTests(unittest.TestCase):
    def test_r_is_scalar_per_image(self):
        aligned = _random_aligned_predictions(batch=5)
        result = racaf.compute_reliability(aligned)
        self.assertEqual(result["r"].shape, (5,))

    def test_r_bounded_in_0_1(self):
        rng = np.random.RandomState(5)
        for _ in range(5):
            aligned = _random_aligned_predictions(batch=1, rng=rng)
            result = racaf.compute_reliability(aligned)
            self.assertTrue(0.0 <= float(result["r"][0]) <= 1.0)

    def test_r_equals_weighted_sum_of_kappa(self):
        aligned = _random_aligned_predictions(batch=1)
        result = racaf.compute_reliability(aligned)
        expected_r = float((result["burden_weight"][0] * result["kappa"][0]).sum())
        self.assertAlmostEqual(float(result["r"][0]), expected_r, places=6)

    def test_zero_burden_image_has_r_equal_one(self):
        aligned = np.zeros((1, 4, 4, 4, 4), dtype="float32")
        result = racaf.compute_reliability(aligned)
        self.assertAlmostEqual(float(result["r"][0]), 1.0, places=6)

    def test_no_learned_projection_of_kappa_to_256_dims(self):
        """The approved design collapses kappa=(4,) to a scalar r via a
        fixed (non-trainable) weighted sum -- there must be no learned
        Dense(4->256)-style projection of kappa anywhere in this module."""
        source = inspect.getsource(racaf)
        # No Dense layer is ever applied to a (4,)-shaped kappa tensor.
        self.assertNotIn("Dense(256)(kappa", source.replace(" ", ""))
        self.assertNotIn("Dense(d_model)(kappa", source.replace(" ", ""))


class GateTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.model = racaf.build_racaf_fusion()

    def test_gate_layer_configuration(self):
        gate_layer = self.model.get_layer("reliability_gate")
        self.assertIsInstance(gate_layer, tf.keras.layers.Dense)
        self.assertEqual(gate_layer.units, 1)
        self.assertEqual(gate_layer.activation.__name__, "sigmoid")

    def test_gate_output_in_open_interval_0_1(self):
        r = np.array([[-5.0], [0.0], [0.5], [1.0], [5.0]], dtype="float32")
        e = np.zeros((5, 256), dtype="float32")
        g = np.zeros((5, 64, 1152), dtype="float32")
        out = self.model.predict([e, g, r], verbose=0)
        gate_model = Model(self.model.input, self.model.get_layer("reliability_gate").output)
        gate_values = gate_model.predict([e, g, r], verbose=0)
        self.assertTrue(np.all(gate_values > 0.0))
        self.assertTrue(np.all(gate_values < 1.0))

    def test_gate_has_exactly_two_trainable_parameters(self):
        gate_layer = self.model.get_layer("reliability_gate")
        total = sum(int(np.prod(v.shape)) for v in gate_layer.trainable_variables)
        self.assertEqual(total, 2)


class GlobalFeaturePathTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.model = racaf.build_racaf_fusion()

    def test_gap_output_shape(self):
        self.assertEqual(self.model.get_layer("gap_g").output.shape, (None, 1152))

    def test_global_projection_output_shape(self):
        self.assertEqual(self.model.get_layer("global_projection").output.shape, (None, 256))

    def test_global_projection_reads_from_gap_not_from_e_or_stage7(self):
        proj_node = self.model.get_layer("global_projection")._inbound_nodes[0]
        source_names = {t._keras_history.operation.name for t in proj_node.input_tensors}
        self.assertEqual(source_names, {"gap_g"})

    def test_global_projection_parameter_count(self):
        proj_layer = self.model.get_layer("global_projection")
        total = sum(int(np.prod(v.shape)) for v in proj_layer.trainable_variables)
        self.assertEqual(total, 1152 * 256 + 256)


class FusionTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.model = racaf.build_racaf_fusion()

    def test_output_shape(self):
        self.assertEqual(self.model.output_shape, (None, 256))

    def test_fusion_is_convex_combination_at_extreme_gates(self):
        """When gate->1 (r large, positive w_g,b_g), F should approach E;
        when gate->0, F should approach G_hat. We can't force the learned
        w_g/b_g directly, but we CAN verify the algebra by constructing a
        model with fixed weights."""
        model = racaf.build_racaf_fusion()
        gate_layer = model.get_layer("reliability_gate")
        gate_layer.set_weights([np.array([[100.0]], dtype="float32"), np.array([0.0], dtype="float32")])
        proj_layer = model.get_layer("global_projection")
        w_shape = proj_layer.get_weights()[0].shape
        proj_layer.set_weights([np.zeros(w_shape, dtype="float32"), np.ones(256, dtype="float32")])

        e = np.random.rand(1, 256).astype("float32")
        g = np.zeros((1, 64, 1152), dtype="float32")
        r_high = np.array([[10.0]], dtype="float32")  # gate ~= sigmoid(100*10) ~= 1
        out_high = model.predict([e, g, r_high], verbose=0)
        np.testing.assert_allclose(out_high[0], e[0], atol=1e-3)

        r_low = np.array([[-10.0]], dtype="float32")  # gate ~= sigmoid(-1000) ~= 0
        out_low = model.predict([e, g, r_low], verbose=0)
        np.testing.assert_allclose(out_low[0], np.ones(256, dtype="float32"), atol=1e-3)

    def test_output_not_dependent_on_e_when_gate_forced_to_zero(self):
        model = racaf.build_racaf_fusion()
        gate_layer = model.get_layer("reliability_gate")
        gate_layer.set_weights([np.array([[0.0]], dtype="float32"), np.array([-100.0], dtype="float32")])
        g = np.random.rand(1, 64, 1152).astype("float32")
        r = np.zeros((1, 1), dtype="float32")
        e1 = np.random.rand(1, 256).astype("float32")
        e2 = np.random.rand(1, 256).astype("float32")
        out1 = model.predict([e1, g, r], verbose=0)
        out2 = model.predict([e2, g, r], verbose=0)
        np.testing.assert_allclose(out1, out2, atol=1e-4)


class ParameterCountTests(unittest.TestCase):
    def test_total_trainable_parameter_count_is_exactly_295170(self):
        model = racaf.build_racaf_fusion()
        total = sum(int(np.prod(v.shape)) for v in model.trainable_variables)
        self.assertEqual(total, 295170)

    def test_no_non_trainable_parameters(self):
        model = racaf.build_racaf_fusion()
        self.assertEqual(len(model.non_trainable_variables), 0)

    def test_model_is_uncompiled(self):
        model = racaf.build_racaf_fusion()
        self.assertIsNone(model.loss)
        self.assertFalse(model.compiled)


class InnovationBoundaryTests(unittest.TestCase):
    """Structural evidence that RACAF does not accidentally become a
    second attention mechanism, a feature extractor, or a classifier, and
    that its code never touches ground truth or evaluation metrics."""

    @classmethod
    def setUpClass(cls):
        cls.path = os.path.join(os.path.dirname(__file__), "..", "racaf.py")
        with open(cls.path) as f:
            cls.source = f.read()

        import tokenize as tokenize_mod
        names = []
        with open(cls.path, "rb") as f:
            for tok in tokenize_mod.tokenize(f.readline):
                if tok.type == tokenize_mod.NAME:
                    names.append(tok.string.lower())
        cls.code_identifiers = set(names)

    def test_no_attention_layer(self):
        model = racaf.build_racaf_fusion()
        for layer in model.layers:
            self.assertNotIsInstance(layer, tf.keras.layers.MultiHeadAttention)

    def test_no_convolutional_or_recurrent_feature_extractor(self):
        model = racaf.build_racaf_fusion()
        forbidden = (
            tf.keras.layers.Conv1D, tf.keras.layers.Conv2D, tf.keras.layers.Conv3D,
            tf.keras.layers.LSTM, tf.keras.layers.GRU,
        )
        for layer in model.layers:
            self.assertNotIsInstance(layer, forbidden)

    def test_no_classification_head(self):
        model = racaf.build_racaf_fusion()
        self.assertEqual(model.output_shape, (None, 256))
        for layer in model.layers:
            self.assertNotIsInstance(layer, tf.keras.layers.Softmax)

    def test_exactly_one_gate_and_one_global_projection_layer(self):
        model = racaf.build_racaf_fusion()
        dense_layers = [l for l in model.layers if isinstance(l, tf.keras.layers.Dense)]
        self.assertEqual(len(dense_layers), 2)

    def test_no_ground_truth_or_metric_identifiers_in_code(self):
        forbidden = {
            "ground_truth", "y_true", "label", "labels", "diagnosis",
            "dice", "dice_coefficient", "iou", "iou_score", "accuracy",
        }
        found = forbidden & self.code_identifiers
        self.assertFalse(found, f"found forbidden identifiers in code: {found}")

    def test_no_training_or_compilation_of_stage4(self):
        forbidden_calls = {"fit"}
        found = forbidden_calls & self.code_identifiers
        self.assertFalse(found, f"found forbidden Stage-4-training identifiers: {found}")
        self.assertNotIn(".compile(", self.source)


class SerializationTests(unittest.TestCase):
    def setUp(self):
        self.tmpdir = tempfile.mkdtemp()

    def tearDown(self):
        shutil.rmtree(self.tmpdir, ignore_errors=True)

    def test_full_keras_save_load_roundtrip(self):
        model = racaf.build_racaf_fusion()
        e = np.random.rand(2, 256).astype("float32")
        g = np.random.rand(2, 64, 1152).astype("float32")
        r = np.random.rand(2, 1).astype("float32")
        out_before = model.predict([e, g, r], verbose=0)

        path = os.path.join(self.tmpdir, "racaf.keras")
        model.save(path)
        loaded = tf.keras.models.load_model(
            path, compile=False, custom_objects={"_OneMinus": racaf._OneMinus},
        )
        out_after = loaded.predict([e, g, r], verbose=0)

        np.testing.assert_allclose(out_before, out_after, atol=1e-5)
        for w_before, w_after in zip(model.get_weights(), loaded.get_weights()):
            np.testing.assert_array_equal(w_before, w_after)

    def test_racaf_stage_build_save_load_predict_roundtrip(self):
        stage = racaf.RACAFStage()
        stage.build()
        e = np.random.rand(256).astype("float32")
        g = np.random.rand(64, 1152).astype("float32")
        r = 0.42
        f_before = stage.predict((e, g, r))

        path = os.path.join(self.tmpdir, "racaf_stage.keras")
        stage.save(path)

        loaded_stage = racaf.RACAFStage()
        loaded_stage.load(path)
        f_after = loaded_stage.predict((e, g, r))

        np.testing.assert_allclose(f_before, f_after, atol=1e-5)
        self.assertEqual(f_before.shape, (256,))


class GradientTests(unittest.TestCase):
    def test_gradients_exist_and_finite_for_racaf_trainable_variables(self):
        model = racaf.build_racaf_fusion()
        e = tf.random.normal((2, 256))
        g = tf.random.normal((2, 64, 1152))
        r = tf.random.uniform((2, 1))
        with tf.GradientTape() as tape:
            out = model([e, g, r])
            loss = tf.reduce_sum(out)
        grads = tape.gradient(loss, model.trainable_variables)
        self.assertGreater(len(model.trainable_variables), 0)
        for var, grad in zip(model.trainable_variables, grads):
            self.assertIsNotNone(grad, f"missing gradient for {var.name}")
            self.assertTrue(np.all(np.isfinite(grad.numpy())), f"non-finite gradient for {var.name}")


class BatchIndependenceTests(unittest.TestCase):
    def test_fusion_samples_do_not_affect_one_another(self):
        model = racaf.build_racaf_fusion()
        rng = np.random.RandomState(7)
        e = rng.rand(2, 256).astype("float32")
        g = rng.rand(2, 64, 1152).astype("float32")
        r = rng.rand(2, 1).astype("float32")

        e2 = e.copy()
        e2[0] = rng.rand(256).astype("float32")

        out1 = model.predict([e, g, r], verbose=0)
        out2 = model.predict([e2, g, r], verbose=0)

        np.testing.assert_allclose(out1[1], out2[1], atol=1e-5)
        self.assertFalse(np.allclose(out1[0], out2[0], atol=1e-5))

    def test_reliability_samples_do_not_affect_one_another(self):
        rng = np.random.RandomState(8)
        aligned = _random_aligned_predictions(batch=2, rng=rng)
        aligned2 = aligned.copy()
        aligned2[0] = rng.rand(4, 16, 16, 4).astype("float32")

        result1 = racaf.compute_reliability(aligned)
        result2 = racaf.compute_reliability(aligned2)

        np.testing.assert_allclose(result1["r"][1], result2["r"][1], atol=1e-6)


class NumericalStabilityTests(unittest.TestCase):
    def test_all_zero_predictions_produce_no_nan_or_inf(self):
        aligned = np.zeros((1, 4, 8, 8, 4), dtype="float32")
        result = racaf.compute_reliability(aligned)
        for key, value in result.items():
            self.assertTrue(np.all(np.isfinite(value)), f"{key} contains non-finite values")

    def test_all_one_predictions_produce_no_nan_or_inf(self):
        aligned = np.ones((1, 4, 8, 8, 4), dtype="float32")
        result = racaf.compute_reliability(aligned)
        for key, value in result.items():
            self.assertTrue(np.all(np.isfinite(value)), f"{key} contains non-finite values")

    def test_near_zero_burden_does_not_blow_up(self):
        aligned = np.full((1, 4, 8, 8, 4), 1e-10, dtype="float32")
        result = racaf.compute_reliability(aligned)
        self.assertTrue(np.all(np.isfinite(result["burden_weight"])))
        np.testing.assert_allclose(result["burden_weight"], 0.25, atol=1e-3)


class ReliabilityCacheTests(unittest.TestCase):
    def setUp(self):
        self.tmpdir = tempfile.mkdtemp()

    def tearDown(self):
        shutil.rmtree(self.tmpdir, ignore_errors=True)

    def test_cache_miss_then_hit_returns_identical_values(self):
        stage4 = _build_dummy_stage4_model()
        prepared = np.random.rand(1, 512, 512, 4).astype("float32")
        cache_path = racaf.reliability_cache_path(self.tmpdir, "TEST_0001")

        self.assertFalse(os.path.exists(cache_path))
        kappa1, r1 = racaf.get_or_compute_reliability(prepared, cache_path, stage4)
        self.assertTrue(os.path.exists(cache_path))

        kappa2, r2 = racaf.get_or_compute_reliability(prepared, cache_path, stage4)
        np.testing.assert_allclose(kappa1, kappa2, atol=1e-6)
        self.assertAlmostEqual(r1, r2, places=6)

    def test_cache_stores_small_derived_values_not_raw_maps(self):
        stage4 = _build_dummy_stage4_model()
        prepared = np.random.rand(1, 512, 512, 4).astype("float32")
        cache_path = racaf.reliability_cache_path(self.tmpdir, "TEST_0002")
        racaf.get_or_compute_reliability(prepared, cache_path, stage4)

        cached = np.load(cache_path)
        self.assertEqual(set(cached.files), {"kappa", "r"})
        self.assertEqual(cached["kappa"].shape, (4,))
        self.assertEqual(cached["r"].shape, ())
        # A cached kappa+r file must be tiny -- nowhere near the size of
        # four raw (512,512,4) float32 probability maps (~16.8MB).
        self.assertLess(os.path.getsize(cache_path), 10_000)

    def test_cache_never_reads_or_writes_ground_truth(self):
        source = inspect.getsource(racaf.get_or_compute_reliability)
        for term in ("ground_truth", "label", "dice", "iou"):
            self.assertNotIn(term, source.lower())


class RACAFStageTests(unittest.TestCase):
    def test_is_a_trainable_and_inference_stage(self):
        stage = racaf.RACAFStage()
        self.assertIsInstance(stage, TrainableStage)
        self.assertIsInstance(stage, InferenceStage)

    def test_train_raises_not_implemented(self):
        stage = racaf.RACAFStage()
        with self.assertRaises(NotImplementedError):
            stage.train(train_data=None)

    def test_evaluate_raises_not_implemented(self):
        stage = racaf.RACAFStage()
        with self.assertRaises(NotImplementedError):
            stage.evaluate(eval_data=None)

    def test_predict_before_build_or_load_raises(self):
        stage = racaf.RACAFStage()
        with self.assertRaises(RuntimeError):
            stage.predict((np.zeros(256), np.zeros((64, 1152)), 0.5))

    def test_save_before_build_raises(self):
        stage = racaf.RACAFStage()
        with self.assertRaises(RuntimeError):
            stage.save("/tmp/should_not_be_created.keras")

    def test_predict_batch(self):
        stage = racaf.RACAFStage()
        stage.build()
        triples = [
            (np.random.rand(256).astype("float32"), np.random.rand(64, 1152).astype("float32"), 0.5)
            for _ in range(3)
        ]
        results = stage.predict_batch(triples)
        self.assertEqual(len(results), 3)
        for result in results:
            self.assertEqual(result.shape, (256,))


class IntegrationTests(unittest.TestCase):
    """Real Stage 06 -> RACAF and Stage 07 -> RACAF integration, using the
    actual, checkpoint-free (randomly initialized) implemented models."""

    def test_real_stage6_to_racaf(self):
        import swin_transformer as st

        global_model = st.create_dual_scale_swin_model()
        racaf_model = racaf.build_racaf_fusion()

        global_raw = np.random.rand(1, *st.DEFAULT_GLOBAL_FEATURE_INPUT_SHAPE).astype("float32")
        G = global_model.predict(global_raw, verbose=0)
        self.assertEqual(G.shape, (1, 64, 1152))

        e = np.random.rand(1, 256).astype("float32")
        r = np.array([[0.6]], dtype="float32")
        F = racaf_model.predict([e, G, r], verbose=0)
        self.assertEqual(F.shape, (1, 256))
        self.assertTrue(np.all(np.isfinite(F)))

    def test_real_stage7_to_racaf(self):
        import feature_fusion as ff

        fusion_model = ff.build_adaptive_cross_attention()
        racaf_model = racaf.build_racaf_fusion()

        local = np.random.rand(1, *ff.DEFAULT_LOCAL_SHAPE).astype("float32")
        glob = np.random.rand(1, *ff.DEFAULT_GLOBAL_SHAPE).astype("float32")
        E = fusion_model.predict([local, glob], verbose=0)
        self.assertEqual(E.shape, (1, 256))

        r = np.array([[0.4]], dtype="float32")
        F = racaf_model.predict([E, glob, r], verbose=0)
        self.assertEqual(F.shape, (1, 256))
        self.assertTrue(np.all(np.isfinite(F)))

    def test_real_stage5_stage6_stage7_racaf_chain(self):
        """Full real chain: Stage 05 -> Stage 06 -> Stage 07 -> RACAF,
        using only real, already-implemented, checkpoint-free models plus
        a synthetic reliability scalar (TTA/reliability computation is
        tested separately above with the dummy Stage 4 stand-in)."""
        import local_feature_extraction_model as lfe
        import swin_transformer as st
        import feature_fusion as ff

        local_model = lfe.build_local_feature_extractor()
        global_model = st.create_dual_scale_swin_model()
        fusion_model = ff.build_adaptive_cross_attention()
        racaf_model = racaf.build_racaf_fusion()

        local_raw = np.random.rand(1, *lfe.DEFAULT_INPUT_SHAPE).astype("float32")
        global_raw = np.random.rand(1, *st.DEFAULT_GLOBAL_FEATURE_INPUT_SHAPE).astype("float32")

        L = local_model.predict(local_raw, verbose=0)
        G = global_model.predict(global_raw, verbose=0)
        E = fusion_model.predict([L, G], verbose=0)
        r = np.array([[0.5]], dtype="float32")
        F = racaf_model.predict([E, G, r], verbose=0)

        self.assertEqual(F.shape, (1, 256))
        self.assertTrue(np.all(np.isfinite(F)))


if __name__ == "__main__":
    unittest.main()
