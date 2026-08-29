"""
Tests for corn.py -- CORN (Stage 08) ordinal classification: architecture,
loss, inference/decoding, and the pipeline.ClassificationStage interface.

Uses hand-computed reference cases wherever possible (not shape checks
alone), per this project's verification discipline. The one real-model
integration test builds real, untrained Stage 06 -> Stage 07 -> RACAF
models (checkpoint-free, buildable directly from code) to prove CORN's
input contract matches RACAF's actual output -- no training, no checkpoint.
"""

import os
import shutil
import tempfile
import unittest

import numpy as np
import tensorflow as tf

import corn
import downstream_split


class ModelArchitectureTests(unittest.TestCase):
    def setUp(self):
        self.model = corn.build_corn_model()

    def test_input_shape(self):
        self.assertEqual(self.model.input_shape, (None, 256))

    def test_output_shape(self):
        self.assertEqual(self.model.output_shape, (None, 4))

    def test_exactly_1028_trainable_parameters(self):
        total = sum(int(np.prod(v.shape)) for v in self.model.trainable_variables)
        self.assertEqual(total, 1028)
        self.assertEqual(total, 256 * 4 + 4)

    def test_zero_non_trainable_parameters(self):
        total = sum(int(np.prod(v.shape)) for v in self.model.non_trainable_variables)
        self.assertEqual(total, 0)

    def test_no_hidden_layer(self):
        """Exactly one Dense layer -- InputLayer + Dense, nothing between."""
        layer_types = [layer.__class__.__name__ for layer in self.model.layers]
        self.assertEqual(layer_types, ["InputLayer", "Dense"])

    def test_no_output_activation(self):
        """Keras represents "no activation" as the `linear` identity
        function -- confirms activation=None was actually honored, not
        silently defaulted to sigmoid/softmax."""
        dense_layer = self.model.get_layer("corn_logits")
        self.assertEqual(dense_layer.activation.__name__, "linear")

    def test_returns_raw_unbounded_logits(self):
        """A Dense layer with no activation can output values outside
        [0,1] -- proving no sigmoid/softmax is silently applied inside the
        model itself (that happens only in corn_loss/decode_logits)."""
        f = np.random.RandomState(0).uniform(-50, 50, size=(4, 256)).astype("float32")
        logits = self.model.predict(f, verbose=0)
        self.assertTrue(np.any(np.abs(logits) > 5.0), "expected at least one large-magnitude raw logit")

    def test_model_is_uncompiled(self):
        self.assertFalse(self.model.compiled)
        self.assertIsNone(getattr(self.model, "optimizer", None))

    def test_no_attention_or_conv_layers(self):
        """No second research innovation smuggled in -- only Dense/Input."""
        for layer in self.model.layers:
            self.assertNotIn("Attention", layer.__class__.__name__)
            self.assertNotIn("Conv", layer.__class__.__name__)


class CornLossExactReferenceTests(unittest.TestCase):
    """Hand-computed reference cases, not shape checks."""

    def test_confident_correct_prediction_gives_near_zero_loss(self):
        logits = tf.constant([[10.0, 10.0, 10.0, 10.0]])
        loss = corn.corn_loss(logits, tf.constant([4]))
        self.assertLess(float(loss), 1e-3)

    def test_confidently_wrong_prediction_gives_large_loss(self):
        logits = tf.constant([[-10.0, -10.0, -10.0, -10.0]])
        loss = corn.corn_loss(logits, tf.constant([4]))
        self.assertGreater(float(loss), 5.0)

    def test_neutral_logits_grade_2_matches_hand_computation(self):
        """grade=2, all logits=0 (p_cond=0.5 each):
        k=0: mask=1 (2>=0), target=1 (2>0) -> BCE(0,1) = ln(2)
        k=1: mask=1 (2>=1), target=1 (2>1) -> BCE(0,1) = ln(2)
        k=2: mask=1 (2>=2), target=0 (2>2 false) -> BCE(0,0) = ln(2)
        k=3: mask=0 (2>=3 false) -- excluded
        total = 3*ln(2) / 3 = ln(2)."""
        logits = tf.constant([[0.0, 0.0, 0.0, 0.0]])
        loss = corn.corn_loss(logits, tf.constant([2]))
        self.assertAlmostEqual(float(loss), np.log(2), places=5)

    def test_pooled_denominator_across_batch_matches_hand_computation(self):
        """Two samples, both logits=0: grade=0 contributes 1 included pair
        (k=0 only), grade=3 contributes 4 (k=0..3) -- total included = 5,
        every per-element loss is ln(2) (since target is 0 or 1 and
        sigmoid(0)=0.5 either way) -> total = 5*ln(2)/5 = ln(2). This is
        the "pooled across all tasks and examples" property, not a
        per-task or per-example average."""
        logits = tf.constant([[0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0]])
        loss = corn.corn_loss(logits, tf.constant([0, 3]))
        self.assertAlmostEqual(float(loss), np.log(2), places=5)

    def test_conditional_inclusion_matches_y_greater_equal_k(self):
        """A sample with grade=1 must be included in tasks k=0,1 (1>=0,
        1>=1) but NOT k=2,3 (1>=2 false). Verified by comparing against a
        loss computed with logits deliberately set so that an incorrectly
        *included* task 2/3 would change the result, but a correctly
        *excluded* one would not."""
        logits_a = tf.constant([[0.0, 0.0, 999.0, 999.0]])  # huge logits at excluded tasks
        logits_b = tf.constant([[0.0, 0.0, -999.0, -999.0]])  # opposite huge logits
        loss_a = corn.corn_loss(logits_a, tf.constant([1]))
        loss_b = corn.corn_loss(logits_b, tf.constant([1]))
        # If tasks 2/3 were (incorrectly) included, these two losses would
        # differ enormously (one near-zero, one huge). Since grade=1
        # excludes both, they must be identical.
        self.assertAlmostEqual(float(loss_a), float(loss_b), places=5)

    def test_target_is_y_greater_than_k_not_y_greater_equal_k(self):
        """grade=1, task k=1: mask=1 (1>=1), target = 1[1>1] = 0, NOT 1.
        A large positive logit (implying target=1) should therefore
        produce a LARGE loss here, not a small one."""
        logits = tf.constant([[0.0, 20.0, 0.0, 0.0]])  # k=1 logit strongly implies target=1
        loss_with_wrong_target_implied = corn.corn_loss(logits, tf.constant([1]))
        logits_correct = tf.constant([[0.0, -20.0, 0.0, 0.0]])  # k=1 logit strongly implies target=0 (correct)
        loss_correct = corn.corn_loss(logits_correct, tf.constant([1]))
        self.assertGreater(float(loss_with_wrong_target_implied), float(loss_correct) + 5.0)

    def test_matches_manual_stable_bce_formula(self):
        """Cross-checks tf.nn.sigmoid_cross_entropy_with_logits against the
        textbook -[t*log(sigmoid(z)) + (1-t)*log(1-sigmoid(z))] formula
        directly (not just trusting the built-in), for one included pair."""
        z = 1.7
        t = 1.0
        manual_bce = -(t * np.log(1 / (1 + np.exp(-z))) + (1 - t) * np.log(1 - 1 / (1 + np.exp(-z))))
        logits = tf.constant([[z, 0.0, 0.0, 0.0]])
        # grade=0 -> only k=0 included, target=1[0>0]=0... need target=1 so use grade s.t. k=0 target=1
        loss = corn.corn_loss(tf.constant([[z]]), tf.constant([1]), num_thresholds=1)
        self.assertAlmostEqual(float(loss), manual_bce, places=5)

    def test_loss_is_scalar(self):
        logits = tf.random.normal((5, 4))
        grades = tf.constant([0, 1, 2, 3, 4])
        loss = corn.corn_loss(logits, grades)
        self.assertEqual(loss.shape, ())

    def test_corn_loss_accepts_float16_logits(self):
        """Regression test for the mixed-precision crash found on the first real T4 joint
        smoke test: under `mixed_float16`, CORN's Dense layer (no explicit float32 output
        override) produces float16 logits, and `corn_loss` used to do
        `tf.convert_to_tensor(logits, dtype=tf.float32)`, which does NOT cast an already-a-
        tensor input of a different dtype -- it raised `ValueError: Tensor conversion
        requested dtype float32 for Tensor with dtype float16`. Must not raise, and must match
        the float32 computation on the same values (within float16's own precision)."""
        logits_f32 = tf.constant([[0.3, -0.2, 1.1, -0.7], [2.0, 1.5, -1.0, 0.5]], dtype=tf.float32)
        logits_f16 = tf.cast(logits_f32, tf.float16)
        grades = tf.constant([2, 4])

        loss_f16 = corn.corn_loss(logits_f16, grades)
        loss_f32 = corn.corn_loss(logits_f32, grades)
        self.assertEqual(loss_f16.dtype, tf.float32)  # loss itself always computed in float32
        self.assertAlmostEqual(float(loss_f16), float(loss_f32), places=3)

    def test_loss_is_differentiable_with_respect_to_logits(self):
        logits = tf.Variable(tf.random.normal((3, 4)))
        grades = tf.constant([0, 2, 4])
        with tf.GradientTape() as tape:
            loss = corn.corn_loss(logits, grades)
        grad = tape.gradient(loss, logits)
        self.assertIsNotNone(grad)
        self.assertTrue(np.all(np.isfinite(grad.numpy())))
        self.assertTrue(np.any(grad.numpy() != 0.0))


class DecodeLogitsTests(unittest.TestCase):
    def test_all_high_logits_predict_max_grade(self):
        decoded = corn.decode_logits(np.array([[10.0, 10.0, 10.0, 10.0]]))
        self.assertEqual(decoded["predicted_grade"][0], 4)

    def test_all_low_logits_predict_grade_zero(self):
        decoded = corn.decode_logits(np.array([[-10.0, -10.0, -10.0, -10.0]]))
        self.assertEqual(decoded["predicted_grade"][0], 0)

    def test_partial_high_logits_predict_intermediate_grade(self):
        decoded = corn.decode_logits(np.array([[10.0, 10.0, -10.0, -10.0]]))
        self.assertEqual(decoded["predicted_grade"][0], 2)

    def test_predicted_grade_always_in_valid_range(self):
        rng = np.random.RandomState(0)
        logits = rng.uniform(-20, 20, size=(200, 4))
        decoded = corn.decode_logits(logits)
        self.assertTrue(np.all(decoded["predicted_grade"] >= 0))
        self.assertTrue(np.all(decoded["predicted_grade"] <= 4))

    def test_sigmoid_and_cumulative_product_match_hand_computation(self):
        logits = np.array([[0.0, 2.0, -1.0, 0.5]])
        decoded = corn.decode_logits(logits)
        expected_p_cond = 1.0 / (1.0 + np.exp(-logits[0]))
        expected_p_cum = np.cumprod(expected_p_cond)
        np.testing.assert_allclose(decoded["p_cond"][0], expected_p_cond, atol=1e-5)
        np.testing.assert_allclose(decoded["p_cum"][0], expected_p_cum, atol=1e-5)

    def test_cumulative_probabilities_are_monotonically_non_increasing(self):
        rng = np.random.RandomState(1)
        logits = rng.uniform(-5, 5, size=(50, 4))
        decoded = corn.decode_logits(logits)
        p_cum = decoded["p_cum"]
        for row in p_cum:
            self.assertTrue(np.all(np.diff(row) <= 1e-9), f"p_cum not monotonic: {row}")

    def test_probability_reconstruction_matches_formula(self):
        logits = np.array([[0.3, -0.2, 1.1, -0.7]])
        decoded = corn.decode_logits(logits)
        p_cum = decoded["p_cum"][0]
        expected = np.array([
            1 - p_cum[0],
            p_cum[0] - p_cum[1],
            p_cum[1] - p_cum[2],
            p_cum[2] - p_cum[3],
            p_cum[3],
        ])
        np.testing.assert_allclose(decoded["class_probabilities"][0], expected, atol=1e-5)

    def test_probabilities_sum_to_approximately_one(self):
        rng = np.random.RandomState(2)
        logits = rng.uniform(-10, 10, size=(100, 4))
        decoded = corn.decode_logits(logits)
        sums = decoded["class_probabilities"].sum(axis=-1)
        np.testing.assert_allclose(sums, np.ones(100), atol=1e-4)

    def test_probabilities_are_non_negative(self):
        rng = np.random.RandomState(3)
        logits = rng.uniform(-10, 10, size=(100, 4))
        decoded = corn.decode_logits(logits)
        self.assertTrue(np.all(decoded["class_probabilities"] >= -1e-6))

    def test_batch_independence(self):
        """Decoding one row of a batch must be identical to decoding it
        alone -- no cross-example leakage in the vectorized implementation."""
        rng = np.random.RandomState(4)
        logits = rng.uniform(-8, 8, size=(6, 4))
        batch_decoded = corn.decode_logits(logits)
        for i in range(6):
            single_decoded = corn.decode_logits(logits[i:i + 1])
            self.assertEqual(batch_decoded["predicted_grade"][i], single_decoded["predicted_grade"][0])
            np.testing.assert_allclose(
                batch_decoded["class_probabilities"][i], single_decoded["class_probabilities"][0], atol=1e-6,
            )


class CORNQuadraticWeightedKappaTests(unittest.TestCase):
    """Covers `corn.CORNQuadraticWeightedKappa` -- the CORN-aware Keras QWK metric used for
    joint-training checkpoint selection (`JOINT_TRAINING_ARCHITECTURE.md` Sec 23). Confusion-
    matrix accumulation and kappa computation are inherited, unmodified, from
    `training.metrics.QuadraticWeightedKappa` -- these tests focus on the new logits->grade
    decode and on the metric's end-to-end numerical correctness against an independent
    reference, not on re-verifying kappa math already covered elsewhere."""

    @staticmethod
    def _logits_for_grade(grades, magnitude=15.0):
        """Builds `(len(grades), 4)` logits that the sigmoid->cumprod->threshold rule decodes
        EXACTLY to `grades` -- large-magnitude logits push p_cond to ~0/~1 so the cumulative
        product is unambiguous."""
        grades = np.asarray(grades)
        thresholds = np.arange(4)[None, :]
        return np.where(thresholds < grades[:, None], magnitude, -magnitude).astype("float32")

    @staticmethod
    def _reference_qwk(y_true, y_pred):
        from sklearn.metrics import cohen_kappa_score
        return cohen_kappa_score(y_true, y_pred, weights="quadratic")

    def test_metric_initialization(self):
        metric = corn.CORNQuadraticWeightedKappa()
        self.assertEqual(metric.name, "QWK")
        self.assertEqual(metric.num_classes, 5)
        self.assertEqual(metric.confusion.shape, (5, 5))
        np.testing.assert_array_equal(metric.confusion.numpy(), np.zeros((5, 5)))

    def test_decode_matches_decode_logits_all_high(self):
        logits = tf.constant([[10.0, 10.0, 10.0, 10.0]])
        expected_grade = corn.decode_logits(logits.numpy())["predicted_grade"][0]
        self.assertEqual(expected_grade, 4)
        metric = corn.CORNQuadraticWeightedKappa()
        metric.update_state(tf.constant([4]), logits)
        self.assertEqual(metric.confusion.numpy()[4, 4], 1.0)

    def test_decode_matches_decode_logits_all_low(self):
        logits = tf.constant([[-10.0, -10.0, -10.0, -10.0]])
        expected_grade = corn.decode_logits(logits.numpy())["predicted_grade"][0]
        self.assertEqual(expected_grade, 0)
        metric = corn.CORNQuadraticWeightedKappa()
        metric.update_state(tf.constant([0]), logits)
        self.assertEqual(metric.confusion.numpy()[0, 0], 1.0)

    def test_decode_matches_decode_logits_on_random_batch(self):
        """The metric's internal decode must match `decode_logits` exactly, sample by sample,
        for a batch spanning intermediate values -- proves no second, incompatible decoding rule
        was introduced. Feeding each sample's OWN `decode_logits`-derived grade as `y_true`
        forces a purely-diagonal confusion matrix if and only if the metric decoded identically."""
        rng = np.random.RandomState(0)
        logits_np = rng.uniform(-15, 15, size=(50, 4)).astype("float32")
        expected_grades = corn.decode_logits(logits_np)["predicted_grade"]

        metric = corn.CORNQuadraticWeightedKappa()
        metric.update_state(tf.constant(expected_grades, dtype=tf.int32), tf.constant(logits_np))
        confusion = metric.confusion.numpy()
        off_diagonal = confusion.sum() - np.trace(confusion)
        self.assertEqual(off_diagonal, 0.0)

    def test_grade_4_is_reachable(self):
        """The exact bug this metric fixes: naively argmaxing 4 threshold logits can never
        produce class index 4. Proves grade 4 IS reachable through the correct decode."""
        logits = tf.constant([[5.0, 5.0, 5.0, 5.0], [5.0, 5.0, 5.0, 5.0]])
        metric = corn.CORNQuadraticWeightedKappa()
        metric.update_state(tf.constant([4, 4]), logits)
        self.assertEqual(metric.confusion.numpy()[4, 4], 2.0)

    def test_reset_state(self):
        metric = corn.CORNQuadraticWeightedKappa()
        metric.update_state(tf.constant([4]), tf.constant([[10.0, 10.0, 10.0, 10.0]]))
        self.assertGreater(float(tf.reduce_sum(metric.confusion)), 0.0)
        metric.reset_state()
        np.testing.assert_array_equal(metric.confusion.numpy(), np.zeros((5, 5)))
        self.assertEqual(float(metric.result()), 0.0)

    def test_accepts_float16_logits(self):
        """Same mixed-precision dtype-safety fix as `corn_loss` -- CORN's raw output is
        float16 under `mixed_float16`, and `update_state` used to do
        `tf.convert_to_tensor(y_pred, dtype=tf.float32)`, which raises on an already-a-tensor
        float16 input instead of casting it. Must not raise, and must decode/accumulate
        identically to the float32 equivalent."""
        logits_f32 = tf.constant([[10.0, 10.0, 10.0, 10.0], [-10.0, -10.0, -10.0, -10.0]])
        logits_f16 = tf.cast(logits_f32, tf.float16)

        metric = corn.CORNQuadraticWeightedKappa()
        metric.update_state(tf.constant([4, 0]), logits_f16)
        self.assertEqual(metric.confusion.numpy()[4, 4], 1.0)
        self.assertEqual(metric.confusion.numpy()[0, 0], 1.0)

    def test_metric_shape_and_dtype(self):
        metric = corn.CORNQuadraticWeightedKappa()
        self.assertEqual(metric.confusion.shape, (5, 5))
        result = metric.result()
        self.assertEqual(result.shape, ())
        self.assertEqual(result.dtype, tf.float32)

    def test_perfect_predictions_give_kappa_of_one(self):
        grades = [0, 1, 2, 3, 4, 0, 1, 2, 3, 4]
        logits_np = self._logits_for_grade(grades)
        metric = corn.CORNQuadraticWeightedKappa()
        metric.update_state(tf.constant(grades), tf.constant(logits_np))
        self.assertAlmostEqual(float(metric.result()), 1.0, places=5)

    def test_completely_different_predictions_matches_reference(self):
        """A systematic full reversal (0<->4, 1<->3, ...) -- the sign/magnitude is verified
        against the independent reference, not assumed."""
        y_true = np.array([0, 0, 1, 1, 2, 2, 3, 3, 4, 4])
        reversed_grades = 4 - y_true
        logits_np = self._logits_for_grade(reversed_grades)
        y_pred = corn.decode_logits(logits_np)["predicted_grade"]
        np.testing.assert_array_equal(y_pred, reversed_grades)

        metric = corn.CORNQuadraticWeightedKappa()
        metric.update_state(tf.constant(y_true, dtype=tf.int32), tf.constant(logits_np))
        expected = self._reference_qwk(y_true, y_pred)
        self.assertAlmostEqual(float(metric.result()), expected, places=5)
        self.assertLess(float(metric.result()), 0.0, "a systematic full reversal must score below chance")

    def test_matches_independent_reference_on_mixed_grades(self):
        """Cross-checks the full metric (decode + kappa) against sklearn's
        cohen_kappa_score(weights='quadratic') computed independently on the SAME decoded
        grades."""
        rng = np.random.RandomState(1)
        y_true = rng.randint(0, 5, size=30)
        logits_np = rng.uniform(-15, 15, size=(30, 4)).astype("float32")
        y_pred = corn.decode_logits(logits_np)["predicted_grade"]

        metric = corn.CORNQuadraticWeightedKappa()
        metric.update_state(tf.constant(y_true, dtype=tf.int32), tf.constant(logits_np))
        expected = self._reference_qwk(y_true, y_pred)
        self.assertAlmostEqual(float(metric.result()), expected, places=5)

    def test_batch_accumulation_uses_the_combined_confusion_matrix_not_an_average(self):
        """Two separate update_state calls must accumulate into ONE confusion matrix -- result()
        computed from the combined counts, never an average of two independently-computed
        per-batch kappas (exactly what a naive stateless metric would get wrong)."""
        rng = np.random.RandomState(2)
        y_true_all = rng.randint(0, 5, size=40)
        logits_all = rng.uniform(-15, 15, size=(40, 4)).astype("float32")

        metric = corn.CORNQuadraticWeightedKappa()
        metric.update_state(tf.constant(y_true_all[:20], dtype=tf.int32), tf.constant(logits_all[:20]))
        metric.update_state(tf.constant(y_true_all[20:], dtype=tf.int32), tf.constant(logits_all[20:]))
        accumulated_result = float(metric.result())

        y_pred_all = corn.decode_logits(logits_all)["predicted_grade"]
        expected = self._reference_qwk(y_true_all, y_pred_all)
        self.assertAlmostEqual(accumulated_result, expected, places=5)


class ModelBatchIndependenceTests(unittest.TestCase):
    def test_model_forward_pass_is_batch_independent(self):
        model = corn.build_corn_model()
        rng = np.random.RandomState(5)
        f_batch = rng.normal(size=(4, 256)).astype("float32")
        batch_out = model.predict(f_batch, verbose=0)
        for i in range(4):
            single_out = model.predict(f_batch[i:i + 1], verbose=0)
            np.testing.assert_allclose(batch_out[i], single_out[0], atol=1e-5)


class SerializationTests(unittest.TestCase):
    def test_full_keras_save_load_round_trip(self):
        tmp_dir = tempfile.mkdtemp(prefix="corn_serialization_")
        try:
            model = corn.build_corn_model()
            f = np.random.RandomState(6).normal(size=(3, 256)).astype("float32")
            original_output = model.predict(f, verbose=0)

            path = os.path.join(tmp_dir, "corn.keras")
            model.save(path)
            loaded = tf.keras.models.load_model(path, compile=False)
            loaded_output = loaded.predict(f, verbose=0)

            np.testing.assert_allclose(original_output, loaded_output, atol=1e-6)
            self.assertEqual(
                [w.shape for w in model.get_weights()], [w.shape for w in loaded.get_weights()],
            )
        finally:
            shutil.rmtree(tmp_dir, ignore_errors=True)


class CORNStageTests(unittest.TestCase):
    def setUp(self):
        self.stage = corn.CORNStage()
        self.stage.build()

    def test_train_raises_not_implemented(self):
        with self.assertRaises(NotImplementedError):
            self.stage.train(None)

    def test_evaluate_raises_not_implemented(self):
        with self.assertRaises(NotImplementedError):
            self.stage.evaluate(None)

    def test_predict_returns_classification_stage_contract(self):
        f = np.zeros(256, dtype="float32")
        result = self.stage.predict(f)
        self.assertIn("label", result)
        self.assertIn("class_index", result)
        self.assertIn("confidence", result)
        self.assertIn("probabilities", result)
        self.assertIn(result["label"], corn.GRADE_NAMES)
        self.assertIsInstance(result["class_index"], int)
        self.assertEqual(len(result["probabilities"]), 5)

    def test_predict_batch_matches_predict_per_item(self):
        rng = np.random.RandomState(7)
        inputs = [rng.normal(size=256).astype("float32") for _ in range(3)]
        batch_results = self.stage.predict_batch(inputs)
        for x, single_result in zip(inputs, batch_results):
            individual_result = self.stage.predict(x)
            self.assertEqual(individual_result["class_index"], single_result["class_index"])
            self.assertAlmostEqual(individual_result["confidence"], single_result["confidence"], places=5)

    def test_predict_before_build_or_load_raises(self):
        fresh_stage = corn.CORNStage()
        with self.assertRaises(RuntimeError):
            fresh_stage.predict(np.zeros(256, dtype="float32"))

    def test_save_load_round_trip_via_stage(self):
        tmp_dir = tempfile.mkdtemp(prefix="corn_stage_serialization_")
        try:
            path = os.path.join(tmp_dir, "corn_stage.keras")
            self.stage.save(path)
            new_stage = corn.CORNStage().load(path)
            f = np.random.RandomState(8).normal(size=256).astype("float32")
            original = self.stage.predict(f)
            reloaded = new_stage.predict(f)
            self.assertEqual(original["class_index"], reloaded["class_index"])
        finally:
            shutil.rmtree(tmp_dir, ignore_errors=True)


class DatasetContractTests(unittest.TestCase):
    """CORN must use the authoritative split -- no second split, no
    test.csv labels, no IDRiD grading labels."""

    def test_corn_module_imports_no_second_split_utility(self):
        """corn.py has no split_train_val_ids/compute_split/etc. of its
        own -- it does not define a dataset split at all."""
        import inspect
        source = inspect.getsource(corn)
        for forbidden in ("train_test_split", "split_train_val", "stratify"):
            self.assertNotIn(forbidden, source)

    def test_corn_module_never_reads_test_csv(self):
        import inspect
        source = inspect.getsource(corn)
        self.assertNotIn("test.csv", source)

    def test_corn_module_never_reads_idrid_grading(self):
        import inspect
        source = inspect.getsource(corn)
        self.assertNotIn("grading", source.lower())
        self.assertNotIn("idrid", source.lower())

    def test_authoritative_manifest_is_the_only_split_source(self):
        """corn.py itself doesn't call downstream_split -- by design its
        only input is F (no raw images), so it has nothing to split. This
        test instead confirms the authoritative manifest downstream_split
        owns is intact and CORN introduces no competing one."""
        train_entries, val_entries = downstream_split.get_authoritative_split()
        self.assertEqual(len(train_entries), 2929)
        self.assertEqual(len(val_entries), 733)
        train_ids = {i for i, _ in train_entries}
        val_ids = {i for i, _ in val_entries}
        self.assertEqual(train_ids & val_ids, set())


class GroundTruthMaskBoundaryTests(unittest.TestCase):
    def test_no_ground_truth_or_dice_iou_identifiers_in_source(self):
        """Tokenize-based check (code identifiers only, never docstring
        prose) -- mirrors racaf.py's RACAFBoundaryTests pattern."""
        import ast
        import tokenize
        from io import BytesIO

        with open(corn.__file__, "rb") as handle:
            tokens = list(tokenize.tokenize(handle.readline))
        names = {tok.string for tok in tokens if tok.type == tokenize.NAME}
        # Specific ground-truth/metric identifiers only -- generic terms
        # like "mask" are legitimate, unrelated vocabulary here (corn_loss's
        # own boolean inclusion mask, y>=k), not a sign of reading a
        # segmentation ground-truth mask.
        forbidden = {
            "ground_truth", "groundtruth", "dice", "iou",
            "lesion_mask", "vessel_mask", "segmentation_mask", "probability_map", "probability_maps",
        }
        offending = names & forbidden
        self.assertEqual(offending, set(), f"forbidden identifiers found in corn.py code: {offending}")

        tree = ast.parse(open(corn.__file__, encoding="utf-8").read())
        imported_names = set()
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                imported_names.update(alias.name for alias in node.names)
            elif isinstance(node, ast.ImportFrom) and node.module:
                imported_names.add(node.module)
        for forbidden_module in ("lesion_segmentation_dataset", "lesion_segmentation_model"):
            self.assertNotIn(forbidden_module, imported_names)


class RacafToCornContractTests(unittest.TestCase):
    """Real, lightweight (checkpoint-free) end-to-end: Stage 06 -> Stage 07
    -> RACAF -> CORN, proving CORN's input contract matches RACAF's actual
    output -- not just an assumed shape. No training, no checkpoint."""

    def test_real_racaf_output_feeds_corn_directly(self):
        import feature_fusion
        import racaf

        batch_size = 2
        rng = np.random.RandomState(9)

        stage7_model = feature_fusion.build_adaptive_cross_attention()
        local_features = rng.normal(size=(batch_size, 32, 32, 256)).astype("float32")
        global_features = rng.normal(size=(batch_size, 64, 1152)).astype("float32")
        e = stage7_model.predict([local_features, global_features], verbose=0)
        self.assertEqual(e.shape, (batch_size, 256))

        racaf_model = racaf.build_racaf_fusion()
        r = rng.uniform(0, 1, size=(batch_size, 1)).astype("float32")
        f = racaf_model.predict([e, global_features, r], verbose=0)
        self.assertEqual(f.shape, (batch_size, 256))

        corn_model = corn.build_corn_model()
        logits = corn_model.predict(f, verbose=0)
        self.assertEqual(logits.shape, (batch_size, 4))

        decoded = corn.decode_logits(logits)
        self.assertTrue(np.all(decoded["predicted_grade"] >= 0))
        self.assertTrue(np.all(decoded["predicted_grade"] <= 4))
        np.testing.assert_allclose(decoded["class_probabilities"].sum(axis=-1), np.ones(batch_size), atol=1e-4)


if __name__ == "__main__":
    unittest.main()
