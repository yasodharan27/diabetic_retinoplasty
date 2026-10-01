"""CPU unit tests for arch1_model.py (Stage 5 prior encoder + Stage 6/7 ConvNeXt-T with gated prior
injection + Stage 8 CORN). As in tests/test_pl_convnext.py, a randomly initialised Keras
ConvNeXtTiny with non-trivial weights stands in for the pinned ImageNet reference."""
import os
import unittest

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "-1")

import keras
import numpy as np

import arch1_model as a1
import pl_convnext as pl
import stage34_cache_v2 as cache

SIZE = 64
CHANNELS = cache.channel_names(("MA", "HE", "EX", "SE"))


def _reference():
    keras.utils.set_random_seed(7)
    from keras.applications import ConvNeXtTiny
    ref = ConvNeXtTiny(include_top=False, weights=None, include_preprocessing=True, pooling="avg",
                       input_shape=(SIZE, SIZE, 3), name=pl.BACKBONE_NAME)
    for v in ref.weights:
        if "prestem" not in v.path:
            v.assign(np.random.default_rng(len(v.path)).normal(0, 0.05, v.shape).astype("float32")
                     + (1.0 if v.path.endswith("gamma") else 0.0))
    return ref, [np.asarray(v.numpy()) for v in ref.weights if "prestem" not in v.path]


REF, REF_ARRAYS = _reference()
MODEL = a1.build_arch1_model(CHANNELS, 42, REF, image_size=SIZE)


def _inputs(n=2, seed=0, k2=len(CHANNELS)):
    rng = np.random.default_rng(seed)
    return {"rgb": rng.uniform(0, 1, (n, SIZE, SIZE, 3)).astype("float32"),
            "vessel": rng.uniform(0, 1, (n, SIZE, SIZE, 1)).astype("float32"),
            "pathology": rng.uniform(0, 1, (n, SIZE, SIZE, k2)).astype("float32")}


class StructureTests(unittest.TestCase):
    def test_parameter_counts(self):
        report = a1.parameter_report(MODEL)
        self.assertEqual(report["backbone"], pl.EXPECTED_BACKBONE_PARAMETERS["P"])
        self.assertEqual(report["head"], pl.EXPECTED_HEAD_PARAMETERS)
        self.assertTrue(1.1e6 < report["prior_encoder"] < 1.5e6, report)
        injection = sum(2 * (p * d + d) + 2 * d for p, d in zip(a1.PRIOR_DIMS, a1.DIMS))
        self.assertEqual(report["injection"], injection)
        self.assertEqual(report["total"], sum(v for k, v in report.items() if k != "total"))
        self.assertTrue(29e6 < report["total"] < 31e6, report)

    def test_k_configurable_pathology_stem(self):
        six = cache.channel_names(("MA", "HE", "EX", "SE", "NV", "IRMA"))
        model = a1.build_arch1_model(six, 42, None, image_size=SIZE)
        kernel = model.get_layer("prior_pathology_stem_conv").kernel
        self.assertEqual(tuple(kernel.shape), (3, 3, 12, 32))
        self.assertEqual(model.predict_on_batch(_inputs(k2=12)).shape, (2, 4))
        with self.assertRaises(cache.ManifestMismatchError):
            a1.build_arch1_model(("MA:max", "MA:mean"), 42, None, image_size=SIZE)

    def test_pyramid_and_injection_shapes_at_512(self):
        model = a1.build_arch1_model(CHANNELS, 42, None, image_size=512)
        expected = [(128, 32), (64, 64), (32, 128), (16, 256)]
        for i, (side, ch) in enumerate(expected, start=1):
            self.assertEqual(tuple(model.get_layer(f"prior_down_{i}_out").output.shape),
                             (None, side, side, ch))
            self.assertEqual(tuple(model.get_layer(f"stage7_injection_{i}").output.shape),
                             (None, side, side, a1.DIMS[i - 1]))
        self.assertEqual(tuple(model.get_layer("prior_concat").output.shape), (None, 256, 256, 48))


class PretrainedCopyTests(unittest.TestCase):
    def test_every_backbone_layer_equals_the_reference(self):
        ref_layers = {l.name: l for l in REF.layers if l.weights and "prestem" not in l.name}
        for layer in a1.backbone_layers(MODEL):
            name = layer.name if layer.name != a1.HEAD_NORM_NAME else [
                n for n in ref_layers if n.startswith("layer_normalization")][0]
            for mine, theirs in zip(layer.get_weights(), ref_layers[name].get_weights()):
                np.testing.assert_array_equal(mine, theirs)
        self.assertEqual(MODEL.arch1_copy_report["copied_arrays"], len(REF_ARRAYS))

    def test_rebuilt_backbone_matches_keras_convnext(self):
        features = keras.Model(MODEL.inputs, MODEL.get_layer(a1.HEAD_NORM_NAME).output)
        x = _inputs(seed=3)
        mine = features.predict_on_batch(x)
        theirs = REF.predict_on_batch(x["rgb"] * 255.0)
        np.testing.assert_allclose(mine, theirs, rtol=0, atol=pl.REFERENCE_PARITY_TOL)


class ZeroInitEquivalenceTests(unittest.TestCase):
    def test_initial_model_equals_P(self):
        p = pl.build_pl_model("P", 42, REF_ARRAYS, image_size=SIZE)
        x = _inputs(seed=1)
        stage5 = np.concatenate([x["rgb"], np.zeros((2, SIZE, SIZE, 5), "float32")], axis=-1)
        p_logits = p.predict_on_batch([stage5, np.zeros((2, 256, 256, 3), "float32"),
                                       np.zeros((2, 1), "float32")])
        np.testing.assert_allclose(MODEL.predict_on_batch(x), p_logits, rtol=0,
                                   atol=pl.PL_P_EQUIVALENCE_TOL)
        np.testing.assert_array_equal(MODEL.get_layer("corn").get_layer("corn_logits").get_weights()[0],
                                      pl.corn_head_initial_weights(42)[0])

    def test_priors_have_exactly_no_influence_at_init(self):
        x = _inputs(seed=2)
        y = dict(x, vessel=1.0 - x["vessel"], pathology=np.zeros_like(x["pathology"]))
        np.testing.assert_array_equal(MODEL.predict_on_batch(x), MODEL.predict_on_batch(y))
        for layer in a1.injection_layers(MODEL):
            self.assertFalse(layer.alpha.numpy().any() or layer.gamma.numpy().any())

    def test_injection_is_wired_and_gates_receive_gradient(self):
        import tensorflow as tf
        model = a1.build_arch1_model(CHANNELS, 42, REF, image_size=SIZE)
        x = {k: tf.constant(v) for k, v in _inputs(seed=4).items()}
        gates = [w for l in a1.injection_layers(model) for w in (l.alpha, l.gamma)]
        with tf.GradientTape() as tape:
            loss = tf.reduce_sum(model(x, training=True))
        grads = tape.gradient(loss, gates)
        self.assertTrue(all(g is not None and float(tf.reduce_sum(tf.abs(g))) > 0 for g in grads))
        for layer in a1.injection_layers(model):
            layer.gamma.assign(np.full(layer.gamma.shape, 0.5, "float32"))
        y = dict(_inputs(seed=4), pathology=np.zeros((2, SIZE, SIZE, 8), "float32"))
        self.assertGreater(np.abs(model.predict_on_batch(_inputs(seed=4))
                                  - model.predict_on_batch(y)).max(), 1e-4)


if __name__ == "__main__":
    unittest.main()
