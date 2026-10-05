"""CPU tests for the E1 multi-task grader (e1_model, e1_data, e1_train, e1_gates). Random ConvNeXt reference
weights and the 5-image synthetic bundle only: nothing here is an E1 experiment and nothing produces a result.
The one-epoch run exists to prove the checkpoint / history / log-alias mechanics."""
import itertools
import os
import tempfile
import unittest

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "-1")

import keras
import numpy as np
import tensorflow as tf

import arch1_data as ad
import arch1_model as a1
import e1_data as ed
import e1_gates as eg
import e1_model as em
import e1_train as et
import pipeline_v2_config as v2cfg
import pl_convnext as pl
import stage34_cache_v2 as cache
import weighted_corn

try:
    from tests import v2_bundle_fixture as fx
except ImportError:
    import v2_bundle_fixture as fx

CW = list(weighted_corn.PREREGISTERED_CLASS_WEIGHTS)
CHANNELS = cache.channel_names(v2cfg.STAGE4_V2A_CLASSES)
PRIOR = (0.03, 0.10, 0.20, 0.00)                # SE prior 0 exercises the clip
SMALL = 64                                       # feature map 2 x 2: fast structure / numerics tests


def _reference(size):
    keras.utils.set_random_seed(7)
    from keras.applications import ConvNeXtTiny
    ref = ConvNeXtTiny(include_top=False, weights=None, include_preprocessing=True, pooling="avg",
                       input_shape=(size, size, 3), name=pl.BACKBONE_NAME)
    for v in ref.weights:
        if "prestem" not in v.path:
            v.assign(np.random.default_rng(len(v.path)).normal(0, 0.05, v.shape).astype("float32")
                     + (1.0 if v.path.endswith("gamma") else 0.0))
    return ref, [np.asarray(v.numpy()) for v in ref.weights if "prestem" not in v.path]


REF, REF_ARRAYS = _reference(SMALL)
MODEL = em.build_e1_model(42, PRIOR, REF, image_size=SMALL)


def _rgb(n=2, seed=0, size=SMALL):
    return np.random.default_rng(seed).uniform(0, 1, (n, size, size, 3)).astype("float32")


def _pathology(seed=0, size=512):
    return cache.from_uint8(np.random.default_rng(seed).integers(0, 256, (size, size, 8), dtype=np.uint8))


def _batch(n=2, seed=0, size=SMALL):
    rng = np.random.default_rng(seed)
    grid = size // em.BACKBONE_STRIDE
    return (_rgb(n, seed, size), rng.integers(0, 5, n).astype("int32"),
            rng.uniform(0, 1, (n, grid, grid, 4)).astype("float32"))


def _bce(t, z):
    t, z = np.asarray(t, np.float64), np.asarray(z, np.float64)
    p = 1.0 / (1.0 + np.exp(-z))
    with np.errstate(divide="ignore", invalid="ignore"):
        terms = -(np.where(t > 0, t * np.log(p), 0.0) + np.where(t < 1, (1 - t) * np.log1p(-p), 0.0))
    return float(np.mean(terms))


class ModelTests(unittest.TestCase):
    def test_01_construction(self):
        self.assertEqual([i.name for i in MODEL.inputs], ["rgb"])
        self.assertEqual(tuple(MODEL.output_names), (em.GRADING_OUTPUT, em.LESION_OUTPUT))
        names = [l.name for l in MODEL.layers]
        self.assertFalse([n for n in names if n.startswith(("prior_", "stage7_injection_")) or "vessel" in n])
        self.assertIn(em.FEATURE_MAP_NAME, names)
        conv = MODEL.get_layer(em.LESION_CONV_NAME)
        self.assertIs(conv.input, MODEL.get_layer(em.FEATURE_MAP_NAME).output)       # the tensor the grader pools
        self.assertIs(MODEL.get_layer(f"{a1.BACKBONE_NAME}_gap").input, MODEL.get_layer(em.FEATURE_MAP_NAME).output)
        self.assertEqual(MODEL.e1_copy_report["copied_arrays"], len(REF_ARRAYS))

    def test_02_parameter_count_is_p_plus_3076(self):
        report = em.parameter_report(MODEL)
        p = pl.build_pl_model("P", 42, REF_ARRAYS, image_size=SMALL)
        self.assertEqual(em.ADDED_PARAMETERS, 3076)
        self.assertEqual(report["lesion_head"], 3076)
        self.assertEqual(report["total_trainable"], pl.trainable_parameter_count(p) + 3076)
        self.assertEqual(report["total_trainable"], report["p_total"] + 3076)
        self.assertEqual(report["non_trainable"], 0)

    def test_03_04_output_shapes_and_dtype_at_512(self):
        model = em.build_e1_model(42, PRIOR, image_size=512)
        self.assertEqual(tuple(model.outputs[0].shape), (None, 4))
        self.assertEqual(tuple(model.outputs[1].shape), (None, 16, 16, 4))
        self.assertEqual(tuple(model.get_layer(em.FEATURE_MAP_NAME).output.shape), (None, 16, 16, 768))
        self.assertEqual(model.outputs[1].dtype, "float32")
        grading, lesion = model.predict_on_batch({"rgb": _rgb(1, 5, 512)})          # test 14: one forward pass
        self.assertEqual((grading.shape, lesion.shape), ((1, 4), (1, 16, 16, 4)))
        self.assertEqual(lesion.dtype, np.float32)
        self.assertTrue(np.isfinite(grading).all() and np.isfinite(lesion).all())

    def test_bias_is_the_clipped_prior_logit_and_kernel_is_seeded(self):
        kernel, bias = MODEL.get_layer(em.LESION_CONV_NAME).get_weights()
        clipped = np.clip(np.asarray(PRIOR, np.float64), *em.PRIOR_CLIP)
        np.testing.assert_allclose(bias, np.log(clipped / (1 - clipped)), rtol=1e-6)
        self.assertAlmostEqual(float(bias[3]), float(np.log(1e-4 / (1 - 1e-4))), places=4)
        other = em.build_e1_model(42, PRIOR, image_size=SMALL)
        np.testing.assert_array_equal(kernel, other.get_layer(em.LESION_CONV_NAME).get_weights()[0])
        for bad in ((0.1, 0.1, 0.1), (0.1, 0.1, 0.1, 1.5), (0.1, 0.1, 0.1, float("nan"))):
            with self.assertRaises(ValueError):
                em.prior_logits(bad)

    def test_13_initialisation_equals_p(self):
        p = pl.build_pl_model("P", 42, REF_ARRAYS, image_size=SMALL)
        x = _rgb(2, 1)
        grading, _ = MODEL.predict_on_batch({"rgb": x})
        np.testing.assert_allclose(grading, p.predict_on_batch(eg.p_inputs(x)), rtol=0, atol=pl.PL_P_EQUIVALENCE_TOL)
        np.testing.assert_array_equal(MODEL.get_layer("corn").get_layer("corn_logits").get_weights()[0],
                                      pl.corn_head_initial_weights(42)[0])
        np.testing.assert_allclose(em.grading_model(MODEL).predict_on_batch(x), grading, rtol=0, atol=0)

    def test_12_p_checkpoint_loads_and_every_array_is_copied_once(self):
        from training import checkpointing as ckpt
        p = pl.build_pl_model("P", 123, REF_ARRAYS, image_size=SMALL)
        rng = np.random.default_rng(11)
        for v in p.trainable_variables:                                   # stand-in for a trained checkpoint
            v.assign(v.numpy() + rng.normal(0, 0.02, v.shape).astype("float32"))
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, ckpt.MODEL_WEIGHTS_FILENAME)
            ckpt.save_model_weights_only(p, path)
            loaded = pl.build_pl_model("P", 123, REF_ARRAYS, image_size=SMALL)
            ckpt.load_model_weights_only(loaded, path)
        e1 = em.build_e1_model(123, PRIOR, image_size=SMALL)              # random backbone before the copy
        before = [w.copy() for w in e1.get_layer(em.LESION_CONV_NAME).get_weights()]
        report = em.copy_from_p(e1, loaded)
        backbone = loaded.get_layer(pl.BACKBONE_NAME)
        self.assertEqual(report["copied_arrays"], len(backbone.weights))
        self.assertEqual(report["corn_arrays"], 2)
        source = {l.name: l for l in backbone.layers if l.weights}
        norm = [n for n in source if n.startswith("layer_normalization")]
        self.assertEqual(len(norm), 1)
        consumed = 0
        for layer in a1.backbone_layers(e1):
            theirs = source[norm[0] if layer.name == a1.HEAD_NORM_NAME else layer.name].get_weights()
            for mine, other in zip(layer.get_weights(), theirs):
                np.testing.assert_array_equal(mine, other)
            consumed += len(theirs)
        self.assertEqual(consumed, len(backbone.weights))                 # nothing skipped, nothing twice
        for mine, other in zip(e1.get_layer("corn").get_weights(), loaded.get_layer("corn").get_weights()):
            np.testing.assert_array_equal(mine, other)
        for mine, other in zip(e1.get_layer(em.LESION_CONV_NAME).get_weights(), before):
            np.testing.assert_array_equal(mine, other)                    # the auxiliary head is not touched
        x = _rgb(3, 4)
        result = eg.compare(np.asarray(loaded.predict_on_batch(eg.p_inputs(x)), np.float64),
                            np.asarray(e1.predict_on_batch({"rgb": x})[0], np.float64))
        self.assertLessEqual(result["logit_max_abs"], eg.FLOAT32_TOL["logit_max_abs"])
        self.assertLessEqual(result["probability_max_abs"], eg.FLOAT32_TOL["probability_max_abs"])
        self.assertTrue(result["grades_equal"])

    def test_copy_refuses_a_backbone_that_does_not_match(self):
        from keras.applications import ConvNeXtSmall
        wrong = ConvNeXtSmall(include_top=False, weights=None, include_preprocessing=False, pooling="avg",
                              input_shape=(SMALL, SMALL, 3), name=pl.BACKBONE_NAME)
        with self.assertRaises(RuntimeError):
            a1.copy_backbone_weights(em.build_e1_model(42, PRIOR, image_size=SMALL), wrong)

    def test_18_checkpoint_round_trip(self):
        from training import checkpointing as ckpt
        source = em.build_e1_model(42, PRIOR, REF, image_size=SMALL)
        rng = np.random.default_rng(5)
        for v in source.trainable_variables:
            v.assign(v.numpy() + rng.normal(0, 0.02, v.shape).astype("float32"))
        x = _rgb(2, 9)
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, ckpt.MODEL_WEIGHTS_FILENAME)
            ckpt.save_model_weights_only(source, path)
            target = em.build_e1_model(7, (0.5, 0.5, 0.5, 0.5), image_size=SMALL)
            ckpt.load_model_weights_only(target, path)
        for a, b in zip(source.get_weights(), target.get_weights()):
            np.testing.assert_array_equal(a, b)
        for a, b in zip(source.predict_on_batch({"rgb": x}), target.predict_on_batch({"rgb": x})):
            np.testing.assert_array_equal(a, b)


class TargetTests(unittest.TestCase):
    def test_05_06_07_shape_range_and_bruteforce_block_maximum(self):
        self.assertEqual(ed.max_channel_indices(CHANNELS), (1, 3, 5, 7))
        for seed in range(3):
            pathology = _pathology(seed)
            target = ed.pool_targets(pathology, CHANNELS)
            self.assertEqual((target.shape, target.dtype), ((16, 16, 4), np.float32))
            self.assertTrue(target.min() >= 0.0 and target.max() <= 1.0)
            np.testing.assert_array_equal(target, ed.pool_targets_bruteforce(pathology, CHANNELS))
            np.testing.assert_array_equal(target, pathology[..., [1, 3, 5, 7]].reshape(16, 32, 16, 32, 4).max(axis=(1, 3)))
            self.assertEqual(float(target[3, 5, 2]), float(pathology[96:128, 160:192, 5].max()))
        sparse = np.zeros((512, 512, 8), np.float32)
        sparse[40, 500, 1] = 0.75                                         # one MA:max pixel -> one cell
        sparse[40, 500, 0] = 0.25                                         # the mean channel is never read
        target = ed.pool_targets(sparse, CHANNELS)
        self.assertEqual(float(target[1, 15, 0]), 0.75)
        self.assertEqual(int((target > 0).sum()), 1)

    def test_wrong_channel_order_or_shape_is_refused(self):
        with self.assertRaises(cache.ManifestMismatchError):
            ed.max_channel_indices(CHANNELS[::-1])
        with self.assertRaises(ValueError):
            ed.pool_targets(np.zeros((500, 500, 8), np.float32), CHANNELS)
        with self.assertRaises(ValueError):
            ed.pool_targets(np.zeros((512, 512, 4), np.float32), CHANNELS)

    def test_08_pooling_is_correct_under_every_flip_and_rotation(self):
        pathology = _pathology(4)
        base = ed.pool_targets(pathology, CHANNELS)
        for flip_h, flip_v, k in itertools.product((False, True), (False, True), range(4)):
            moved, pooled = pathology, base
            if flip_h:
                moved, pooled = moved[:, ::-1, :], pooled[:, ::-1, :]
            if flip_v:
                moved, pooled = moved[::-1, :, :], pooled[::-1, :, :]
            if k:
                moved, pooled = np.rot90(moved, k=k, axes=(0, 1)), np.rot90(pooled, k=k, axes=(0, 1))
            target = ed.pool_targets(np.ascontiguousarray(moved), CHANNELS)
            np.testing.assert_array_equal(target, pooled)                 # pool(transform) == transform(pool)
            np.testing.assert_array_equal(target, ed.pool_targets_bruteforce(np.ascontiguousarray(moved), CHANNELS))

    def test_09_augmentation_moves_rgb_and_targets_together_and_jitters_rgb_only(self):
        import improved_training_data as itd
        import local_feature_extraction_dataset as lfed
        rng0 = np.random.default_rng(2)
        sample = {"rgb": rng0.uniform(0.2, 0.8, (512, 512, 3)).astype("float32"), "pathology": _pathology(6)}
        seen_geometry, intensity_changed = set(), 0
        for epoch in range(12):
            key = (42, epoch, "00000000000a")
            out = ed.augment_inputs(sample, itd.per_image_augmentation_rng(*key))
            # The geometric part alone, replayed with the same RNG: flips / rot90 on the whole stack.
            stack = np.concatenate([sample["rgb"], sample["pathology"]], axis=-1)
            spatial = lfed._augment_spatial(stack, itd.per_image_augmentation_rng(*key))
            np.testing.assert_array_equal(out["pathology"], spatial[..., 3:])      # targets: bit-identical values
            self.assertEqual(out["pathology"].dtype, np.float32)
            np.testing.assert_array_equal(np.sort(out["pathology"].ravel()), np.sort(sample["pathology"].ravel()))
            intensity_changed += int(not np.array_equal(out["rgb"], spatial[..., :3]))
            # The same transform reached RGB: undo the jitter's effect by checking the geometry on a marker.
            marker = np.zeros((512, 512, 11), np.float32)
            marker[10, 20, :] = 1.0
            moved = lfed._augment_spatial(marker, itd.per_image_augmentation_rng(*key))
            self.assertEqual(np.argwhere(moved[..., 0] == 1).tolist(), np.argwhere(moved[..., 5] == 1).tolist())
            seen_geometry.add(tuple(np.argwhere(moved[..., 0] == 1)[0]))
            # Identical to Architecture 1's augmentation of the same image in the same epoch (RGB and maps).
            ref = ad.augment_sample({"rgb": sample["rgb"], "vessel": np.zeros((512, 512, 1), np.float32),
                                     "pathology": sample["pathology"]}, itd.per_image_augmentation_rng(*key))
            np.testing.assert_array_equal(out["rgb"], ref["rgb"])
            np.testing.assert_array_equal(out["pathology"], ref["pathology"])
            np.testing.assert_array_equal(ed.pool_targets(out["pathology"], CHANNELS),
                                          ed.pool_targets_bruteforce(out["pathology"], CHANNELS))
        self.assertGreater(intensity_changed, 0)                           # RGB is jittered
        self.assertGreater(len(seen_geometry), 1)                          # several geometries were exercised


class LossTests(unittest.TestCase):
    def test_10_lesion_loss_matches_an_independent_reference(self):
        rng = np.random.default_rng(0)
        shape = (2, 16, 16, 4)
        cases = {"all-zero target": (np.zeros(shape), rng.normal(0, 2, shape)),
                 "all-one target": (np.ones(shape), rng.normal(0, 2, shape)),
                 "mixed target": (rng.uniform(0, 1, shape), rng.normal(0, 2, shape)),
                 "extreme positive logits": (rng.uniform(0, 1, shape), np.full(shape, 30.0)),
                 "extreme negative logits": (rng.uniform(0, 1, shape), np.full(shape, -30.0))}
        for name, (t, z) in cases.items():
            value = float(et.lesion_loss(t.astype("float32"), z.astype("float32")))
            self.assertTrue(np.isfinite(value), name)
            # The loss is computed and averaged in float32 (as specified): the mean of 2,048 float32 terms agrees
            # with the float64 reference to float32 summation accuracy, and every single term agrees closely.
            self.assertAlmostEqual(value, _bce(t, z), delta=1e-4 * max(1.0, abs(_bce(t, z))), msg=name)
            self.assertAlmostEqual(value, et.lesion_loss_numpy(t, z), delta=1e-4 * max(1.0, value), msg=name)
            t32, z32 = t.astype("float32"), z.astype("float32")
            terms = tf.nn.sigmoid_cross_entropy_with_logits(labels=t32, logits=z32).numpy().astype(np.float64)
            exact = (np.maximum(z32, 0.0).astype(np.float64) - z32.astype(np.float64) * t32.astype(np.float64)
                     + np.log1p(np.exp(-np.abs(z32.astype(np.float64)))))
            np.testing.assert_allclose(terms, exact, rtol=1e-5, atol=1e-6, err_msg=name)
            self.assertAlmostEqual(value, float(np.mean(terms)), delta=1e-4 * max(1.0, value), msg=name)
        for big in (1e4, -1e4):                                            # no overflow at absurd logits
            self.assertTrue(np.isfinite(float(et.lesion_loss(np.full(shape, 0.3, "float32"), np.full(shape, big, "float32")))))
        self.assertEqual(et.lesion_loss(np.zeros(shape, "float32"), tf.zeros(shape, tf.float16)).dtype, tf.float32)
        self.assertAlmostEqual(float(et.lesion_loss(np.zeros(shape, "float32"), np.zeros(shape, "float32"))),
                               float(np.log(2.0)), places=5)          # float32 accuracy

    def test_15_16_each_loss_alone_reaches_every_encoder_variable(self):
        rgb, grades, targets = _batch(2, 3)
        report = eg.gradient_report(MODEL, rgb, grades, targets, CW)
        self.assertTrue(report["PASS"], report["failures"])
        self.assertEqual(report["encoder_variables"], sum(len(l.trainable_weights) for l in em.encoder_layers(MODEL)))
        for name in ("corn", "lesion"):
            self.assertEqual(report[name]["variables_without_gradient"], 0)
            self.assertEqual(report[name]["zero_gradients"], 0)
            self.assertTrue(report[name]["all_finite"])
        self.assertEqual(report["lesion_head_gradient_from_corn_loss"], [False, False])   # beside the grading path
        with tf.GradientTape() as tape:                                    # the lesion loss cannot reach the CORN head
            _, lesion = MODEL({"rgb": rgb}, training=True)
            loss = et.lesion_loss(targets, lesion)
        grading_only = MODEL.get_layer("corn").trainable_weights + MODEL.get_layer(a1.HEAD_NORM_NAME).trainable_weights
        self.assertTrue(all(g is None for g in tape.gradient(loss, grading_only)))

    def test_17_total_loss_is_corn_plus_lesion_and_one_step_trains(self):
        model = et.compile_e1_model(em.build_e1_model(42, PRIOR, REF, image_size=SMALL), CW, 1e-4, 0.05)
        rgb, grades, targets = _batch(4, 8)
        grading, lesion = model.predict_on_batch({"rgb": rgb})
        corn_value = float(weighted_corn.weighted_corn_loss_value(grading, grades, CW))
        lesion_value = et.lesion_loss_numpy(targets, lesion)
        logs = model.test_on_batch({"rgb": rgb}, (grades, targets), return_dict=True)
        self.assertEqual(et.LAMBDA, 1.0)
        self.assertAlmostEqual(float(logs["loss"]), corn_value + lesion_value, places=4)
        self.assertAlmostEqual(float(logs["corn_loss"]), corn_value, places=4)
        self.assertAlmostEqual(float(logs["lesion_logits_loss"]), lesion_value, places=4)
        before = [w.copy() for w in model.get_weights()]
        out = model.train_on_batch({"rgb": rgb}, (grades, targets), return_dict=True)      # test 15: backward pass
        self.assertTrue(np.isfinite(float(out["loss"])))
        changed = sum(not np.array_equal(a, b) for a, b in zip(before, model.get_weights()))
        self.assertGreater(changed, len(before) // 2)
        self.assertTrue(all(np.isfinite(w).all() for w in model.get_weights()))

    def test_optimizer_is_p_adamw_with_no_decay_on_1d_parameters(self):
        model = et.compile_e1_model(em.build_e1_model(42, PRIOR, image_size=SMALL), CW, 1e-4, 0.05)
        optimizer = pl.inner_optimizer(model)
        self.assertEqual(type(optimizer).__name__, "AdamW")
        self.assertAlmostEqual(float(optimizer.learning_rate.numpy()), 1e-4, places=9)
        self.assertAlmostEqual(float(optimizer.weight_decay), 0.05, places=9)


class MixedPrecisionTests(unittest.TestCase):
    def test_11_mixed_precision_dtypes_and_finite_outputs(self):
        from training import trainer as tr
        previous = keras.mixed_precision.global_policy().name
        keras.mixed_precision.set_global_policy("mixed_float16")
        try:
            model = em.build_e1_model(42, PRIOR, image_size=SMALL)
            self.assertEqual(model.outputs[0].dtype, "float16")                   # as P's CORN logits
            self.assertEqual(model.outputs[1].dtype, "float32")
            self.assertEqual(model.get_layer(em.LESION_OUTPUT).dtype_policy.name, "float32")
            self.assertEqual(model.get_layer(em.LESION_CONV_NAME).dtype_policy.name, "mixed_float16")
            self.assertTrue(tr.precision_is_consistent("mixed_float16", tr.model_precision_policies(model)))
            rgb, grades, targets = _batch(2, 2)
            grading, lesion = model.predict_on_batch({"rgb": rgb})
            self.assertEqual((grading.dtype, lesion.dtype), (np.float16, np.float32))
            self.assertTrue(np.isfinite(grading).all() and np.isfinite(lesion).all())
            et.compile_e1_model(model, CW, 1e-4, 0.05)
            out = model.train_on_batch({"rgb": rgb}, (grades, targets), return_dict=True)
            self.assertTrue(np.isfinite(float(out["loss"])) and np.isfinite(float(out["lesion_logits_loss"])))
            self.assertTrue(all(np.isfinite(np.asarray(w, np.float32)).all() for w in model.get_weights()))
        finally:
            keras.mixed_precision.set_global_policy(previous)


class LogAliasTests(unittest.TestCase):
    def test_19_actual_keras_keys_and_aliases(self):
        model = et.compile_e1_model(em.build_e1_model(42, PRIOR, REF, image_size=SMALL), CW, 1e-4, 0.05)
        rgb, grades, targets = _batch(4, 1)
        seen = {}

        class Later(keras.callbacks.Callback):
            def on_epoch_end(self, epoch, logs=None):
                seen["keys"] = sorted(logs)
                seen["val_QWK"] = logs.get("val_QWK")

        callbacks = et.with_aliases([Later()])
        self.assertEqual(type(callbacks[0]).__name__, "LogAliases")
        history = model.fit({"rgb": rgb}, (grades, targets), validation_data=({"rgb": rgb}, (grades, targets)),
                            batch_size=2, epochs=1, verbose=0, callbacks=callbacks)
        raw = callbacks[0].observed_keys
        print("\nKeras", keras.__version__, "raw log keys:", raw)
        # The names Keras actually uses for the two-output model (observed, not assumed).
        self.assertEqual(raw, sorted(["corn_QWK", "corn_corn_loss_unweighted", "corn_loss", "lesion_logits_loss", "loss",
                                      "val_corn_QWK", "val_corn_corn_loss_unweighted", "val_corn_loss",
                                      "val_lesion_logits_loss", "val_loss"]))
        for source, alias in et.ALIASES.items():
            self.assertIn(source, raw)
            self.assertNotIn(alias, raw)                                   # no alias existed before the callback
            self.assertIn(alias, seen["keys"])                             # the later callback saw it
            self.assertEqual(history.history[alias], history.history[source])
        self.assertIsNotNone(seen["val_QWK"])
        self.assertEqual(sorted(callbacks[0].created), sorted(et.ALIASES.values()))

    def test_alias_only_if_the_source_exists_and_strict_mode_refuses_missing_metrics(self):
        loose = et.make_log_aliases(strict=False)
        logs = {"corn_QWK": 0.5, "loss": 1.0}
        loose.on_epoch_end(0, logs)
        self.assertEqual(logs, {"corn_QWK": 0.5, "loss": 1.0, "QWK": 0.5})
        with self.assertRaises(RuntimeError):
            et.make_log_aliases().on_epoch_end(0, {"loss": 1.0, "val_loss": 1.0, "QWK": 0.4})
        with self.assertRaises(RuntimeError):
            et.make_log_aliases(strict=False).on_epoch_end(0, {"corn_QWK": 0.5, "QWK": 0.4})
        with self.assertRaises(RuntimeError):
            et.with_aliases(et.with_aliases([]))


class DiagnosticTests(unittest.TestCase):
    def test_logit_difference_report_finds_the_worst_image_and_counts_exceedances(self):
        rng = np.random.default_rng(0)
        ids = [f"{n:012x}" for n in range(50)]
        stored = rng.normal(0, 4, (50, 4))
        recomputed = stored + rng.normal(0, 0.002, (50, 4))
        recomputed[17, 2] = stored[17, 2] + 0.07                       # one isolated exceedance
        recomputed[30, 0] = stored[30, 0] - 0.2
        summary, frame = eg.logit_difference_report(ids, stored, recomputed)
        self.assertEqual(summary["worst_image"]["image_id"], ids[30])
        self.assertEqual((summary["worst_image"]["index"], summary["worst_image"]["threshold_index"]), (30, 0))
        self.assertAlmostEqual(summary["logit_abs_diff"]["max"], 0.2, places=9)
        self.assertEqual((summary["logit_abs_diff"]["n_above_0.05"], summary["logit_abs_diff"]["n_above_0.1"],
                          summary["logit_abs_diff"]["n_above_0.5"]), (2, 1, 0))
        self.assertEqual(summary["shift"]["images_with_max_diff_above_0.05"], 2)
        self.assertEqual(summary["saturation"]["elements_above_0.05"], 2)
        self.assertEqual(sum(b["n"] for b in summary["saturation"]["by_abs_reference_logit"]), 200)
        self.assertEqual(len(frame), 50)
        np.testing.assert_allclose(frame["max_abs_logit_diff"].to_numpy(), np.abs(recomputed - stored).max(axis=1))
        np.testing.assert_allclose(summary["worst_image"]["reference_cumulative_probabilities"], eg.cumulative(stored)[30])
        self.assertEqual(summary["decoded_grades_differ"], int((eg.decode(stored) != eg.decode(recomputed)).sum()))
        same, _ = eg.logit_difference_report(ids, stored, stored)
        self.assertEqual((same["logit_abs_diff"]["max"], same["decoded_grades_differ"]), (0.0, 0))


class DataAndGateTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tmp = tempfile.TemporaryDirectory()
        b = fx.build(os.path.join(cls.tmp.name, "data"))
        cls.bundle = ad.Arch1Bundle(expected_bundle_id=b["bundle"]["bundle_id"], expected_split_sha256=v2cfg.SPLIT_SHA256,
                                    expected_stage4_sha256=fx.MODEL_SHA, roots=b["roots"], expected_population=5)
        cls.grade_of = dict(fx.TRAIN + fx.VAL)
        cls.reference, cls.reference_arrays = _reference(512)

    @classmethod
    def tearDownClass(cls):
        cls.tmp.cleanup()

    def test_batches_carry_rgb_grades_and_aligned_targets_and_no_vessel_is_read(self):
        import shutil
        bundle = self.bundle
        entries = [(i, self.grade_of[i]) for i in bundle.train_ids]
        hidden = bundle.dirs["stage3_cache_v2"] + "_hidden"
        shutil.move(bundle.dirs["stage3_cache_v2"], hidden)               # E1 must work without the vessel files
        try:
            x, (grades, targets) = ed.make_epoch_sequence(bundle, entries, 0, 42, 2, augment=False)[0]
            self.assertEqual(set(x), {"rgb"})
            self.assertEqual((x["rgb"].shape, grades.shape, targets.shape), ((2, 512, 512, 3), (2,), (2, 16, 16, 4)))
            self.assertEqual((grades.dtype, targets.dtype), (np.int32, np.float32))
            for row, (image_id, grade) in enumerate(entries[:2]):
                full = bundle.load_sample if False else None
                maps = cache.from_uint8(ed.load_pathology(bundle, image_id))
                np.testing.assert_array_equal(targets[row], ed.pool_targets_bruteforce(maps, bundle.channels))
                self.assertEqual(int(grades[row]), grade)
            ax, (_, at_targets) = ed.make_epoch_sequence(bundle, entries, 3, 42, 2, augment=True)[0]
            self.assertTrue(at_targets.min() >= 0 and at_targets.max() <= 1)
            self.assertTrue(ax["rgb"].min() >= 0 and ax["rgb"].max() <= 1)
        finally:
            shutil.move(hidden, bundle.dirs["stage3_cache_v2"])
        whole = bundle.load_sample(entries[0][0])                         # same RGB and maps as Architecture 1 read
        mine = ed.load_inputs(bundle, entries[0][0])
        np.testing.assert_array_equal(mine["rgb"], whole["rgb"])
        np.testing.assert_array_equal(mine["pathology"], whole["pathology"])

    def test_target_gate_statistics_and_hard_failures(self):
        result = eg.targets(self.bundle, official=True, log=lambda *a: None)
        self.assertTrue(result["PASS"], result["failures"])
        stats = result["statistics"]
        self.assertEqual((stats["images"], stats["shape"], stats["non_finite"]), (3, [16, 16, 4], 0))
        self.assertEqual(stats["bruteforce_checked_images"], 3)
        self.assertTrue(0.0 <= stats["min"] <= stats["max"] <= 1.0)
        manual = np.stack([ed.pool_targets(cache.from_uint8(ed.load_pathology(self.bundle, i)), self.bundle.channels)
                           for i in self.bundle.train_ids])
        np.testing.assert_allclose(result["lesion_prior"], manual.mean(axis=(0, 1, 2)), rtol=1e-6)
        np.testing.assert_allclose(list(stats["fraction_cells_above_0.9_per_class"].values()),
                                   (manual > 0.9).mean(axis=(0, 1, 2)), rtol=1e-9)
        np.testing.assert_allclose(list(stats["positive_cell_rate_per_class"].values()),
                                   (manual >= 0.5).mean(axis=(0, 1, 2)), rtol=1e-9)
        self.assertFalse(eg.targets(self.bundle, image_ids=self.bundle.train_ids[:2], log=lambda *a: None)["PASS"])
        self.assertTrue(eg.targets(self.bundle, image_ids=self.bundle.train_ids[:2], official=False,
                                   log=lambda *a: None)["PASS"])
        original = ed.pool_targets                                        # a wrong pooling must fail the gate
        ed.pool_targets = lambda p, c, grid=ed.GRID: np.zeros((grid, grid, 4), np.float32)
        try:
            broken = eg.targets(self.bundle, log=lambda *a: None)
        finally:
            ed.pool_targets = original
        self.assertFalse(broken["PASS"])
        self.assertTrue(any("brute-force" in f for f in broken["failures"]))
        self.assertTrue(any("collapse" in f for f in broken["failures"]))

    def test_p_parity_gate_on_synthetic_checkpoints(self):
        from training import checkpointing as ckpt
        weights_file = os.path.join(self.tmp.name, "reference.weights.h5")
        self.reference.save_weights(weights_file)
        paths, shas = {}, {}
        rng = np.random.default_rng(1)
        for seed in (42, 123):
            p = pl.build_pl_model("P", seed, self.reference_arrays)
            for v in p.trainable_variables:
                v.assign(v.numpy() + rng.normal(0, 0.01, v.shape).astype("float32"))
            paths[seed] = os.path.join(self.tmp.name, f"p_{seed}.weights.h5")
            ckpt.save_model_weights_only(p, paths[seed])
            shas[seed] = cache.sha256_file(paths[seed])
        result = eg.p_parity(self.bundle, paths, weights_file, PRIOR, policies=("float32",), expected_sha256=shas,
                             official=False, log=lambda *a: None)
        self.assertTrue(result["PASS"], result["failures"])
        for seed in (42, 123):
            row = result["tiers"]["float32"][seed]
            self.assertLessEqual(row["e1_vs_live_p"]["logit_max_abs"], 1e-4)
            self.assertTrue(row["e1_vs_live_p"]["grades_equal"])
            self.assertEqual(row["lesion_output_shape"], [16, 16, 4])
            self.assertLessEqual(result["initialisation"][seed]["logit_max_abs"], eg.INIT_TOL)
        self.assertFalse(result["official"])
        bad = eg.p_parity(self.bundle, paths, weights_file, PRIOR, policies=("float32",),
                          expected_sha256={42: "0" * 64, 123: shas[123]}, official=False, log=lambda *a: None)
        self.assertFalse(bad["PASS"])                                      # an unpinned checkpoint is refused
        official = eg.p_parity(self.bundle, {42: paths[42]}, weights_file, PRIOR, policies=("float32",),
                               expected_sha256=shas, official=True, log=lambda *a: None)
        self.assertFalse(official["PASS"])                                 # official = GPU runtime + both tiers

    def test_one_epoch_run_checkpoints_history_aliases_and_resume(self):
        import multiseed_runs as msr
        work = os.path.join(self.tmp.name, "alias_gate")
        gate = eg.log_alias(work, self.bundle, self.grade_of, self.reference, CW, mixed_precision=False,
                            official=False, log=lambda *a: None)
        self.assertTrue(gate["PASS"], gate["failures"])
        self.assertEqual(gate["callback_order"][0], "LogAliases")
        for key in et.REQUIRED_TRAIN_KEYS + et.REQUIRED_VAL_KEYS:
            self.assertIn(key, gate["raw_keras_log_keys"])
        for alias in et.ALIASES.values():
            self.assertIn(alias, gate["keys_after_alias"])
            self.assertIsNotNone(gate["stored_history_row"][alias])
        self.assertEqual(gate["checkpoint_monitor"], "val_QWK")
        run_dir = os.path.join(work, "alias_gate_run")
        import json
        with open(os.path.join(run_dir, "config.json")) as fh:
            cfg = json.load(fh)
        self.assertEqual((cfg["experiment"], cfg["lesion_loss_weight"], cfg["vessel_supervision"]), ("E1MultiTask", 1.0, False))
        self.assertEqual(cfg["lesion_target"]["source_channels"], ["MA:max", "HE:max", "EX:max", "SE:max"])
        self.assertEqual((cfg["batch_size"], cfg["learning_rate"], cfg["weight_decay"], cfg["ema"]), (2, 1e-4, 0.05, "none"))
        self.assertIsNotNone(msr.read_stop_decision(run_dir))             # epoch cap reached
        results = et.evaluate_run(run_dir, self.bundle, 42, (0.1, 0.1, 0.1, 0.1), self.reference, CW,
                                  grade_of=self.grade_of, mixed_precision=False)
        self.assertEqual(set(results), {"best", "last"})
        self.assertIn("qwk", results["best"]["metrics"])
        self.assertTrue(np.isfinite(results["best"]["lesion_head_validation"]["lesion_loss"]))
        self.assertTrue(os.path.exists(os.path.join(run_dir, "metrics", "per_sample_best.csv")))
        payload = et.write_result(run_dir, results)
        self.assertEqual(et.seed_state(run_dir), "complete")
        self.assertEqual(len(payload["history"]), 1)
        with self.assertRaises(RuntimeError):                              # another prior = another configuration
            et.train_seed(run_dir, self.bundle, 42, (0.2, 0.1, 0.1, 0.1), self.reference, CW, repo_dir=os.getcwd(),
                          staging_dir=os.path.join(work, "s"), max_epochs=1, grade_of=self.grade_of,
                          mixed_precision=False, log=lambda *a: None)
        with self.assertRaises(RuntimeError):                              # the sequence needs the three gates
            et.run_sequence(os.path.join(work, "exp"), self.bundle, (0.1,) * 4, self.reference, CW,
                            repo_dir=os.getcwd(), staging_root=os.path.join(work, "st"), gates={"p_parity": {"PASS": True}})


if __name__ == "__main__":
    unittest.main()
