"""CPU tests for eyepacs_adaptation_train. Random ConvNeXt weights and synthetic 64-pixel frames only: no EyePACS
image is decoded, no APTOS file is read, and nothing here is an adaptation result. The short run exists to prove
the checkpoint / selection / pinning / held-out-evaluation mechanics."""
import inspect
import json
import os
import tempfile
import unittest

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "-1")

import keras
import numpy as np

import arch1_train as at
import eyepacs_adaptation_data as ea
import eyepacs_adaptation_train as et
import improved_training_data as itd
import local_feature_extraction_dataset as lfed
import pl_convnext as pl

REPO = os.path.dirname(os.path.abspath(et.__file__))
SMALL = 64


def reference_arrays(size=SMALL):
    keras.utils.set_random_seed(7)
    from keras.applications import ConvNeXtTiny
    ref = ConvNeXtTiny(include_top=False, weights=None, include_preprocessing=True, pooling="avg",
                       input_shape=(size, size, 3), name=pl.BACKBONE_NAME)
    for v in ref.weights:
        if "prestem" not in v.path:
            v.assign(np.random.default_rng(len(v.path)).normal(0, 0.05, v.shape).astype("float32")
                     + (1.0 if v.path.endswith("gamma") else 0.0))
    return [np.asarray(v.numpy()) for v in ref.weights if "prestem" not in v.path]


REF_ARRAYS = reference_arrays()
CW = list(et.APPROVED_CLASS_WEIGHTS)


class FakeFrames:
    """Stands in for eyepacs_adaptation_data.FrameCache with small frames."""
    fingerprint = "e" * 64

    def __init__(self, splits=("train", "val"), n=(20, 10), size=SMALL, manifest_sha256="f" * 64, first_patient=1):
        self.manifest_sha256 = manifest_sha256
        self._entries, patient = {}, first_patient
        for split, count in zip(splits, n):
            self._entries[split] = []
            for _ in range(count // 2):
                for k, eye in enumerate(("left", "right")):
                    self._entries[split].append((f"{patient}_{eye}.jpeg", (patient + k) % 5))
                patient += 1
        self.report = {"frames": {s: len(v) for s, v in self._entries.items()}}
        self.size = size
        self.read = []

    def entries(self, split):
        return list(self._entries[split])

    def frame(self, image):
        ea.parse_name(image)
        self.read.append(image)
        patient = int(image.split("_")[0])
        return np.random.default_rng(patient + image.endswith("left.jpeg")).random((self.size, self.size, 3)).astype(np.float32)


def small_adaptation(root, frames, tmp):
    """A two-epoch synthetic adaptation run through the real loop (used here and by the APTOS-arm tests)."""
    def train(run_dir, frames, reference_arrays, weights, **kw):
        return et.train(run_dir, frames, reference_arrays, weights, max_epochs=2, mixed_precision=False,
                        image_size=SMALL, workers=1, **kw)

    def evaluate(run_dir, frames, reference_arrays, weights):
        return et.evaluate_run(run_dir, frames, reference_arrays, weights, mixed_precision=False, image_size=SMALL)
    result = et.run_adaptation(root, frames, REF_ARRAYS, CW, repo_dir=REPO, staging_root=os.path.join(tmp, "staging"),
                               log=lambda *a: None, train_fn=train, evaluate_fn=evaluate)
    return result, train, evaluate


class ProtocolTests(unittest.TestCase):
    def test_protocol_is_p_with_batch_16_only(self):
        self.assertEqual({k: v for k, v in et.PROTOCOL.items() if k != "batch_size"},
                         {k: v for k, v in at.P_PROTOCOL.items() if k != "batch_size"})
        self.assertEqual((et.PROTOCOL["batch_size"], at.P_PROTOCOL["batch_size"]), (16, 2))
        self.assertEqual((et.PROTOCOL["learning_rate"], et.PROTOCOL["weight_decay"], et.PROTOCOL["max_epochs"],
                          et.PROTOCOL["early_stopping_patience"], et.PROTOCOL["monitor"], et.PROTOCOL["mode"]),
                         (1e-4, 0.05, 50, 12, "val_QWK", "max"))

    def test_exactly_one_adaptation_run(self):
        self.assertEqual((et.ADAPTATION_RUNS, et.ADAPTATION_SEED), (1, 42))
        self.assertFalse(hasattr(et, "run_sequence") or hasattr(et, "SEEDS"))
        for function in (et.run_adaptation, et.train, et.run_dir_for, et.run_mapping):
            self.assertFalse({"seed", "seeds"} & set(inspect.signature(function).parameters))   # no seed can be passed
        self.assertTrue(et.run_dir_for("x", "a" * 64).endswith("p_eyepacs_aaaaaaaaaaaa_seed42"))

    def test_class_weights_come_from_the_pinned_manifest_training_part(self):
        rows = ea.read_split(os.path.join(REPO, "dataset_splits"), ea.MANIFEST_SHA256)
        counts, weights = et.class_weights(rows)
        self.assertEqual(counts, [23209, 2209, 4768, 782, 642])
        np.testing.assert_allclose(weights, et.APPROVED_CLASS_WEIGHTS, atol=5e-5)
        changed = [dict(r, grade=0) if r["split"] == "train" and r["grade"] == 4 and int(r["patient"]) % 2 else r for r in rows]
        with self.assertRaises(RuntimeError):
            et.class_weights(changed)

    def test_no_aptos_idrid_ddr_or_quality_dependency_and_no_test_set_in_training(self):
        source = inspect.getsource(et)
        for forbidden in ("verify_split", "Arch1Bundle", "arch1_data", "stage34_cache_v2", "DOWNSTREAM_SPLIT", "Label_EyeQ",
                          "image_quality", "idrid", "ddr_probe", "train_images", "e1_model", "e2_control", "lesion_loss"):
            self.assertNotIn(forbidden, source)
        for function in (et.train, et.run_adaptation, et.training_loop, et.pin_checkpoint, et.selected_epoch):
            self.assertFalse([p for p in inspect.signature(function).parameters if "test" in p])
            self.assertNotIn("heldout", inspect.getsource(function).replace("evaluate_heldout_test", ""))
        self.assertNotIn('entries("test")', inspect.getsource(et.train) + inspect.getsource(et.run_adaptation))
        with tempfile.TemporaryDirectory() as tmp:                         # a cache that holds test frames is refused
            with self.assertRaises(RuntimeError):
                et.train(os.path.join(tmp, "r"), FakeFrames(splits=("train", "val", "test"), n=(4, 4, 4)), REF_ARRAYS, CW,
                         repo_dir=REPO, staging_dir=os.path.join(tmp, "s"))


class ModelAndInputTests(unittest.TestCase):
    def test_the_model_is_p_compiled_like_p(self):
        model = et.build_compiled_model(42, REF_ARRAYS, CW, mixed_precision=False, image_size=SMALL)
        p = pl.build_pl_model("P", 42, REF_ARRAYS, image_size=SMALL)
        self.assertEqual(model.count_params(), p.count_params())
        self.assertEqual(model.count_params(), pl.EXPECTED_BACKBONE_PARAMETERS["P"] + pl.EXPECTED_HEAD_PARAMETERS)
        for a, b in zip(model.get_weights(), p.get_weights()):
            np.testing.assert_array_equal(a, b)                            # same seed -> the same initial model
        optimizer = pl.inner_optimizer(model)
        self.assertEqual(type(optimizer).__name__, "AdamW")
        self.assertAlmostEqual(float(optimizer.learning_rate), 1e-4)
        self.assertAlmostEqual(float(optimizer.weight_decay), 0.05)
        frames = FakeFrames()
        x = et.p_inputs(np.stack([frames.frame(i) for i, _ in frames.entries("val")[:3]]))
        self.assertEqual(model.predict_on_batch(x).shape, (3, 4))

    def test_inputs_follow_ps_three_input_contract(self):
        rgb = np.random.default_rng(0).random((2, SMALL, SMALL, 3)).astype(np.float32)
        x = et.p_inputs(rgb)
        self.assertEqual(set(x), {"stage5_input", "stage6_input", "reliability"})
        np.testing.assert_array_equal(x["stage5_input"][..., :3], rgb)
        self.assertFalse(x["stage5_input"][..., 3:].any() or x["stage6_input"].any() or x["reliability"].any())
        self.assertEqual(x["stage5_input"].shape, (2, SMALL, SMALL, 8))

    def test_epoch_sequence_is_ps_order_and_ps_augmentation(self):
        frames = FakeFrames()
        entries = frames.entries("train")
        seq = et.make_epoch_sequence(frames, entries, 3, 42, 16, augment=True)
        self.assertEqual(len(seq), 2)
        order = itd.epoch_training_order(entries, 42, 3)
        x0, y0 = seq[0]
        self.assertEqual(y0.dtype, np.int32)
        np.testing.assert_array_equal(y0, [g for _, g in order[:16]])
        image = order[5][0]
        rng = itd.per_image_augmentation_rng(42, 3, image)
        expected = lfed._augment_intensity_rgb(lfed._augment_spatial(frames.frame(image), rng), rng)
        np.testing.assert_array_equal(x0["stage5_input"][5, ..., :3], expected)
        x0b, _ = et.make_epoch_sequence(frames, entries, 3, 42, 16, augment=True, workers=3)[0]
        np.testing.assert_array_equal(x0["stage5_input"], x0b["stage5_input"])   # independent of workers
        seen = [g for i in range(len(seq)) for g in seq[i][1]]
        self.assertEqual(sorted(seen), sorted(g for _, g in entries))             # every image once per epoch
        val = et.make_epoch_sequence(frames, frames.entries("val"), 0, 42, 16, augment=False)
        xv, yv = val[0]
        np.testing.assert_array_equal(yv, [g for _, g in frames.entries("val")])   # given order, unaugmented
        np.testing.assert_array_equal(xv["stage5_input"][2, ..., :3], frames.frame(frames.entries("val")[2][0]))

    def test_selected_epoch_is_the_first_maximum(self):
        history = [{"epoch": 1, "val_QWK": 0.2}, {"epoch": 2, "val_QWK": 0.5}, {"epoch": 3, "val_QWK": 0.5}, {"epoch": 4, "val_QWK": 0.4}]
        self.assertEqual(et.selected_epoch(history), (2, 0.5))
        with self.assertRaises(RuntimeError):
            et.selected_epoch([{"epoch": 1, "val_QWK": None}])


class RunTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tmp = tempfile.TemporaryDirectory(ignore_cleanup_errors=True)
        cls.frames = FakeFrames()
        cls.root = os.path.join(cls.tmp.name, "exp")
        cls.result, train, evaluate = small_adaptation(cls.root, cls.frames, cls.tmp.name)
        cls.train, cls.evaluate = staticmethod(train), staticmethod(evaluate)
        cls.run_dir = et.run_dir_for(cls.root, cls.frames.manifest_sha256)

    @classmethod
    def tearDownClass(cls):
        cls.tmp.cleanup()

    def test_run_writes_config_history_metadata_and_a_pinned_checkpoint(self):
        import multiseed_runs as msr
        with open(os.path.join(self.run_dir, "config.json")) as fh:
            cfg = json.load(fh)
        self.assertEqual((cfg["experiment"], cfg["seed"], cfg["adaptation_runs"], cfg["batch_size"], cfg["learning_rate"],
                          cfg["weight_decay"], cfg["quality_filter"]), ("EyePACSAdaptation", 42, 1, 16, 1e-4, 0.05, "none"))
        self.assertEqual((cfg["split_manifest_sha256"], cfg["checkpoint_selection"]), (self.frames.manifest_sha256, et.SELECTION))
        history = msr.read_history(self.run_dir)
        self.assertEqual([h["epoch"] for h in history], [1, 2])
        self.assertEqual(msr.read_stop_decision(self.run_dir)["stop_reason"], "epoch_cap")
        frozen = self.result["frozen"]
        completed, top = et.selected_epoch(history)                         # selection = validation QWK, first maximum
        self.assertEqual((frozen["best_epoch_index"], frozen["val_qwk"]), (completed - 1, top))
        self.assertEqual(self.result["best"]["checkpoint"]["weights_sha256"], frozen["sha256"])
        self.assertEqual(self.result["best"]["set"], "EyePACS adaptation validation (selection set)")
        for key in ("git_commit", "python", "tensorflow", "keras", "numpy", "gpus", "platform", "utc"):
            self.assertIn(key, frozen["pinned"])                            # commit, library versions, hardware
        self.assertEqual(frozen["identity"]["split_manifest_sha256"], self.frames.manifest_sha256)
        self.assertEqual((frozen["training_configuration"]["batch_size"], frozen["training_configuration"]["max_epochs"],
                          frozen["training_configuration"]["early_stopping_patience"]), (16, 50, 12))
        with open(os.path.join(self.run_dir, "run_metadata.json")) as fh:
            self.assertEqual(len(json.load(fh)["invocations"]), 1)
        self.assertEqual(et.state(self.run_dir), "complete")
        self.assertTrue(set(self.frames.read) <= {i for s in ("train", "val") for i, _ in self.frames.entries(s)})

    def test_pinned_checkpoint_is_immutable_and_hands_over_the_encoder_only(self):
        path, record = et.read_frozen(self.run_dir)
        self.assertEqual(at._sha256(path), record["sha256"])
        self.assertEqual(et.pin_checkpoint(self.run_dir), record)           # same file: accepted, unchanged
        model, again = et.load_frozen(self.run_dir, REF_ARRAYS, mixed_precision=False, image_size=SMALL)
        self.assertFalse(model.trainable_variables)
        self.assertEqual(again, record)
        arrays = et.adapted_backbone_arrays(model)
        self.assertFalse(all(np.array_equal(a, b) for a, b in zip(arrays, REF_ARRAYS)))   # trained, not ImageNet's
        for seed in (42, 123):
            p_ep = pl.build_pl_model("P", seed, arrays, image_size=SMALL)
            for a, v in zip(arrays, p_ep.get_layer(pl.BACKBONE_NAME).weights):
                np.testing.assert_array_equal(a, v.numpy())                 # the adapted encoder, exactly
            np.testing.assert_array_equal(p_ep.get_layer("corn").get_layer("corn_logits").get_weights()[0],
                                          pl.corn_head_initial_weights(seed)[0])
        self.assertFalse(np.array_equal(model.get_layer("corn").get_layer("corn_logits").get_weights()[0],
                                        pl.corn_head_initial_weights(42)[0]))   # the EyePACS-trained head is not carried over
        with open(path, "rb") as fh:
            original = fh.read()
        try:
            with open(path, "ab") as fh:
                fh.write(b"x")
            with self.assertRaises(RuntimeError):
                et.read_frozen(self.run_dir)                                # altered weights are refused
        finally:
            with open(path, "wb") as fh:
                fh.write(original)
        et.read_frozen(self.run_dir)

    def test_heldout_test_is_scored_once_from_the_pinned_checkpoint_and_selects_nothing(self):
        frozen_before = dict(et.read_frozen(self.run_dir)[1])
        with self.assertRaises(RuntimeError):                               # not the pinned held-out manifest
            et.evaluate_heldout_test(self.run_dir, FakeFrames(splits=("test",), n=(6,), first_patient=5000), REF_ARRAYS,
                                     mixed_precision=False, image_size=SMALL, log=lambda *a: None)
        with self.assertRaises(RuntimeError):                               # the adaptation cache is not a test cache
            et.evaluate_heldout_test(self.run_dir, self.frames, REF_ARRAYS, mixed_precision=False, image_size=SMALL)
        test = FakeFrames(splits=("test",), n=(6,), manifest_sha256=ea.TEST_MANIFEST_SHA256, first_patient=5000)
        expected = ea.EXPECTED_TEST_IMAGES
        try:
            ea.EXPECTED_TEST_IMAGES = 6                                     # synthetic stand-in for the 16,249
            result = et.evaluate_heldout_test(self.run_dir, test, REF_ARRAYS, mixed_precision=False, image_size=SMALL,
                                              log=lambda *a: None)
        finally:
            ea.EXPECTED_TEST_IMAGES = expected
        self.assertEqual((result["images"], result["checkpoint_sha256"], result["role"]), (6, frozen_before["sha256"], et.HELDOUT_ROLE))
        self.assertIn("qwk", result["metrics"])
        self.assertTrue(os.path.exists(os.path.join(self.run_dir, et.HELDOUT_DIR, "per_sample.csv")))
        reads = len(test.read)
        again = et.evaluate_heldout_test(self.run_dir, test, REF_ARRAYS, mixed_precision=False, image_size=SMALL, log=lambda *a: None)
        self.assertEqual((again, len(test.read)), (json.loads(json.dumps(result, default=float)), reads))   # never recomputed
        self.assertEqual(et.read_frozen(self.run_dir)[1], frozen_before)    # the pinned checkpoint is untouched
        with open(os.path.join(self.run_dir, "result.json")) as fh:
            self.assertNotIn("heldout", fh.read())                          # and the run's own record does not read it
        with tempfile.TemporaryDirectory() as tmp:                          # an unpinned run cannot be scored
            with self.assertRaises(RuntimeError):
                et.evaluate_heldout_test(tmp, test, REF_ARRAYS)

    def test_one_configuration_per_directory_and_a_finished_run_is_kept(self):
        with self.assertRaises(RuntimeError):                               # other class weights = another configuration
            self.train(self.run_dir, self.frames, REF_ARRAYS, [1.0] * 5, repo_dir=REPO,
                       staging_dir=os.path.join(self.tmp.name, "s2"), log=lambda *a: None)
        calls = []
        kept = et.run_adaptation(self.root, self.frames, REF_ARRAYS, CW, repo_dir=REPO, staging_root=os.path.join(self.tmp.name, "st"),
                                 log=lambda *a: None, train_fn=lambda *a, **k: calls.append(a), evaluate_fn=self.evaluate)
        self.assertEqual(calls, [])
        self.assertEqual(kept["frozen"], self.result["frozen"])
        with self.assertRaises(RuntimeError):                               # an unfinished run cannot be pinned
            et.pin_checkpoint(os.path.join(self.tmp.name, "nothing"))


if __name__ == "__main__":
    unittest.main()
