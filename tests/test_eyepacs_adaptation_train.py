"""CPU tests for eyepacs_adaptation_train. Random ConvNeXt weights and synthetic 64-pixel frames only: no EyePACS
image is decoded, no APTOS file is read, and nothing here is an adaptation result. The short run exists to prove
the checkpoint / selection / freezing mechanics."""
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


def _reference_arrays(size):
    keras.utils.set_random_seed(7)
    from keras.applications import ConvNeXtTiny
    ref = ConvNeXtTiny(include_top=False, weights=None, include_preprocessing=True, pooling="avg",
                       input_shape=(size, size, 3), name=pl.BACKBONE_NAME)
    for v in ref.weights:
        if "prestem" not in v.path:
            v.assign(np.random.default_rng(len(v.path)).normal(0, 0.05, v.shape).astype("float32")
                     + (1.0 if v.path.endswith("gamma") else 0.0))
    return [np.asarray(v.numpy()) for v in ref.weights if "prestem" not in v.path]


REF_ARRAYS = _reference_arrays(SMALL)
CW = list(et.APPROVED_CLASS_WEIGHTS)


class FakeFrames:
    """Stands in for eyepacs_adaptation_data.FrameCache with small frames."""
    manifest_sha256 = "f" * 64
    fingerprint = "e" * 64

    def __init__(self, n_train=20, n_val=10, size=SMALL):
        self._entries = {"train": [(f"{p}_{eye}.jpeg", (p + k) % 5) for p in range(1, n_train // 2 + 1) for k, eye in enumerate(("left", "right"))],
                         "val": [(f"{p}_{eye}.jpeg", (p + k) % 5) for p in range(900, 900 + n_val // 2) for k, eye in enumerate(("left", "right"))]}
        self.size = size
        self.read = []

    def entries(self, split):
        return list(self._entries[split])

    def frame(self, image):
        ea.parse_name(image)
        self.read.append(image)
        patient = int(image.split("_")[0])
        return np.random.default_rng(patient + image.endswith("left.jpeg")).random((self.size, self.size, 3)).astype(np.float32)


class ProtocolTests(unittest.TestCase):
    def test_protocol_is_p_with_batch_16_only(self):
        self.assertEqual({k: v for k, v in et.PROTOCOL.items() if k != "batch_size"},
                         {k: v for k, v in at.P_PROTOCOL.items() if k != "batch_size"})
        self.assertEqual((et.PROTOCOL["batch_size"], at.P_PROTOCOL["batch_size"]), (16, 2))
        self.assertEqual((et.PROTOCOL["learning_rate"], et.PROTOCOL["weight_decay"], et.PROTOCOL["max_epochs"],
                          et.PROTOCOL["early_stopping_patience"], et.PROTOCOL["monitor"]), (1e-4, 0.05, 50, 12, "val_QWK"))
        self.assertEqual(et.SEEDS, (42, 123, 2026))

    def test_class_weights_come_from_the_pinned_manifest_training_part(self):
        rows = ea.read_split(os.path.join(REPO, "dataset_splits"), ea.MANIFEST_SHA256)
        counts, weights = et.class_weights(rows)
        self.assertEqual(counts, [23209, 2209, 4768, 782, 642])
        np.testing.assert_allclose(weights, et.APPROVED_CLASS_WEIGHTS, atol=5e-5)
        changed = [dict(r, grade=0) if r["split"] == "train" and r["grade"] == 4 and int(r["patient"]) % 2 else r for r in rows]
        with self.assertRaises(RuntimeError):
            et.class_weights(changed)

    def test_no_aptos_idrid_or_quality_dependency(self):
        source = inspect.getsource(et)
        for forbidden in ("verify_split", "Arch1Bundle", "arch1_data", "stage34_cache_v2", "DOWNSTREAM_SPLIT", "Label_EyeQ",
                          "image_quality", "idrid", "ddr_probe", "train_images"):
            self.assertNotIn(forbidden, source)
        self.assertNotIn("test", {p for f in (et.train_seed, et.run_sequence, et.freeze_checkpoint)
                                  for p in inspect.signature(f).parameters})


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
        other = et.make_epoch_sequence(frames, entries, 4, 42, 16, augment=True)[0][1]
        self.assertFalse(np.array_equal(other, y0) and itd.epoch_training_order(entries, 42, 4) == order)
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
        cls.kw = dict(repo_dir=REPO, staging_root=os.path.join(cls.tmp.name, "staging"), log=lambda *a: None)

        def train(run_dir, frames, seed, reference_arrays, weights, **kw):
            return et.train_seed(run_dir, frames, seed, reference_arrays, weights, max_epochs=2, mixed_precision=False,
                                 image_size=SMALL, workers=1, **kw)

        def evaluate(run_dir, frames, seed, reference_arrays, weights):
            return et.evaluate_run(run_dir, frames, seed, reference_arrays, weights, mixed_precision=False, image_size=SMALL)
        cls.train, cls.evaluate = staticmethod(train), staticmethod(evaluate)
        cls.results = et.run_sequence(cls.root, cls.frames, REF_ARRAYS, CW, (42,), train_fn=train, evaluate_fn=evaluate, **cls.kw)
        cls.run_dir = et.run_dir_for(cls.root, cls.frames.manifest_sha256, 42)

    @classmethod
    def tearDownClass(cls):
        cls.tmp.cleanup()

    def test_run_writes_config_history_best_and_a_frozen_checkpoint(self):
        import multiseed_runs as msr
        with open(os.path.join(self.run_dir, "config.json")) as fh:
            cfg = json.load(fh)
        self.assertEqual((cfg["experiment"], cfg["batch_size"], cfg["learning_rate"], cfg["weight_decay"], cfg["quality_filter"]),
                         ("EyePACSAdaptation", 16, 1e-4, 0.05, "none"))
        self.assertEqual(cfg["split_manifest_sha256"], self.frames.manifest_sha256)
        self.assertEqual(cfg["checkpoint_selection"], et.SELECTION)
        history = msr.read_history(self.run_dir)
        self.assertEqual([h["epoch"] for h in history], [1, 2])
        self.assertTrue(all(h["val_QWK"] is not None for h in history))
        self.assertEqual(msr.read_stop_decision(self.run_dir)["stop_reason"], "epoch_cap")
        result = self.results[42]
        frozen = result["frozen"]
        completed, top = et.selected_epoch(history)                         # selection = validation QWK, first maximum
        self.assertEqual((frozen["best_epoch_index"], frozen["val_qwk"]), (completed - 1, top))
        self.assertEqual(result["best"]["checkpoint"]["weights_sha256"], frozen["sha256"])
        self.assertEqual(result["best"]["set"], "EyePACS adaptation validation (selection set)")
        self.assertIn("qwk", result["best"]["metrics"])
        self.assertTrue(os.path.exists(os.path.join(self.run_dir, "metrics", "per_sample_best.csv")))
        self.assertEqual(et.seed_state(self.run_dir), "complete")
        self.assertTrue(set(self.frames.read) <= {i for s in ("train", "val") for i, _ in self.frames.entries(s)})

    def test_frozen_checkpoint_is_pinned_and_never_replaced(self):
        path, record = et.read_frozen(self.run_dir)
        self.assertEqual(at._sha256(path), record["sha256"])
        self.assertEqual(et.freeze_checkpoint(self.run_dir), record)        # same file: accepted, unchanged
        model, again = et.load_frozen(self.run_dir, REF_ARRAYS, mixed_precision=False, image_size=SMALL)
        self.assertFalse(model.trainable_variables)
        self.assertEqual(again, record)
        logits = et.predict_logits(model, self.frames, [i for i, _ in self.frames.entries("val")])
        self.assertEqual(logits.shape, (10, 4))
        arrays = et.adapted_backbone_arrays(model)                          # the hand-over to the APTOS code
        for seed in (42, 123):
            p_ep = pl.build_pl_model("P", seed, arrays, image_size=SMALL)
            for a, v in zip(arrays, p_ep.get_layer(pl.BACKBONE_NAME).weights):
                np.testing.assert_array_equal(a, v.numpy())                 # the adapted encoder, exactly
            kernel, bias = pl.corn_head_initial_weights(seed)
            np.testing.assert_array_equal(p_ep.get_layer("corn").get_layer("corn_logits").get_weights()[0], kernel)
        self.assertFalse(np.array_equal(model.get_layer("corn").get_layer("corn_logits").get_weights()[0],
                                        pl.corn_head_initial_weights(42)[0]))   # the trained head is not carried over
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

    def test_a_run_directory_holds_one_configuration_and_a_finished_seed_is_kept(self):
        with self.assertRaises(RuntimeError):                               # other class weights = another configuration
            self.train(self.run_dir, self.frames, 42, REF_ARRAYS, [1.0] * 5, repo_dir=REPO,
                       staging_dir=os.path.join(self.tmp.name, "s2"), log=lambda *a: None)
        calls = []
        kept = et.run_sequence(self.root, self.frames, REF_ARRAYS, CW, (42,), train_fn=lambda *a, **k: calls.append(a),
                               evaluate_fn=self.evaluate, **self.kw)
        self.assertEqual(calls, [])
        self.assertEqual(kept[42]["frozen"], self.results[42]["frozen"])
        with self.assertRaises(ValueError):
            et.run_dir_for(self.root, self.frames.manifest_sha256, 7)
        with self.assertRaises(RuntimeError):                               # an unfinished run cannot be frozen
            et.freeze_checkpoint(os.path.join(self.tmp.name, "nothing"))


if __name__ == "__main__":
    unittest.main()
