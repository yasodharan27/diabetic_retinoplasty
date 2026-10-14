"""CPU smoke tests for the model-dependent later-stage code: Grad-CAM on a real (randomly initialised) P-graph grader,
the external re-test evaluator on a fabricated drive with fake models, and the demonstration figure. No real
checkpoint, no external image and no result is involved."""
import json
import os
import tempfile
import unittest

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "-1")

import numpy as np
import pandas as pd

import arch1_train as at
import final_external_eval as fx
import idrid_grading_eval as ig
import joint_train as jt
from test_eyepacs_adaptation_train import REF_ARRAYS


class GradCamTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        import keras

        import pl_convnext as pl
        cls.tmp = tempfile.TemporaryDirectory(ignore_cleanup_errors=True)
        keras.mixed_precision.set_global_policy("float32")
        cls.p = pl.build_pl_model("P", 42, REF_ARRAYS)
        cls.weights = os.path.join(cls.tmp.name, "model.weights.h5")
        cls.p.save_weights(cls.weights)
        cls.rgb = np.random.default_rng(0).random((1, 512, 512, 3)).astype(np.float32)
        import eyepacs_adaptation_train as eat
        cls.p_logits = np.asarray(cls.p.predict_on_batch(eat.p_inputs(cls.rgb)), np.float64)

    @classmethod
    def tearDownClass(cls):
        cls.tmp.cleanup()

    def test_the_map_is_computed_on_the_graders_own_logits(self):
        import gradcam
        grad_model, _ = gradcam.load_explainer(42, self.weights, REF_ARRAYS)
        cam, logits = gradcam.grad_cam(grad_model, self.rgb)
        self.assertEqual(cam.shape, (1, 512, 512))
        self.assertTrue(0.0 <= float(cam.min()) and float(cam.max()) <= 1.0 + 1e-6)
        self.assertLess(float(np.abs(logits - self.p_logits).max()), 1e-3)          # the explained model is the grader
        cut_cam, _ = gradcam.grad_cam(grad_model, self.rgb, cut=1)
        self.assertEqual(cut_cam.shape, (1, 512, 512))
        mean, per_seed = gradcam.ensemble_cam([grad_model, grad_model], self.rgb)
        self.assertTrue(np.allclose(mean, cam, atol=1e-5))
        self.assertEqual(per_seed.shape, (2, 1, 4))


class FakeNet:
    def __init__(self, shift):
        self.shift = shift

    def predict_on_batch(self, x):
        m = x["stage5_input"][..., :3].reshape(len(x["stage5_input"]), -1).mean(axis=1)
        base = (m - 0.5) * 12.0 + self.shift
        return np.stack([base + 3, base + 1, base - 1, base - 3], axis=1).astype(np.float32)


def fake_two_frames(path):
    value = (sum(open(path, "rb").read()) % 97) / 97.0
    direct = np.full((512, 512, 3), value, np.float32)
    return direct, np.clip(direct + 0.05, 0, 1), (0, 1, 0, 1), (1, 1)


class ExternalEvalTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory(ignore_cleanup_errors=True)
        self.drive = os.path.join(self.tmp.name, "drive")
        rng = np.random.default_rng(1)
        self.ids = [f"img{k}.jpg" for k in range(40)]
        self.grades = rng.integers(0, 5, 40)
        for model, arm in fx.ARMS.items():                                # six finished joint runs
            for seed in fx.SEEDS:
                run = jt.run_dir_for(os.path.join(self.drive, "experiments"), seed, arm)
                os.makedirs(os.path.join(run, "checkpoints", "best_a"))
                os.makedirs(os.path.join(run, "metrics"))
                path = os.path.join(run, "checkpoints", "best_a", "model.weights.h5")
                open(path, "wb").write(f"{model}{seed}".encode())
                result = {"best": {"checkpoint": {"slot": "best_a", "weights_sha256": ig.sha256_file(path), "epoch_index": 7}},
                          "continuation": {"continue": True}}
                json.dump(result, open(os.path.join(run, "result.json"), "w"))
        for dataset, tables in fx.STORED.items():                         # the stored reference tables
            for m, pattern in tables.items():
                for seed in fx.SEEDS:
                    path = os.path.join(self.drive, *pattern.format(seed=seed).split("/"))
                    os.makedirs(os.path.dirname(path), exist_ok=True)
                    z = (self.grades[:, None] - np.arange(4)[None, :] - 0.5) * 2.0 + rng.normal(0, 1, (40, 4))
                    pd.DataFrame(at.metrics_from_logits(self.ids, self.grades, z)[1]).to_csv(path, index=False)
        self.protocol_path = os.path.join(self.tmp.name, "protocol.json")

    def tearDown(self):
        self.tmp.cleanup()

    def test_the_protocol_is_written_once_from_the_six_runs_and_pins_everything(self):
        protocol, sha = fx.make_protocol(self.drive, self.protocol_path)
        self.assertEqual(sorted(protocol["checkpoints"]), sorted(f"{m}_seed{s}" for m in fx.ARMS for s in fx.SEEDS))
        self.assertEqual(protocol["comparisons"]["main"], ["j_minus_p", "jc_minus_p"])
        self.assertIn("equal standing", protocol["comparisons"]["standing"])
        self.assertTrue(any("post hoc" in d for d in protocol["disclosures"]))
        self.assertTrue(any("re-use" in d for d in protocol["disclosures"]))
        with self.assertRaises(fx.ProtocolError):                         # written once
            fx.make_protocol(self.drive, self.protocol_path)
        loaded, loaded_sha, _, parent = fx.load_protocol(self.protocol_path)
        self.assertEqual(loaded_sha, sha)
        files = fx.verify_files(loaded, self.drive, parent)
        self.assertEqual(len(files["checkpoints"]), 6)
        victim = os.path.join(self.drive, *loaded["checkpoints"]["jc_seed123"]["path"].split("/"))
        open(victim, "ab").write(b"changed")
        with self.assertRaises(fx.ProtocolError):
            fx.verify_files(loaded, self.drive, parent)

    def test_an_unfinished_run_stops_the_protocol(self):
        os.remove(os.path.join(jt.run_dir_for(os.path.join(self.drive, "experiments"), 2026, "crop"), "result.json"))
        with self.assertRaises(fx.ProtocolError):
            fx.make_protocol(self.drive, self.protocol_path)

    def test_prediction_tables_and_analysis(self):
        import geometry_diagnostic as gd
        protocol, _ = fx.make_protocol(self.drive, self.protocol_path)
        images = os.path.join(self.tmp.name, "images")
        os.makedirs(images)
        for name in self.ids:
            open(os.path.join(images, name), "wb").write(name.encode() * 3)
        original = gd.two_frames
        gd.two_frames = fake_two_frames
        try:
            models = {(m, s): FakeNet(0.1 * k) for k, (m, s) in enumerate((m, s) for m in fx.ARMS for s in fx.SEEDS)}
            logits = fx.predict(models, [os.path.join(images, n) for n in self.ids], 8, log=lambda *a: None)
        finally:
            gd.two_frames = original
        self.assertEqual(sorted(logits), sorted(models))
        self.assertTrue(all(z.shape == (40, 4) for z in logits.values()))
        self.assertFalse(np.allclose(logits[("j", 42)], logits[("jc", 42)]))      # the two arms see different frames
        tables, frames = fx._tables(self.drive, protocol, "ddr", self.ids, self.grades, logits)
        self.assertEqual(sorted(tables[42]), sorted(fx.MODELS))
        families = np.asarray(["timestamp" if k % 2 else "007" for k in range(40)])
        out = fx.analyse(tables, frames, np.ones(40, bool), families)
        self.assertEqual(sorted(out["contrasts_qwk"]), sorted(fx.CONTRASTS))
        self.assertIn(out["contrasts_qwk"]["j_minus_p"]["reading"].split(":")[0], ("supported", "reversed", "no difference shown"))
        self.assertEqual(out["metric_block"]["paired_differences"]["j_minus_p"]["role"], "primary")
        self.assertEqual(out["metric_block"]["paired_differences"]["jc_minus_p"]["role"], "primary")
        self.assertEqual(out["metric_block"]["paired_differences"]["jc_minus_j"]["role"], "secondary, descriptive")
        self.assertEqual(sorted(out["by_family"]["007"]), ["j_minus_p", "jc_minus_p", "n"])
        self.assertEqual(sorted(out["by_family"]), ["007", "limit", "timestamp"])
        with self.assertRaises(fx.ProtocolError):                         # stored tables of other images
            fx._tables(self.drive, protocol, "ddr", list(reversed(self.ids)), self.grades, logits)


class DemoFigureTests(unittest.TestCase):
    def test_the_summary_picture_is_written(self):
        import demo_pipeline as demo
        rng = np.random.default_rng(0)
        result = {"image": "x.png", "grade": 2, "grade_name": demo.GRADES[2], "class_probabilities": [0.1, 0.2, 0.5, 0.1, 0.1], "referable": True,
                  "per_seed_grades": {"42": 2, "123": 2, "2026": 3}, "uncertainty": {"entropy": 1.2, "seed_disagreement": 0.3, "seeds_agree": False},
                  "quality": {"label": "Good"}, "flag_for_review": True}
        views = {"frame": rng.random((64, 64, 3)), "vessel": rng.random((64, 64)), "lesion_max": rng.random((64, 64, 4)),
                 "grad_cam": rng.random((64, 64)), "overlay": (rng.random((64, 64, 3)) * 255).astype(np.uint8)}
        with tempfile.TemporaryDirectory() as tmp:
            path = demo.figure(result, views, os.path.join(tmp, "x.png"))
            self.assertGreater(os.path.getsize(path), 10_000)
        self.assertIn("not a medical device", demo.NOTICE)


if __name__ == "__main__":
    unittest.main()
