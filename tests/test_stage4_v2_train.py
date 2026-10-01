"""CPU tests for stage4_v2_train.py, stage4_v2_gate.py and colab/common/stage4_v2_setup.py. The training
LOOP is exercised with a two-layer toy network on a 64-px synthetic cache; no Stage-4 model is trained."""
import json
import os
import sys
import tempfile
import unittest

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "-1")

import numpy as np
import torch

import pipeline_v2_config as v2cfg
import stage4_v2 as s4
import stage4_v2_data as data
import stage4_v2_gate as gate
import stage4_v2_train as train

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "colab", "common"))
import stage4_v2_setup  # noqa: E402

try:  # run as tests.<module> or via `unittest discover -s tests`
    from tests.test_stage4_v2_data import _synthetic_cache  # noqa: E402
except ImportError:
    from test_stage4_v2_data import _synthetic_cache  # noqa: E402


class Toy(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = torch.nn.Conv2d(3, 8, 3, padding=1)
        self.decoder = torch.nn.Conv2d(8, 4, 1)

    def forward(self, x):
        return self.decoder(torch.relu(self.encoder(x)))


def _cfg(**kw):
    return train.TrainConfig.from_defaults(**{"iterations": 4, "warmup_steps": 1, "batch_size": 4, "patch": 32,
                                              "val_every": 2, "checkpoint_every": 2, "num_workers": 0,
                                              "amp": False, **kw})


class ConfigTests(unittest.TestCase):
    def test_defaults_match_the_fixed_design(self):
        cfg = train.TrainConfig.from_defaults()
        self.assertTrue(cfg.assert_fixed_design())
        self.assertEqual((cfg.encoder_lr_factor, cfg.grad_clip, cfg.ema_decay, cfg.patch, cfg.p_lesion),
                         (0.1, 1.0, 0.999, 512, 0.5))
        with self.assertRaises(AssertionError):
            train.TrainConfig.from_defaults(encoder_lr_factor=0.2).assert_fixed_design()

    def test_optimizer_lr_ratio_and_schedule(self):
        cfg = _cfg()
        opt = train.build_optimizer(Toy(), cfg)
        lrs = {g["name"]: g["lr"] for g in opt.param_groups}
        self.assertAlmostEqual(lrs["encoder"] / lrs["decoder_head"], 0.1)
        self.assertAlmostEqual(lrs["encoder_no_decay"] / lrs["decoder_head_no_decay"], 0.1)
        self.assertEqual({g["name"]: g["weight_decay"] for g in opt.param_groups}["encoder_no_decay"], 0.0)
        full = train.TrainConfig.from_defaults()
        self.assertAlmostEqual(train.lr_factor(full.warmup_steps - 1, full), 1.0)
        self.assertAlmostEqual(train.lr_factor(full.iterations, full), full.min_lr_factor)
        smp_model, _ = s4.build_stage4_model(v2cfg.STAGE4_V2A_CLASSES, pretrained=False)
        groups = {g["name"]: g["lr"] for g in train.build_optimizer(smp_model, full).param_groups}
        self.assertAlmostEqual(groups["encoder"], 0.1 * groups["decoder_head"])

    def test_loss_weighting(self):
        w = train.loss_weights(np.array([100.0, 1.0, 50.0, 10.0]), np.array([100.0, 10_000.0, 5_000.0, 10.0]))
        np.testing.assert_allclose(w["pos_weight"], [1.0, 20.0, 10.0, 1.0])
        self.assertEqual(w["lambda"], [1.0, 1.0, 1.0, 1.0])          # core classes
        with tempfile.TemporaryDirectory() as tmp:
            _synthetic_cache(tmp)
            tc = data.TrainingCache(tmp, require_complete=False)
            pos, neg = train.pixel_counts(tc)
            self.assertTrue(np.all(pos == 6 * 100))                    # 6 train images x 10x10 lesion
            self.assertTrue(np.all(neg > 0))


class MetricTests(unittest.TestCase):
    def test_ap_matches_sklearn_and_dice_matches_project_definition(self):
        from sklearn.metrics import average_precision_score
        rng = np.random.default_rng(0)
        y = (rng.random((4, 64, 64)) < 0.05).astype(np.uint8)
        p = np.clip(0.6 * y + 0.5 * rng.random((4, 64, 64)), 0, 1)
        acc = train.MetricAccumulator()
        acc.update(p[:, :32], y[:, :32])
        acc.update(p[:, 32:], y[:, 32:])
        res = acc.result()
        for k, c in enumerate(v2cfg.STAGE4_V2A_CLASSES):
            self.assertAlmostEqual(res[c]["aupr"], average_precision_score(y[k].ravel(), p[k].ravel()), places=3)
            import tensorflow as tf
            from training.metrics import dice_coefficient
            ref = float(dice_coefficient(tf.constant(y[k][None], tf.float32), tf.constant(p[k][None], tf.float32)))
            self.assertAlmostEqual(res[c]["dice_pooled_soft"], ref, places=5)

    def test_camera_groups_never_silently_pooled(self):
        self.assertEqual(train.reporting_groups({"dataset": "IDRiD", "camera": "IDRiD-Kowa"}), ["IDRiD-val"])
        g = train.reporting_groups({"dataset": "TJDR", "camera": "TJDR-CLARUS500"})
        self.assertEqual(g[0], "TJDR-val-CLARUS500")
        self.assertIn("TRC50DX+CLARUS500", g[1])                       # combined row is labelled explicitly
        self.assertEqual(train.SELECTION_GROUPS, ("IDRiD-val", "TJDR-val-TRC50DX", "TJDR-val-CLARUS500"))


class LoopTests(unittest.TestCase):
    def test_train_step_updates_ema(self):
        model = Toy()
        ema = s4.ModelEMA(model, 0.999)
        before = [p.detach().clone() for p in ema.module.parameters()]
        cfg = _cfg()
        opt = train.build_optimizer(model, cfg)
        sch = train.build_scheduler(opt, cfg)
        scaler = torch.amp.GradScaler("cuda", enabled=False)
        batch = {"x": torch.randn(2, 3, 16, 16), "y": (torch.rand(2, 4, 16, 16) > 0.9).float(),
                 "fov": torch.ones(2, 1, 16, 16), "annotated": torch.ones(2, 4, dtype=torch.bool)}
        out = train.train_step(model, ema, opt, sch, scaler, batch, train.loss_weights(np.ones(4), np.ones(4) * 9), cfg, "cpu")
        self.assertTrue(np.isfinite(out["loss"]))
        for b, e, m in zip(before, ema.module.parameters(), model.parameters()):
            torch.testing.assert_close(e.detach(), 0.999 * b + 0.001 * m.detach())

    def test_run_checkpoint_metadata_resume_and_fingerprints(self):
        with tempfile.TemporaryDirectory() as tmp:
            cdir, rdir = os.path.join(tmp, "cache"), os.path.join(tmp, "run")
            _synthetic_cache(cdir)
            orig = data.TrainingCache.__init__

            def lenient(self, directory, verify_files=True, require_complete=True):
                orig(self, directory, verify_files, False)
            data.TrainingCache.__init__ = lenient
            try:
                res = train.run(rdir, cdir, cfg=_cfg(), device="cpu", model_factory=Toy, max_steps=2, log=lambda *a: None)
                self.assertEqual(res["step"], 2)
                ck = train.load_checkpoint(os.path.join(rdir, "checkpoints", "latest.pt"))
                for key in ("model", "ema", "optimizer", "scheduler", "scaler", "config", "loss_weights",
                            "fingerprints", "model_weights_sha256", "history", "environment", "rng"):
                    self.assertIn(key, ck)
                fp = ck["fingerprints"]
                self.assertEqual(fp["idrid_split_sha256"], v2cfg.IDRID_V2_SPLIT_SHA256)
                self.assertEqual(fp["encoder_weights_sha256"], v2cfg.STAGE4_ENCODER_SHA256)
                self.assertEqual(fp["tjdr_excluded"]["test"], list(v2cfg.TJDR_EXCLUDED["test"]))
                self.assertEqual(len(fp["training_cache_fingerprint"]), 64)
                self.assertEqual(ck["model_weights_sha256"]["ema"], train.state_dict_sha256(ck["ema"]))
                self.assertTrue(os.path.exists(os.path.join(rdir, "checkpoints", "best_ema.pt")))
                with open(os.path.join(rdir, "validation_history.json"), encoding="utf-8") as fh:
                    hist = json.load(fh)
                self.assertIn("IDRiD-val", hist[0]["validation"])
                self.assertIn("TJDR-val-CLARUS500", hist[0]["validation"])
                res = train.run(rdir, cdir, cfg=_cfg(), device="cpu", model_factory=Toy, log=lambda *a: None)
                self.assertEqual(res["step"], 4)                        # resumed, not restarted
                with self.assertRaises(RuntimeError):                   # changed config -> refused
                    train.run(rdir, cdir, cfg=_cfg(decoder_lr=1e-3), device="cpu", model_factory=Toy, log=lambda *a: None)
            finally:
                data.TrainingCache.__init__ = orig
            ck = torch.load(os.path.join(rdir, "checkpoints", "latest.pt"), weights_only=False)
            ck["ema"][next(iter(ck["ema"]))] += 1
            torch.save(ck, os.path.join(rdir, "checkpoints", "tampered.pt"))
            with self.assertRaises(RuntimeError):
                train.load_checkpoint(os.path.join(rdir, "checkpoints", "tampered.pt"))


class GateTests(unittest.TestCase):
    def test_reference_values_pinned(self):
        self.assertEqual(v2cfg.STAGE4_GATE_REFERENCE_DICE, {"MA": 0.0165, "HE": 0.1273, "EX": 0.3574, "SE": 0.0244})
        self.assertEqual(v2cfg.STAGE4_GATE_MIN_MEAN_AUPR, 0.55)

    def test_refused_without_token_and_only_once(self):
        with tempfile.TemporaryDirectory() as tmp:
            model_path = os.path.join(tmp, "model.pt")
            with self.assertRaises(gate.GateRefused):
                gate.run_gate(model_path, expected_sha256="a" * 64, confirm="yes")
            with open(gate._lock_path(model_path), "w") as fh:
                fh.write("{}")
            with self.assertRaises(gate.GateRefused):
                gate.run_gate(model_path, expected_sha256="a" * 64, confirm=v2cfg.STAGE4_GATE_CONFIRM_TOKEN)

    def test_old_protocol_target_matches_old_loader(self):
        import lesion_segmentation_dataset as lsd
        masks = np.zeros((4, 300, 450), np.uint8)
        masks[0, 100, 200] = 1
        masks[2, 10:60, 30:90] = 1
        mine = gate.old_protocol_target(masks, size=64)
        ref = lsd._resize_target(np.moveaxis(masks, 0, -1).astype(np.float32), (64, 64))
        np.testing.assert_array_equal(mine, np.moveaxis(ref, -1, 0))

    def test_decision_rule(self):
        res = {"old_protocol_512": {c: {"dice_pooled_soft": v + 0.01} for c, v in v2cfg.STAGE4_GATE_REFERENCE_DICE.items()},
               "frame_1536": {"mean_aupr": 0.56}}
        self.assertTrue(gate.decide(res)["PASS"])
        res["old_protocol_512"]["MA"]["dice_pooled_soft"] = 0.0165
        self.assertFalse(gate.decide(res)["PASS"])

    def test_test_images_reachable_only_from_the_gate(self):
        root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        for module in ("stage4_v2_data.py", "stage4_v2_train.py", "tjdr_dataset.py"):
            with open(os.path.join(root, module), encoding="utf-8") as fh:
                self.assertNotIn("b. Testing Set", fh.read(), module)
        with open(os.path.join(root, "stage4_v2_gate.py"), encoding="utf-8") as fh:
            self.assertIn("b. Testing Set", fh.read())


class SetupTests(unittest.TestCase):
    def test_weight_verification_refuses_on_failure(self):
        with self.assertRaises(s4.Stage4WeightsUnavailableError):
            stage4_v2_setup.verify_encoder_weights("/tmp/x", downloader=lambda *a: (_ for _ in ()).throw(OSError("offline")))
        with tempfile.TemporaryDirectory() as tmp:
            bad = os.path.join(tmp, "model.safetensors")
            with open(bad, "wb") as fh:
                fh.write(b"0" * 10)
            with self.assertRaises(s4.Stage4WeightsUnavailableError):
                stage4_v2_setup.verify_encoder_weights(tmp, downloader=lambda *a: bad)

    def test_cache_staging_survives_drive_drop_and_resumes(self):
        import errno
        from unittest import mock
        import dataset_staging
        with tempfile.TemporaryDirectory() as tmp:
            drive, local = os.path.join(tmp, "drive"), os.path.join(tmp, "local")
            _synthetic_cache(drive)
            real_copy, calls, remounts = dataset_staging._copy_one, {"n": 0}, []

            def flaky(src, dst, *a, **k):
                calls["n"] += 1
                if calls["n"] == 3:                      # the mount drops mid-copy, once
                    raise OSError(errno.ENOTCONN, "Transport endpoint is not connected")
                return real_copy(src, dst, *a, **k)

            with mock.patch.object(dataset_staging, "_copy_one", side_effect=flaky):
                res = stage4_v2_setup.stage_training_cache(drive, local, remount=lambda: remounts.append(1),
                                                           log=lambda *a: None)
            self.assertEqual((res["copied"], res["skipped"], res["remounts"]), (8, 0, 1))
            tc = data.TrainingCache(local, verify_files=True, require_complete=False)
            self.assertEqual(len(tc.manifest["files"]), 8)
            with open(os.path.join(drive, "manifest.json"), "rb") as a, open(os.path.join(local, "manifest.json"), "rb") as b:
                self.assertEqual(a.read(), b.read())          # manifest copied byte-for-byte
            again = stage4_v2_setup.stage_training_cache(drive, local, remount=lambda: None, log=lambda *a: None)
            self.assertEqual((again["copied"], again["skipped"]), (0, 8))     # resume skips verified files
            with open(os.path.join(local, "IDRiD__val__IDRiD_05.npz"), "ab") as fh:
                fh.write(b"x")                                                # corrupt one local file
            fixed = stage4_v2_setup.stage_training_cache(drive, local, remount=lambda: None, log=lambda *a: None)
            self.assertEqual((fixed["copied"], fixed["skipped"]), (1, 7))

    def test_tjdr_env_points_at_drive_copy(self):
        class FakeColabConfig:
            DATASET_ROOT = "/content/drive/MyDrive/DiabeticRetinopathy/datasets"
        saved = {k: os.environ.get(k) for k in ("TJDR_RAW_DIR", "TJDR_PROCESSED_DIR")}
        try:
            env = stage4_v2_setup.configure_tjdr_env(FakeColabConfig)
            self.assertEqual(env["TJDR_RAW_DIR"], "/content/drive/MyDrive/DiabeticRetinopathy/datasets/TJDR/raw")
            self.assertTrue(env["TJDR_RAW_DIR"].endswith(v2cfg.TJDR_DRIVE_RAW_DIR))
        finally:
            for k, v in saved.items():
                if v is None:
                    os.environ.pop(k, None)
                else:
                    os.environ[k] = v


if __name__ == "__main__":
    unittest.main()
