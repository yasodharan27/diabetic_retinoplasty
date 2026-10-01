"""CPU unit tests for stage4_v2.py. No weights are downloaded: the pinned-weights path is exercised
with injected downloaders (a failing one, a wrong file, and a locally written stand-in whose SHA is
patched in for the duration of one test)."""
import os
import tempfile
import unittest
from unittest import mock

import numpy as np
import torch

import pipeline_v2_config as v2cfg
import stage34_cache_v2 as cache
import stage4_v2 as s4

CLASSES = v2cfg.STAGE4_V2A_CLASSES


def _model(classes=CLASSES):
    torch.manual_seed(0)
    return s4.build_stage4_model(classes, pretrained=False)[0]


class ModelTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.model = _model()

    def test_k_outputs_for_k4_and_k6(self):
        x = torch.zeros(1, 3, 64, 64)
        with torch.no_grad():
            self.assertEqual(tuple(self.model.eval()(x).shape), (1, 4, 64, 64))
            six = _model(("MA", "HE", "EX", "SE", "NV", "IRMA"))
            self.assertEqual(tuple(six.eval()(x).shape), (1, 6, 64, 64))

    def test_architecture_is_se_resnet101_unet(self):
        import segmentation_models_pytorch as smp
        self.assertIsInstance(self.model, smp.Unet)
        self.assertEqual(type(self.model.encoder).__name__, "SENetEncoder")
        self.assertEqual(len(self.model.encoder.layer3), 23)            # ResNet-101 depth
        n = sum(p.numel() for p in self.model.parameters())
        self.assertTrue(45e6 < n < 60e6, n)

    def test_parameter_groups(self):
        groups = s4.parameter_groups(self.model, 1e-3)
        self.assertEqual(groups[0]["lr"], 1e-4)
        self.assertEqual(sum(len(g["params"]) for g in groups), len(list(self.model.parameters())))

    def test_full_frame_prediction_and_cache_maps(self):
        rgb = np.random.default_rng(0).integers(0, 256, (150, 200, 3), dtype=np.uint8)
        probs = s4.predict_full_frame(self.model, s4.resize_full_frame(rgb, 96))
        self.assertEqual(probs.shape, (96, 96, 4))
        self.assertTrue(0.0 <= probs.min() and probs.max() <= 1.0)
        maps = s4.pathology_cache_maps(self.model, rgb, size=96, cache_size=32)
        self.assertEqual((maps.shape, maps.dtype), ((32, 32, 8), np.uint8))
        np.testing.assert_array_equal(maps, cache.pack_pathology_maps(probs, 3))


class PinnedWeightTests(unittest.TestCase):
    def test_download_failure_raises_with_exact_error_and_no_fallback(self):
        def failing(*_args):
            raise OSError("simulated: 503 Service Unavailable")
        with self.assertRaises(s4.Stage4WeightsUnavailableError) as ctx:
            s4.build_stage4_model(CLASSES, pretrained=True, downloader=failing)
        message = str(ctx.exception)
        self.assertIn("OSError: simulated: 503 Service Unavailable", message)
        self.assertIn(v2cfg.STAGE4_ENCODER_HF_REPO, message)
        self.assertIn("No fallback", message)

    def test_unverified_file_raises(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "model.safetensors")
            with open(path, "wb") as fh:
                fh.write(b"not the pinned weights")
            with self.assertRaises(s4.Stage4WeightsUnavailableError):
                s4.fetch_pinned_encoder_weights(downloader=lambda *_: path)

    def test_no_other_encoder_is_ever_constructed(self):
        import segmentation_models_pytorch as smp
        with mock.patch.object(smp, "Unet", wraps=smp.Unet) as unet:
            s4.build_stage4_model(CLASSES, pretrained=False)
        self.assertEqual(unet.call_args.kwargs["encoder_name"], "se_resnet101")
        self.assertIsNone(unet.call_args.kwargs["encoder_weights"])     # smp never downloads

    def test_verified_file_is_loaded_strictly(self):
        from safetensors.torch import save_file
        donor = _model().encoder
        state = {k: v.clone().contiguous() for k, v in donor.state_dict().items()}
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "model.safetensors")
            save_file(state, path)
            patches = {"STAGE4_ENCODER_SHA256": cache.sha256_file(path),
                       "STAGE4_ENCODER_BYTES": os.path.getsize(path)}
            with mock.patch.multiple(v2cfg, **patches):
                torch.manual_seed(123)
                model, prov = s4.build_stage4_model(CLASSES, pretrained=True, downloader=lambda *_: path)
        self.assertEqual(prov["pretrained"]["sha256"], patches["STAGE4_ENCODER_SHA256"])
        for k, v in model.encoder.state_dict().items():
            torch.testing.assert_close(v, state[k], rtol=0, atol=0)


class LossTests(unittest.TestCase):
    def setUp(self):
        g = torch.Generator().manual_seed(0)
        self.b, self.k, self.h = 3, 4, 8
        self.logits = torch.randn(self.b, self.k, self.h, self.h, generator=g, requires_grad=True)
        self.targets = (torch.rand(self.b, self.k, self.h, self.h, generator=g) > 0.8).float()
        self.fov = torch.ones(self.b, 1, self.h, self.h)
        self.fov[:, :, :2, :] = 0
        self.annotated = np.array([[1, 1, 1, 1], [1, 1, 0, 1], [0, 1, 0, 1]], dtype=bool)
        self.w = np.array([2.0, 1.0, 3.0, 5.0], np.float32)
        self.lam = s4.class_loss_weights(CLASSES)

    def _loss(self, logits=None, targets=None, fov=None, annotated=None):
        return s4.partial_label_loss(self.logits if logits is None else logits,
                                     self.targets if targets is None else targets,
                                     self.fov if fov is None else fov,
                                     self.annotated if annotated is None else annotated,
                                     self.w, self.lam)[0]

    def test_matches_the_spec_formula(self):
        z = self.logits.detach().double().numpy()
        y = self.targets.double().numpy()
        f = self.fov[:, 0].double().numpy()
        sig = 1 / (1 + np.exp(-z))
        total, active = 0.0, 0
        for c in range(self.k):
            rows = np.flatnonzero(self.annotated[:, c])
            if not rows.size:
                continue
            bce = [(f[i] * (-self.w[c] * y[i, c] * np.log(sig[i, c])
                            - (1 - y[i, c]) * np.log(1 - sig[i, c]))).sum() / f[i].sum() for i in rows]
            inter = sum((f[i] * sig[i, c] * y[i, c]).sum() for i in rows)
            denom = sum((f[i] * (sig[i, c] + y[i, c])).sum() for i in rows)
            total += self.lam[c] * (np.mean(bce) + 1 - (2 * inter + 1) / (denom + 1))
            active += 1
        self.assertAlmostEqual(float(self._loss()), total / active, places=5)

    def test_unannotated_classes_have_zero_loss_and_zero_gradient(self):
        self._loss().backward()
        grad = self.logits.grad
        for i in range(self.b):
            for c in range(self.k):
                if self.annotated[i, c]:
                    self.assertGreater(float(grad[i, c].abs().sum()), 0.0)
                else:
                    self.assertEqual(float(grad[i, c].abs().sum()), 0.0)
        # changing an unannotated target (even to all-positive) changes nothing
        altered = self.targets.clone()
        altered[2, 0] = 1.0
        altered[1, 2] = 1.0
        self.assertEqual(float(self._loss()), float(self._loss(targets=altered)))

    def test_class_with_no_annotated_image_is_skipped(self):
        annotated = self.annotated.copy()
        annotated[:, 2] = False
        loss, per_class = s4.partial_label_loss(self.logits, self.targets, self.fov, annotated,
                                                self.w, self.lam)
        self.assertIsNone(per_class[2])
        self.assertTrue(torch.isfinite(loss))
        loss, _ = s4.partial_label_loss(self.logits, self.targets, self.fov,
                                        np.zeros((self.b, self.k), bool), self.w, self.lam)
        self.assertEqual(float(loss.detach()), 0.0)

    def test_outside_fov_is_ignored(self):
        logits = self.logits.detach().clone()
        logits[:, :, :2, :] = 50.0
        targets = self.targets.clone()
        targets[:, :, :2, :] = 0.0
        self.assertAlmostEqual(float(self._loss(logits=logits, targets=targets)),
                               float(self._loss(logits=self.logits.detach())), places=6)

    def test_weights(self):
        np.testing.assert_allclose(s4.positive_weights([100, 1, 50], [100, 10_000, 5_000]),
                                   [1.0, 20.0, 10.0])
        with self.assertRaises(ValueError):
            s4.positive_weights([0], [10])
        np.testing.assert_array_equal(s4.class_loss_weights(("MA", "HE", "EX", "SE", "NV")),
                                      [1, 1, 1, 1, 0.5])


class GeometryAndSamplingTests(unittest.TestCase):
    def test_full_frame_resize_matches_the_canonical_frame_operation(self):
        import joint_training_dataset as jtd
        rgb = np.random.default_rng(1).integers(0, 256, (150, 200, 3), dtype=np.uint8)
        np.testing.assert_array_equal(s4.resize_full_frame(rgb, 96), jtd._resize_rgb_01(rgb, (96, 96)))

    def test_mask_resize_and_fov(self):
        mask = np.zeros((300, 300), bool)
        mask[:150] = True
        out = s4.resize_mask_full_frame(mask, 96)
        self.assertEqual((out.shape, out.dtype), ((96, 96), np.uint8))
        self.assertEqual(int(out[:47].min()), 1)
        self.assertEqual(int(out[49:].max()), 0)
        rgb = np.zeros((4, 4, 3), np.float32)
        rgb[1:3, 1:3] = 0.5
        self.assertEqual(int(s4.fov_mask(rgb).sum()), 4)

    def test_class_aware_sampler(self):
        rng = np.random.default_rng(0)
        masks = np.zeros((4, 1536, 1536), np.uint8)
        masks[1, 1500, 20] = 1
        masks[3, 700, 800] = 1
        annotated = np.array([True, True, True, False])
        chosen = set()
        for _ in range(200):
            y0, x0, c = s4.sample_patch_origin(rng, masks, annotated, patch=512, p_lesion=0.5)
            self.assertTrue(0 <= y0 <= 1024 and 0 <= x0 <= 1024)
            chosen.add(c)
            if c == 1:
                self.assertTrue(y0 <= 1500 < y0 + 512 and x0 <= 20 < x0 + 512)
                self.assertEqual((y0, x0), (1024, 0))
        self.assertEqual(chosen, {None, 1})                 # never the unannotated class 3
        self.assertIsNone(s4.sample_patch_origin(rng, masks, annotated, p_lesion=0.0)[2])


class DatasetSpecTests(unittest.TestCase):
    def test_specs(self):
        self.assertTrue(s4.IDRID_SEG.verified)
        self.assertTrue(s4.TJDR.verified)                  # §43; details in tests/test_tjdr_dataset.py
        unverified = s4.DatasetSpec("X", ("MA",), verified=False)
        with self.assertRaises(RuntimeError):
            s4.require_verified([s4.IDRID_SEG, unverified])
        m = s4.annotated_matrix(["IDRiD-seg"], ("MA", "HE", "NV"))
        np.testing.assert_array_equal(m, [[True, True, False]])


class SaveLoadEmaTests(unittest.TestCase):
    def test_save_load_and_guards(self):
        model = _model()
        with tempfile.TemporaryDirectory() as tmp:
            sha = s4.save_stage4_v2(model, os.path.join(tmp, "run1"), {"classes": list(CLASSES)})
            path = os.path.join(tmp, "run1", "model.pt")
            loaded = s4.load_stage4_v2(path, expected_sha256=sha, classes=CLASSES)
            for (k, a), (_, b) in zip(model.state_dict().items(), loaded.state_dict().items()):
                torch.testing.assert_close(a, b, rtol=0, atol=0, msg=k)
            self.assertTrue(os.path.exists(os.path.join(tmp, "run1", "stage4_model_manifest.json")))
            with self.assertRaises(cache.ManifestMismatchError):
                s4.load_stage4_v2(path, expected_sha256="b" * 64, classes=CLASSES)
            with self.assertRaises(cache.DenyListedModelError):
                s4.load_stage4_v2(path, expected_sha256=v2cfg.LEGACY_STAGE4_SHA256, classes=CLASSES)
            with self.assertRaises(TypeError):
                s4.load_stage4_v2(path)
        with self.assertRaises(cache.LegacyArtifactError):
            s4.load_stage4_v2(os.path.join(v2cfg.LEGACY_DIRS[4], "best_model.keras"),
                              expected_sha256="b" * 64, classes=CLASSES)
        with self.assertRaises(cache.LegacyArtifactError):
            s4.load_stage4_v2(os.path.join(tempfile.gettempdir(), "best_model.keras"),
                              expected_sha256="b" * 64, classes=CLASSES)

    def test_ema(self):
        net = torch.nn.Linear(2, 1)
        ema = s4.ModelEMA(net, decay=0.9)
        before = ema.module.weight.detach().clone()
        with torch.no_grad():
            net.weight.add_(1.0)
        ema.update(net)
        torch.testing.assert_close(ema.module.weight, 0.9 * before + 0.1 * net.weight.detach())


if __name__ == "__main__":
    unittest.main()
