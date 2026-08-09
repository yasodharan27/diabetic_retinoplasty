"""
Regression tests for Stage 03 (Vessel Segmentation) device handling.

Root cause under test: the Colab checkpoint-verification cell raised
    RuntimeError: Input type (torch.FloatTensor) and weight type
    (torch.cuda.FloatTensor) should be the same
because the model was loaded onto CUDA (the notebook's `load_vessel_model()`
call, on a GPU runtime) while a verification tensor was created with plain
`torch.rand(...)` (always CPU) and never moved to match. The underlying fix
is architectural, not a one-off `.to(device)` call: `resolve_device()`
(vessel_segmentation_model.py) is now the single place "which device"
is decided, and `predict_vessel_mask` (vessel_segmentation_inference.py)
derives its input tensor's device from the *model's own actual device*
(`next(model.parameters()).device`) rather than independently recomputing
a default -- so a model and its input can no longer disagree about where
they live, regardless of which device the model happened to be loaded onto.

No GPU is required to run this suite -- CUDA-specific *decisions* are
exercised by mocking `torch.cuda.is_available()`, never by requiring real
CUDA hardware. Every model in this file is the real, untrained
architecture (`build_vessel_segmentation_model()`) with a synthetic,
from-scratch checkpoint built in `setUp` -- the real vendored LWNet
checkpoint (gitignored, not guaranteed present in every environment this
suite runs in) is never touched, per this project's "unit tests use
synthetic/temporary data only" rule (PROJECT_CODE.md).
"""

import os
import sys
import tempfile
import unittest
from unittest import mock

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from vessel_segmentation_model import (  # noqa: E402
    build_vessel_segmentation_model,
    load_state_dict_from_checkpoint,
    resolve_device,
)
from vessel_segmentation_inference import predict_vessel_mask  # noqa: E402


def _synthetic_fundus_image(size=256, seed=0):
    """A bright filled circle on a black background -- enough like a real
    fundus photo's basic light/dark structure for `compute_fov_mask`'s
    circle-fit to succeed on, without needing any real dataset image."""
    rng = np.random.RandomState(seed)
    image = np.zeros((size, size, 3), dtype=np.uint8)
    yy, xx = np.ogrid[:size, :size]
    center = size // 2
    radius = int(size * 0.4)
    circle = (xx - center) ** 2 + (yy - center) ** 2 <= radius ** 2
    base = 140 + rng.randint(-20, 20, size=(size, size, 3))
    image[circle] = np.clip(base[circle], 60, 220).astype(np.uint8)
    return image


class ResolveDeviceTests(unittest.TestCase):
    """`resolve_device()` is the single source of truth every device
    decision in this stage goes through -- these tests cover its four
    possible outcomes directly."""

    def test_explicit_string_device_is_honored(self):
        self.assertEqual(resolve_device("cpu"), torch.device("cpu"))

    def test_explicit_torch_device_object_is_honored(self):
        given = torch.device("cpu")
        self.assertEqual(resolve_device(given), given)

    def test_none_resolves_to_cpu_when_cuda_unavailable(self):
        with mock.patch("torch.cuda.is_available", return_value=False):
            self.assertEqual(resolve_device(None), torch.device("cpu"))

    def test_none_resolves_to_cuda_when_available(self):
        # Mocked deliberately -- proves the *decision logic* picks CUDA
        # when it's reported available; this suite must not require real
        # CUDA hardware to run.
        with mock.patch("torch.cuda.is_available", return_value=True):
            self.assertEqual(resolve_device(None), torch.device("cuda"))

    def test_explicit_device_overrides_cuda_availability(self):
        # Lets a caller deliberately force CPU even on a GPU-equipped
        # runtime -- explicit input always wins over the auto-detected default.
        with mock.patch("torch.cuda.is_available", return_value=True):
            self.assertEqual(resolve_device("cpu"), torch.device("cpu"))


class CheckpointLoaderDevicePlacementTests(unittest.TestCase):
    def setUp(self):
        model = build_vessel_segmentation_model()
        tmp_dir = tempfile.mkdtemp()
        self.addCleanup(lambda: __import__("shutil").rmtree(tmp_dir, ignore_errors=True))
        self.checkpoint_path = os.path.join(tmp_dir, "synthetic_checkpoint.pth")
        torch.save(
            {"model_state_dict": model.state_dict(), "optimizer_state_dict": {}, "stats": None},
            self.checkpoint_path,
        )

    def test_default_device_none_resolves_like_resolve_device(self):
        model = build_vessel_segmentation_model()
        with mock.patch("torch.cuda.is_available", return_value=False):
            loaded = load_state_dict_from_checkpoint(model, self.checkpoint_path, device=None)
        self.assertEqual(next(loaded.parameters()).device, torch.device("cpu"))

    def test_every_parameter_and_buffer_ends_up_on_the_requested_device(self):
        # Regression-relevant: a partially-moved model (some tensors still
        # on their original device) is exactly the failure mode that
        # produces "weight type" mismatches deep inside a forward pass.
        model = build_vessel_segmentation_model()
        loaded = load_state_dict_from_checkpoint(model, self.checkpoint_path, device="cpu")
        devices = {p.device for p in loaded.parameters()} | {b.device for b in loaded.buffers()}
        self.assertEqual(devices, {torch.device("cpu")})

    def test_explicit_device_is_used_as_torch_load_map_location(self):
        model = build_vessel_segmentation_model()
        with mock.patch("torch.load", wraps=torch.load) as mocked_load:
            load_state_dict_from_checkpoint(model, self.checkpoint_path, device="cpu")
        _, kwargs = mocked_load.call_args
        self.assertEqual(kwargs["map_location"], torch.device("cpu"))


class PredictVesselMaskDeviceConsistencyTests(unittest.TestCase):
    """End-to-end: proves predict_vessel_mask() follows the *passed-in
    model's* actual device, not an independently-recomputed default --
    the exact bug class this fix addresses."""

    def setUp(self):
        model = build_vessel_segmentation_model()
        tmp_dir = tempfile.mkdtemp()
        self.addCleanup(lambda: __import__("shutil").rmtree(tmp_dir, ignore_errors=True))
        checkpoint_path = os.path.join(tmp_dir, "synthetic_checkpoint.pth")
        torch.save(
            {"model_state_dict": model.state_dict(), "optimizer_state_dict": {}, "stats": None},
            checkpoint_path,
        )
        self.checkpoint_path = checkpoint_path
        self.image = _synthetic_fundus_image()

    def test_tensor_follows_model_device_even_when_cuda_appears_available(self):
        """The exact reported regression: a model already loaded onto CPU
        must not have its input tensor sent to CUDA just because
        torch.cuda.is_available() happens to return True elsewhere in the
        process (e.g. a GPU Colab runtime) -- device must come from the
        model actually passed in, never re-derived independently. Before
        the fix, this raised: "Input type (torch.FloatTensor) and weight
        type (torch.cuda.FloatTensor) should be the same"."""
        model = build_vessel_segmentation_model()
        loaded = load_state_dict_from_checkpoint(model, self.checkpoint_path, device="cpu")

        with mock.patch("torch.cuda.is_available", return_value=True):
            result = predict_vessel_mask(self.image, model=loaded)

        self.assertEqual(result["probability_map"].shape[:2], self.image.shape[:2])
        self.assertTrue(np.isfinite(result["probability_map"]).all())

    def test_passing_a_stale_device_argument_is_overridden_by_the_models_actual_device(self):
        """Even if a caller passes `device="cuda"` alongside an
        already-loaded CPU model, the model's real device wins -- `device`
        only controls where a *freshly loaded* model goes, never overrides
        one that's already loaded (see predict_vessel_mask's docstring)."""
        model = build_vessel_segmentation_model()
        loaded = load_state_dict_from_checkpoint(model, self.checkpoint_path, device="cpu")

        result = predict_vessel_mask(self.image, model=loaded, device="cuda")

        self.assertEqual(result["probability_map"].shape[:2], self.image.shape[:2])

    def test_no_model_given_loads_onto_resolved_device(self):
        with mock.patch("torch.cuda.is_available", return_value=False):
            result = predict_vessel_mask(
                self.image, model=None, model_path=self.checkpoint_path,
            )
        self.assertEqual(result["probability_map"].shape[:2], self.image.shape[:2])


if __name__ == "__main__":
    unittest.main()
