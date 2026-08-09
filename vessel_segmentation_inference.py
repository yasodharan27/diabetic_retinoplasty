"""
Inference for the Vessel Segmentation module (pipeline Stage 03).

Loads the vendored, pretrained LWNet checkpoint (see vessel_segmentation_model.py
and SEGMENTATION_ARCHITECTURE.md Sec 1.2/2) and exposes reusable prediction
functions -- single-image (`predict_vessel_mask`) and batch
(`predict_vessel_mask_batch`) -- plus a `pipeline.SegmentationStage`
implementation (`VesselSegmentationStage`), mirroring the shape of
`image_quality_inference.py`'s IQA module. This stage has no training half
(SEGMENTATION_ARCHITECTURE.md Sec 7.0), so unlike IQA there is no matching
`train_vessel_segmentation.py` -- this module *is* the whole of Stage 03.

The FOV-detection / crop / resize / TTA / threshold / paste-back procedure
below is ported from `external/lwnet/predict_one_image.py`'s `get_fov`,
`get_circ`, `create_circular_mask`, `crop_to_fov`, and `create_pred`
functions (not imported -- that file is an argparse CLI script, not a
module meant to be imported). One deliberate fix during porting:
`skimage.draw.circle`, which the upstream code uses, was removed in
scikit-image 0.19 in favor of `skimage.draw.disk` -- `_fit_circle` below
uses `disk`, with the same row/col/radius semantics, and is otherwise an
unaltered port of the upstream circle-fitting algorithm.

Stage 02's own preprocessing (Gamma Correction + CLAHE) is untouched by
this module -- everything here runs *after* Stage 02, on its RGB PNG
output, and never re-derives or duplicates that computation.
"""

import os
import time

import numpy as np
import torch
from PIL import Image
from scipy.ndimage import binary_fill_holes
from scipy.optimize import fmin
from skimage.draw import disk
from skimage.exposure import equalize_adapthist
from skimage.filters import threshold_minimum
from skimage.measure import regionprops
from skimage.transform import resize as sk_resize
from torchvision import transforms

import config
from pipeline import SegmentationStage
from vessel_segmentation_model import (
    build_vessel_segmentation_model,
    load_state_dict_from_checkpoint,
    resolve_device,
)

DEFAULT_MODEL_PATH = os.path.join(config.VESSEL_SEG_MODEL_DIR, "best_model.pth")

# Fixed by the vendored `wnet_drive` checkpoint's own config.cfg (im_size).
MODEL_INPUT_SIZE = 512

# The LWNet authors' own DRIVE-tuned binarizing threshold (predict_one_image.py's
# --bin_thresh default). Used here as a documented starting default, NOT a
# value validated against this project's datasets (EyeQ/APTOS2019/IDRiD) --
# see SEGMENTATION_ARCHITECTURE.md Sec 2.3's distribution-shift caveat.
LWNET_REFERENCE_THRESHOLD = 0.4196

_INPUT_TRANSFORM = transforms.Compose([
    transforms.Resize((MODEL_INPUT_SIZE, MODEL_INPUT_SIZE)),
    transforms.ToTensor(),  # HxWxC uint8 [0,255] -> CxHxW float32 [0,1]; no mean/std normalization
])


# --- FOV mask (ported from predict_one_image.py, skimage.draw.circle -> disk) ---

def _fit_circle(binary_mask):
    """Fit a circle to `binary_mask` by local search, starting from its
    largest connected region's centroid/major-axis-length. Returns
    (col0, row0, radius) in image (x, y) convention, matching
    `external/lwnet/predict_one_image.py`'s `get_circ` exactly, except for
    the `skimage.draw.circle` -> `skimage.draw.disk` swap (removed in
    scikit-image >=0.19; `disk((row, col), radius, shape=...)` returns the
    same `(rr, cc)` pixel-coordinate pair `circle(row, col, radius, ...)` did)."""
    labeled = binary_mask.astype(int)
    region = regionprops(labeled)[0]
    row0, col0 = region.centroid
    radius0 = region.major_axis_length / 2.0

    def cost(params):
        col, row, r = params
        rr, cc = disk((row, col), r, shape=labeled.shape)
        template = np.zeros_like(labeled)
        template[rr, cc] = 1
        return -np.sum(template == labeled)

    col0, row0, radius0 = fmin(cost, (col0, row0, radius0), disp=False)
    return col0, row0, radius0


def _circular_mask(shape, center, radius):
    """Boolean circular mask of `shape`, ported unchanged from
    `create_circular_mask` (`center` is (x/col, y/row), matching `_fit_circle`'s
    return convention)."""
    h, w = shape
    col_c, row_c = center
    y_idx, x_idx = np.ogrid[:h, :w]
    dist_from_center = np.sqrt((x_idx - col_c) ** 2 + (y_idx - row_c) ** 2)
    return dist_from_center <= radius


def compute_fov_mask(pil_image):
    """Compute the field-of-view (fundus circle) boolean mask for
    `pil_image`, at `pil_image`'s own native resolution. Ported from
    `get_fov`: CLAHE-like contrast-enhance (`equalize_adapthist`) -> minimum
    thresholding -> hole-filling -> circle fit -> circular mask, upsampled
    back to the original resolution. Large images (>500px on the smaller
    edge) are downscaled before the circle fit purely for speed/robustness,
    same as upstream -- the returned mask is always native resolution."""
    original_size = pil_image.size  # PIL convention: (W, H)
    working_image = pil_image
    if max(original_size) > 500:
        working_image = transforms.Resize(500)(pil_image)

    with np.errstate(divide="ignore"):
        value_channel = equalize_adapthist(np.array(working_image))[:, :, 1]
    threshold = threshold_minimum(value_channel)
    binary = binary_fill_holes(value_channel > threshold)

    col0, row0, radius = _fit_circle(binary)
    small_mask = _circular_mask(binary.shape, center=(col0, row0), radius=radius)

    full_h, full_w = original_size[1], original_size[0]
    # Upsample the small-scale mask back to native resolution (a smooth
    # resize + 0.5 threshold, rather than upstream's PIL-Resize-of-a-bool-
    # image round trip -- same intent: a clean circular mask at full size).
    fov_mask = sk_resize(small_mask.astype(float), (full_h, full_w), order=1) > 0.5
    return fov_mask


def crop_to_fov(image_array, fov_mask):
    """Crop `image_array` (H, W, 3) to `fov_mask`'s bounding box. Ported
    from `crop_to_fov`, operating on numpy arrays directly rather than
    round-tripping through PIL. Returns (crop, (minr, minc, maxr, maxc))."""
    minr, minc, maxr, maxc = regionprops(fov_mask.astype(int))[0].bbox
    crop = image_array[minr:maxr, minc:maxc]
    return crop, (minr, minc, maxr, maxc)


# --- Preprocessing + model I/O ---

def _load_rgb_array(image):
    """Accept a file path, a PIL Image, or an HxWx3 uint8 numpy array, and
    return an HxWx3 uint8 RGB numpy array in every case."""
    if isinstance(image, str):
        return np.array(Image.open(image).convert("RGB"))
    if isinstance(image, Image.Image):
        return np.array(image.convert("RGB"))
    if isinstance(image, np.ndarray):
        if image.ndim != 3 or image.shape[2] != 3:
            raise ValueError(f"Expected an (H, W, 3) RGB array, got shape {image.shape}.")
        return image.astype(np.uint8) if image.dtype != np.uint8 else image
    raise TypeError(f"Unsupported image type {type(image)!r}: expected a path, PIL.Image, or numpy array.")


def _predict_probability_map(model, tensor, fov_mask, coords_crop, crop_shape, tta):
    """Run the model (optionally with the LWNet authors' 4-view flip TTA),
    resize the 512x512 prediction back to the FOV crop's own size, paste it
    into a full-resolution zero canvas, and zero everything outside the FOV
    mask. Ported from `predict_one_image.py`'s `create_pred` (`tta='no'` vs.
    `'from_preds'` collapsed to a bool here, per this project's request to
    keep TTA as a simple on/off switch rather than reproducing the upstream
    `from_logits` variant, which this project does not use)."""
    with torch.no_grad():
        output = model(tensor.unsqueeze(0))
    if isinstance(output, tuple):
        # Should be unreachable: build_vessel_segmentation_model() always
        # sets model.mode = "eval". Verified explicitly here rather than
        # assumed, since this exact bug (mode vs. .eval()) is easy to
        # reintroduce silently.
        raise RuntimeError(
            "WNet.forward() returned (x1, x2) instead of a single prediction -- "
            "model.mode was not set to 'eval'."
        )
    logits = output.squeeze(0)
    pred = torch.sigmoid(logits)

    if tta:
        with torch.no_grad():
            logits_lr = model(tensor.flip(-1).unsqueeze(0)).squeeze(0).flip(-1)
            logits_ud = model(tensor.flip(-2).unsqueeze(0)).squeeze(0).flip(-2)
            logits_lrud = model(tensor.flip(-1).flip(-2).unsqueeze(0)).squeeze(0).flip(-1).flip(-2)
        pred_lr = torch.sigmoid(logits_lr)
        pred_ud = torch.sigmoid(logits_ud)
        pred_lrud = torch.sigmoid(logits_lrud)
        pred = torch.mean(torch.stack([pred, pred_lr, pred_ud, pred_lrud]), dim=0)

    pred = pred.detach().cpu().numpy()[-1]  # (1, 512, 512) -> (512, 512); n_classes=1
    pred = sk_resize(pred, output_shape=crop_shape, order=3)

    full_pred = np.zeros(fov_mask.shape, dtype=np.float64)
    minr, minc, maxr, maxc = coords_crop
    full_pred[minr:maxr, minc:maxc] = pred
    full_pred[~fov_mask] = 0
    return full_pred


def load_vessel_model(model_path=DEFAULT_MODEL_PATH, device=None):
    """Build the vessel segmentation architecture and load the vendored
    LWNet checkpoint's weights into it. `device=None` (the default)
    resolves to CUDA when available, else CPU (`resolve_device()`). Raises
    FileNotFoundError with a clear message if the checkpoint hasn't been
    placed at `model_path` -- this stage has no training entry point that
    could produce one."""
    device = resolve_device(device)
    if not os.path.exists(model_path):
        raise FileNotFoundError(
            f"No vessel segmentation checkpoint found at {model_path}. "
            "This stage does not train its own weights -- place the vendored "
            "LWNet checkpoint there (see SEGMENTATION_ARCHITECTURE.md Sec 6)."
        )
    model = build_vessel_segmentation_model()
    model = load_state_dict_from_checkpoint(model, model_path, device=device)
    return model


def predict_vessel_mask(image, model=None, model_path=DEFAULT_MODEL_PATH,
                         tta=None, device=None, threshold=LWNET_REFERENCE_THRESHOLD):
    """
    Run Stage 03 vessel segmentation on a single image (a Stage 02
    processed RGB image, in practice -- but this function has no opinion on
    what produced it, and enforces no Gamma/CLAHE re-check).

    `image` may be a file path, a PIL Image, or an (H, W, 3) uint8 RGB
    numpy array. Pass an already-loaded `model` (`load_vessel_model()`) to
    reuse it across many calls; otherwise it is loaded from `model_path` on
    every call. `tta=None` (the default) resolves to `config.VESSEL_SEG_TTA`.

    `device` only controls where a *freshly-loaded* model ends up (when
    `model` is not given) -- once a model exists (whether just loaded here
    or passed in by the caller), every input tensor is placed on **that
    model's own actual device** (`next(model.parameters()).device`), never
    on a separately/independently guessed device. This is what makes it
    impossible for a caller to end up with a "model on CUDA, tensor on CPU"
    (or vice versa) mismatch: there is exactly one place device placement
    is decided, and it always follows the model, not the other way around.

    Returns a dict:
      - "probability_map": (H, W, 1) float32 in [0, 1] -- the primary output.
      - "binary_mask": (H, W, 1) uint8 {0, 1}, thresholded at `threshold`.
      - "threshold_used": the float threshold applied.
      - "threshold_source": "lwnet_reference_default_not_project_validated"
        when `threshold` is left at its default -- see module docstring.
      - "tta_used": bool.
      - "inference_seconds": wall-clock time for FOV-detection + forward pass(es).
      - "input_shape": (H, W) of `image`, which "probability_map" and
        "binary_mask" always match exactly.
    """
    if model is None:
        model = load_vessel_model(model_path, device=device)
    device = next(model.parameters()).device
    if tta is None:
        tta = config.VESSEL_SEG_TTA

    image_array = _load_rgb_array(image)
    input_shape = image_array.shape[:2]

    start = time.perf_counter()
    fov_mask = compute_fov_mask(Image.fromarray(image_array))
    crop, coords_crop = crop_to_fov(image_array, fov_mask)
    tensor = _INPUT_TRANSFORM(Image.fromarray(crop)).to(device)
    probability_map = _predict_probability_map(
        model, tensor, fov_mask, coords_crop, crop.shape[:2], tta=tta,
    )
    elapsed = time.perf_counter() - start

    probability_map = probability_map.astype(np.float32)
    if probability_map.shape != input_shape:
        raise RuntimeError(
            f"Vessel probability map shape {probability_map.shape} does not match "
            f"input image shape {input_shape} -- Stage 02's dimensions must be preserved."
        )
    binary_mask = (probability_map > threshold).astype(np.uint8)

    return {
        "probability_map": probability_map[..., np.newaxis],
        "binary_mask": binary_mask[..., np.newaxis],
        "threshold_used": float(threshold),
        "threshold_source": (
            "lwnet_reference_default_not_project_validated"
            if threshold == LWNET_REFERENCE_THRESHOLD else "custom"
        ),
        "tta_used": bool(tta),
        "inference_seconds": elapsed,
        "input_shape": input_shape,
    }


def predict_vessel_mask_batch(images, model=None, model_path=DEFAULT_MODEL_PATH,
                               tta=None, device=None, threshold=LWNET_REFERENCE_THRESHOLD):
    """
    Run `predict_vessel_mask` over a list of images (paths, PIL Images, or
    numpy arrays), reusing a single loaded model across all of them.

    Each image is FOV-cropped to a different size, so -- unlike
    `image_quality_inference.predict_quality_batch`'s `tf.data`-batched
    approach -- this is a plain per-image loop, not a single batched
    forward pass; this stage's images are few enough per pipeline run
    (one per fundus photo) that this is not a throughput concern.

    Returns a list of `predict_vessel_mask` result dicts, in the same order
    as `images`, each augmented with "filename"/"path" when the
    corresponding input was a file path.
    """
    if model is None:
        model = load_vessel_model(model_path, device=device)

    results = []
    for image in images:
        # No `device=` passed through here: predict_vessel_mask always
        # derives it from `model` itself once a model exists, so every
        # image in this loop is guaranteed to run on the same device this
        # (single, shared) model actually lives on.
        result = predict_vessel_mask(
            image, model=model, tta=tta, threshold=threshold,
        )
        if isinstance(image, str):
            result["path"] = image
            result["filename"] = os.path.basename(image)
        results.append(result)
    return results


class VesselSegmentationStage(SegmentationStage):
    """`pipeline.SegmentationStage` implementation wrapping the pretrained
    LWNet checkpoint. `train()`/`evaluate()`/`save()` are not backed by a
    real workflow -- this stage never trains within this project (see
    SEGMENTATION_ARCHITECTURE.md Sec 5/7.0) -- and raise `NotImplementedError`
    with an explanatory message rather than silently no-op-ing."""

    def __init__(self, device=None):
        # Only controls where load() places a freshly-loaded model --
        # predict()/predict_batch() always defer to self.model's own actual
        # device afterward (see predict_vessel_mask's docstring), so this
        # is never a second, independent source of truth.
        self.device = resolve_device(device)
        self.model = None

    def load(self, path=DEFAULT_MODEL_PATH):
        self.model = load_vessel_model(path, device=self.device)
        return self

    def predict(self, input_data):
        if self.model is None:
            raise RuntimeError("VesselSegmentationStage.load() must be called before predict().")
        return predict_vessel_mask(input_data, model=self.model)["probability_map"]

    def predict_batch(self, inputs):
        if self.model is None:
            raise RuntimeError("VesselSegmentationStage.load() must be called before predict_batch().")
        results = predict_vessel_mask_batch(inputs, model=self.model)
        return [r["probability_map"] for r in results]

    def train(self, train_data, val_data=None, **kwargs):
        raise NotImplementedError(
            "Vessel Segmentation has no training workflow in this project -- it runs the "
            "pretrained LWNet checkpoint for inference only. See SEGMENTATION_ARCHITECTURE.md Sec 1.2/2/7.0."
        )

    def evaluate(self, eval_data, **kwargs):
        raise NotImplementedError(
            "Vessel Segmentation has no evaluation workflow against project data in this project -- "
            "the vendored checkpoint's own DRIVE benchmark is documented upstream (LWNet's paper/repo), "
            "not reproduced here."
        )

    def save(self, path):
        raise NotImplementedError(
            "Nothing to save -- this stage's model is a vendored pretrained checkpoint "
            f"({DEFAULT_MODEL_PATH}), not a training output this project produces."
        )
