"""
Reliability-Aware Cross-Attention Fusion (RACAF) -- this project's ONE
approved research innovation (`PROJECT_CODE.md`'s "Approved Research
Innovation" section), wrapping Stage 07 (Adaptive Cross-Attention)'s
output before it reaches CORN (Stage 08).

Implements exactly the architecture fixed in `RACAF_ARCHITECTURE.md`
(Sec 4/5/6/7) -- no equation, parameter, or tensor contract here departs
from that document. This module is a direct translation of the approved,
finalized design into Python/Keras/TensorFlow, not a redesign.

Kept in four independently testable pieces, per the approved design's own
modularity requirement (do not hide the mechanism inside one function):

1. TTA view generation (`tta_views`) -- calls the frozen Stage 04 model
   directly, four times, on deterministically transformed copies of its
   native (512,512,4) input, and inverse-transforms each output back to
   canonical alignment. Never calls `predict_lesion_mask()` /
   `predict_lesion_mask_batch()` (`lesion_segmentation_model.py`) inside
   this loop -- that convenience wrapper performs an *additional* resize
   of its output back to the original image's resolution, unrelated to
   the augmentation itself and unnecessary here (`RACAF_ARCHITECTURE.md`
   Sec 4's "Implementation boundary").
2. Reliability computation (`compute_reliability`) -- a fully
   deterministic, non-trainable, pure-NumPy function: population variance
   disagreement, foreground-restricted pooling, per-class reliability
   `kappa`, burden-weighted aggregation, scalar image-level reliability
   `r`. Never reads a label, a Dice/IoU score, or any other test-set
   statistic -- its only inputs are Stage 04's own frozen predictions.
3. A per-image disk cache (`get_or_compute_reliability`) -- mirrors
   `local_feature_extraction_dataset.py`'s `_get_or_compute_lesion_maps`
   pattern exactly, but is a NEW cache (that existing one only covers a
   single, non-TTA prediction already used by Stage 05) storing only the
   small derived `kappa`/`r` values, not the four raw probability maps.
4. The trainable RACAF fusion model (`build_racaf_fusion`) -- a small
   Keras model with exactly two learned pieces: the reliability gate
   (`w_g, b_g`) and the Global readout projection (`W_r, b_r`), consuming
   already-computed `E`, `G`, and `r` (never computing TTA/reliability
   itself). This is the only trainable, jointly-optimized part of RACAF.

RACAF boundary (`RACAF_ARCHITECTURE.md` Sec 3/9): this module reads
Stage 04's frozen predictions (never a ground-truth mask, never a Dice/IoU
value, never a DR label) and Stage 06/07's already-computed outputs
(never recomputing either). It introduces no second attention mechanism,
no new feature extractor, and no classification head -- CORN (not yet
implemented) remains the only classifier in this pipeline.
"""

import os

import numpy as np
import tensorflow as tf
from keras import Input, Model, layers
from keras.saving import register_keras_serializable
from skimage.transform import resize as sk_resize

import config
from feature_fusion import D_MODEL as STAGE7_D_MODEL
from lesion_segmentation_dataset import LESION_CLASSES
from lesion_segmentation_model import (
    DEFAULT_INPUT_SHAPE as STAGE4_INPUT_SHAPE,
    DEFAULT_THRESHOLD,
    load_lesion_model,
)
from pipeline.inference import InferenceStage
from pipeline.trainable import TrainableStage
from swin_transformer import (
    DUAL_SCALE_OUTPUT_CHANNELS as STAGE6_CHANNELS,
    DUAL_SCALE_OUTPUT_TOKENS as STAGE6_TOKENS,
)

# --- Fixed by the approved RACAF design (RACAF_ARCHITECTURE.md Sec 4/5/6) ---

TTA_TRANSFORMS = ("identity", "horizontal_flip", "vertical_flip", "rotate_180")
NUM_LESION_CLASSES = len(LESION_CLASSES)  # 4, fixed by LESION_CLASSES order (MA,HE,EX,SE)
FOREGROUND_THRESHOLD = DEFAULT_THRESHOLD  # reused, not reinvented -- lesion_segmentation_model.py
DELTA_MAX = 0.25  # max POPULATION variance of 4 samples bounded in [0,1] -- see compute_reliability
BURDEN_EPSILON = 1e-8  # standard numerical-stability constant, not fit to any data
ZERO_BURDEN_THRESHOLD = 1e-6  # "effectively zero" total burden -> equal-weight fallback

# Reused from upstream, not duplicated: RACAF's own two tensor inputs.
D_MODEL = STAGE7_D_MODEL  # 256, Stage 07's E
GLOBAL_TOKENS = STAGE6_TOKENS  # 64, Stage 06's G token count
GLOBAL_CHANNELS = STAGE6_CHANNELS  # 1152, Stage 06's C_G

DEFAULT_MODEL_PATH = os.path.join(config.RACAF_MODEL_DIR, "best_model.keras")
DEFAULT_CACHE_DIR = os.path.join(config.RACAF_RESULTS_DIR, "reliability_cache")


# =====================================================================
# 1. TTA view generation -- frozen Stage 04, called directly, 4 views.
# =====================================================================

def load_frozen_stage4_model(model_path=None):
    """Loads Stage 04 (frozen, Experiment 2C) via the existing, unmodified
    `lesion_segmentation_model.load_lesion_model()`, then explicitly sets
    `trainable = False`. `load_lesion_model()` recompiles the model with a
    live optimizer on load and does not itself set `trainable=False` --
    this is defense-in-depth alongside the `tf.stop_gradient` applied in
    `tta_views()` below, matching `RACAF_ARCHITECTURE.md` Sec 7 and the
    same frozen-upstream-input precaution `local_feature_extraction_model.py`'s
    `_StopGradientBoundary` layer already establishes for an analogous case
    in this project. Does not modify `lesion_segmentation_model.py`."""
    model = load_lesion_model(model_path) if model_path else load_lesion_model()
    model.trainable = False
    return model


def prepare_stage4_input(rgb_image, vessel_probability_map, model_input_shape=STAGE4_INPUT_SHAPE[:2]):
    """Prepares Stage 04's native `(512, 512, 4)` input tensor ONCE, from
    an RGB image (`(H, W, 3)`, `uint8` or float) and its Stage 03 vessel
    probability map -- the same RGB/255 + vessel concatenation + resize
    logic `predict_lesion_mask()` uses internally (`lesion_segmentation_model.py`),
    extracted here so RACAF's four TTA views can all reuse this single
    resize, rather than `predict_lesion_mask()`'s own per-call resize (and
    its additional, unrelated resize-back-to-original-resolution step --
    see this module's docstring and `RACAF_ARCHITECTURE.md` Sec 4).
    Returns `(1, 512, 512, 4)` float32."""
    rgb = np.asarray(rgb_image, dtype=np.float32)
    if rgb.max() > 1.0:
        rgb = rgb / 255.0
    vessel = np.asarray(vessel_probability_map, dtype=np.float32)
    if vessel.ndim == 2:
        vessel = vessel[..., np.newaxis]
    combined = np.concatenate([rgb, vessel], axis=-1)
    resized = sk_resize(
        combined, (*model_input_shape, combined.shape[-1]),
        order=1, mode="reflect", anti_aliasing=True, preserve_range=True,
    ).astype(np.float32)
    return resized[np.newaxis, ...]


def _apply_transform(x, transform):
    """Applies one of the four deterministic geometric transforms to a
    `(B, H, W, C)` tensor. Every transform here is its own exact inverse
    (a flip undoes a flip; a 180-degree rotation undone by another
    180-degree rotation is the identity) -- this function is therefore
    also `_invert_transform`, reused as such in `tta_views()` below, per
    `RACAF_ARCHITECTURE.md` Sec 4's "exact pixel permutation" requirement:
    no interpolation is introduced by any of these four operations."""
    if transform == "identity":
        return x
    if transform == "horizontal_flip":
        return tf.reverse(x, axis=[2])
    if transform == "vertical_flip":
        return tf.reverse(x, axis=[1])
    if transform == "rotate_180":
        return tf.reverse(x, axis=[1, 2])
    raise ValueError(f"Unknown TTA transform: {transform!r}. Expected one of {TTA_TRANSFORMS}.")


def tta_views(stage4_model, prepared_input):
    """Runs the frozen Stage 04 `stage4_model` directly (never through
    `predict_lesion_mask()`/`predict_lesion_mask_batch()`) on each of the
    four deterministically transformed copies of `prepared_input`
    (`(B, 512, 512, 4)`, already resized once via `prepare_stage4_input`),
    inverse-transforming each prediction back to canonical alignment.

    Returns a stacked tensor `(B, 4, 512, 512, 4)` -- one aligned
    `(B, 512, 512, 4)` probability map per view, in `TTA_TRANSFORMS`
    order. Each prediction is wrapped in `tf.stop_gradient` immediately,
    so no gradient from anything computed downstream of this function can
    reach `stage4_model`'s parameters, regardless of whether the model
    passed in already has `trainable=False` set (see
    `load_frozen_stage4_model`)."""
    prepared_input = tf.convert_to_tensor(prepared_input, dtype=tf.float32)
    aligned_views = []
    for transform in TTA_TRANSFORMS:
        transformed_input = _apply_transform(prepared_input, transform)
        raw_prediction = stage4_model(transformed_input, training=False)
        raw_prediction = tf.stop_gradient(raw_prediction)
        aligned_prediction = _apply_transform(raw_prediction, transform)  # self-inverse
        aligned_views.append(aligned_prediction)
    return tf.stack(aligned_views, axis=1)


# =====================================================================
# 2. Reliability computation -- deterministic, non-trainable, no labels.
# =====================================================================

def compute_reliability(aligned_predictions):
    """Computes RACAF's reliability signal from `aligned_predictions`
    (`(B, 4, H, W, C)`, `C=NUM_LESION_CLASSES`, the output of `tta_views`)
    -- exactly `RACAF_ARCHITECTURE.md` Sec 5's formulation, with no
    ground truth, label, or evaluation metric read anywhere in this
    function; its only input is Stage 04's own frozen predictions.

    Returns a dict:
      - "p_bar": (B,H,W,C) mean prediction across the 4 views.
      - "D": (B,H,W,C) per-pixel disagreement -- POPULATION variance
        (N=4, ddof=0) across the 4 views. `np.var`'s default already is
        `ddof=0`, but it is passed explicitly here for auditability, per
        `RACAF_ARCHITECTURE.md` Sec 4/5: this convention is what makes
        DELTA_MAX=0.25 correct (verified: population variance of the
        maximally-disagreeing 4-sample set {0,0,1,1} is exactly 0.25;
        sample variance, ddof=1, would give 0.333 instead, breaking the
        kappa in [0,1] guarantee below).
      - "delta": (B,C) foreground-restricted mean disagreement per class.
      - "kappa": (B,C) per-class reliability, in [0,1].
      - "burden": (B,C) predicted lesion burden per class.
      - "burden_weight": (B,C) burden-normalized class weights, summing to 1.
      - "r": (B,) scalar image-level reliability, in [0,1].
    """
    aligned = np.asarray(aligned_predictions, dtype=np.float32)
    batch_size, num_views, height, width, num_classes = aligned.shape

    p_bar = aligned.mean(axis=1)  # (B,H,W,C)
    D = aligned.var(axis=1, ddof=0)  # (B,H,W,C) -- population variance, N=4, explicit ddof=0

    foreground = p_bar > FOREGROUND_THRESHOLD  # (B,H,W,C) boolean, U_c

    delta = np.zeros((batch_size, num_classes), dtype=np.float32)
    for b in range(batch_size):
        for c in range(num_classes):
            mask = foreground[b, :, :, c]
            if mask.any():
                delta[b, c] = D[b, :, :, c][mask].mean()
            else:
                delta[b, c] = 0.0

    kappa = 1.0 - delta / DELTA_MAX
    # Defensive numerical-stability clip only -- mathematically kappa is
    # already guaranteed in [0,1] given population variance (see docstring
    # above); this guards against floating-point summation noise only, it
    # does not change the design.
    kappa = np.clip(kappa, 0.0, 1.0).astype(np.float32)

    burden = p_bar.mean(axis=(1, 2))  # (B,C)
    total_burden = burden.sum(axis=1, keepdims=True)  # (B,1)
    burden_weight = burden / (total_burden + BURDEN_EPSILON)
    zero_burden = (total_burden[:, 0] < ZERO_BURDEN_THRESHOLD)
    if zero_burden.any():
        burden_weight = burden_weight.copy()
        burden_weight[zero_burden] = 1.0 / num_classes
    burden_weight = burden_weight.astype(np.float32)

    r = (burden_weight * kappa).sum(axis=1).astype(np.float32)  # (B,)

    return {
        "p_bar": p_bar,
        "D": D,
        "delta": delta,
        "kappa": kappa,
        "burden": burden,
        "burden_weight": burden_weight,
        "r": r,
    }


def compute_image_reliability(stage4_model, prepared_input):
    """Convenience composition of `tta_views` + `compute_reliability` for
    one already-prepared `(B, 512, 512, 4)` input -- kept as a thin
    composition, not a merged function, so each half remains
    independently testable per the approved design's modularity
    requirement."""
    aligned = tta_views(stage4_model, prepared_input)
    return compute_reliability(aligned.numpy())


# =====================================================================
# 3. Per-image reliability cache -- NEW, RACAF-specific (not a reuse of
#    Stage 05's existing single-prediction cache).
# =====================================================================

def reliability_cache_path(cache_dir, id_code):
    """Mirrors `local_feature_extraction_dataset.py`'s `_cache_path()`
    naming convention exactly, in RACAF's own cache directory."""
    return os.path.join(cache_dir, f"APTOS_{id_code}_racaf_reliability.npz")


def get_or_compute_reliability(prepared_input, cache_path, stage4_model):
    """Returns `(kappa, r)` for one image (`kappa`: `(4,)`, `r`: scalar
    float), from `cache_path` if already computed, otherwise via
    `compute_image_reliability` -- cached to disk afterward. Mirrors
    `local_feature_extraction_dataset.py`'s `_get_or_compute_lesion_maps`
    cache-then-reuse pattern, but is a NEW cache: the existing one only
    stores the single, non-TTA prediction Stage 05 already consumes and
    is unrelated to and unaffected by RACAF. Stores only the small derived
    `kappa`/`r` values, never the four raw `(512,512,4)` probability maps
    (`RACAF_ARCHITECTURE.md` Sec 7) -- orders of magnitude smaller, and
    `kappa`/`r` are the only quantities the gate ever consumes. Because
    Stage 04 is frozen, reliability for a given image is deterministic and
    never changes across epochs, so this is computed once and reused
    identically across training, validation, and testing."""
    if os.path.exists(cache_path):
        cached = np.load(cache_path)
        return cached["kappa"], float(cached["r"])
    result = compute_image_reliability(stage4_model, prepared_input)
    kappa = result["kappa"][0]
    r = float(result["r"][0])
    os.makedirs(os.path.dirname(cache_path), exist_ok=True)
    np.savez(cache_path, kappa=kappa, r=np.float32(r))
    return kappa, r


# =====================================================================
# 4. RACAF trainable fusion model -- gate + Global readout only.
# =====================================================================

@register_keras_serializable(package="racaf")
class _OneMinus(layers.Layer):
    """Computes `1 - x`. A plain `Lambda(lambda x: 1 - x)` does not
    round-trip through a full `.keras` save/load in Keras 3 (a bare
    function reference is not deserializable) -- mirrors
    `local_feature_extraction_model.py`'s `_StopGradientBoundary` and
    `feature_fusion.py`'s custom-layer approach to the same class of
    issue, keeping the whole model fully save/load-safe."""

    def call(self, inputs):
        return 1.0 - inputs


def build_racaf_fusion(d_model=D_MODEL, global_tokens=GLOBAL_TOKENS, global_channels=GLOBAL_CHANNELS):
    """Builds RACAF's own small trainable model -- exactly
    `RACAF_ARCHITECTURE.md` Sec 5's fusion formula:

        gate = sigmoid(w_g * r + b_g)
        G_hat = W_r . GAP(G) + b_r
        F = gate * E + (1 - gate) * G_hat

    Inputs (all already computed elsewhere -- this model does not compute
    TTA, reliability, or Stage 07's cross-attention itself):
      - `E`: `(B, d_model)` -- Stage 07's output, unmodified.
      - `G`: `(B, global_tokens, global_channels)` -- Stage 06's raw,
        already-flattened output, read independently of Stage 07 (never
        derived from `E`).
      - `r`: `(B, 1)` -- precomputed, cached scalar reliability (this
        module's `compute_reliability`/`get_or_compute_reliability`).

    Returns an **uncompiled** `keras.Model`. Only two pieces are
    trainable: the gate (`Dense(1)`, `w_g,b_g`, 2 params) and the Global
    readout projection (`Dense(d_model)` applied to `GAP(G)`, `W_r,b_r`,
    `global_channels*d_model + d_model` params) -- `295,168 + 2 =
    295,170` total for the default `d_model=256`, `global_channels=1152`.
    Everything else (`GlobalAveragePooling1D`, `Multiply`, `Add`,
    `_OneMinus`) is parameter-free. No attention mechanism, no
    convolutional feature extractor, and no classification head is
    introduced -- CORN (not yet implemented) remains the only classifier
    downstream of this model."""
    e_input = Input(shape=(d_model,), name="E")
    g_input = Input(shape=(global_tokens, global_channels), name="G")
    r_input = Input(shape=(1,), name="r")

    gap_g = layers.GlobalAveragePooling1D(name="gap_g")(g_input)
    g_hat = layers.Dense(d_model, name="global_projection")(gap_g)

    gate = layers.Dense(1, activation="sigmoid", name="reliability_gate")(r_input)
    one_minus_gate = _OneMinus(name="one_minus_gate")(gate)

    gated_e = layers.Multiply(name="gated_e")([e_input, gate])
    gated_g_hat = layers.Multiply(name="gated_g_hat")([g_hat, one_minus_gate])
    fused = layers.Add(name="fusion")([gated_e, gated_g_hat])

    outputs = layers.Activation("linear", dtype="float32", name="racaf_output")(fused)

    return Model(inputs=[e_input, g_input, r_input], outputs=outputs, name="racaf_fusion")


# --- pipeline.TrainableStage / pipeline.InferenceStage implementation ---
#
# Not `pipeline.FeatureExtractionStage`, for the same reason
# `AdaptiveCrossAttentionStage` (feature_fusion.py) isn't: RACAF consumes
# three inputs and produces a pooled vector, not a single spatial feature
# map from a single input.

class RACAFStage(TrainableStage, InferenceStage):
    """`pipeline.TrainableStage`/`pipeline.InferenceStage` implementation
    for RACAF's trainable fusion model. Deliberately does NOT wrap the
    frozen Stage 04 TTA/reliability computation -- that stays a separate,
    explicit step (`tta_views`/`compute_reliability`/
    `get_or_compute_reliability`) callers invoke themselves, so each piece
    of RACAF remains independently testable, per the approved design's
    modularity requirement. `predict`/`predict_batch`/`save`/`load`/
    `build` are fully functional; `train`/`evaluate` raise
    `NotImplementedError` -- RACAF has no standalone training procedure of
    its own (see this module's docstring), mirroring
    `AdaptiveCrossAttentionStage.train()`'s identical pattern."""

    def __init__(self, d_model=D_MODEL, global_tokens=GLOBAL_TOKENS, global_channels=GLOBAL_CHANNELS):
        self.d_model = d_model
        self.global_tokens = global_tokens
        self.global_channels = global_channels
        self.model = None

    def train(self, train_data, val_data=None, **kwargs):
        raise NotImplementedError(
            "RACAF has no standalone training procedure in this project -- its two learned "
            "pieces (the reliability gate and the Global readout projection) train only "
            "jointly with Stages 05-08, through the downstream CORN ordinal loss (see "
            "RACAF_ARCHITECTURE.md Sec 7). That joint training script does not exist yet."
        )

    def evaluate(self, eval_data, **kwargs):
        raise NotImplementedError(
            "RACAF has no standalone evaluation metric in this project -- its usefulness is "
            "only observable through the joint downstream pipeline's classification metrics "
            "(RACAF_ARCHITECTURE.md Sec 10), not reproduced here."
        )

    def save(self, path):
        if self.model is None:
            raise RuntimeError("No model to save -- call build() (or assign self.model) first.")
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        self.model.save(path)
        return path

    def load(self, path=DEFAULT_MODEL_PATH):
        self.model = tf.keras.models.load_model(
            path, compile=False, custom_objects={"_OneMinus": _OneMinus},
        )
        return self

    def build(self):
        """Builds a fresh, uncompiled model from `self.d_model`/
        `self.global_tokens`/`self.global_channels` and assigns it to
        `self.model`. Separate from `__init__`, mirroring
        `AdaptiveCrossAttentionStage`'s identical lazy-build pattern."""
        self.model = build_racaf_fusion(
            d_model=self.d_model, global_tokens=self.global_tokens, global_channels=self.global_channels,
        )
        return self.model

    def predict(self, input_data):
        """`input_data`: an `(E, G, r)` triple for a single (un-batched)
        image -- `E`: `(d_model,)`, `G`: `(global_tokens, global_channels)`,
        `r`: a scalar float. Returns `F`, shape `(d_model,)`."""
        if self.model is None:
            raise RuntimeError("RACAFStage.build() or .load() must be called before predict().")
        e, g, r = input_data
        e = np.asarray(e, dtype="float32")[None, ...]
        g = np.asarray(g, dtype="float32")[None, ...]
        r = np.asarray([[r]], dtype="float32")
        return self.model.predict([e, g, r], verbose=0)[0]

    def predict_batch(self, inputs):
        """`inputs`: a list of `(E, G, r)` triples. Returns one fused
        embedding per triple, in the same order."""
        if self.model is None:
            raise RuntimeError("RACAFStage.build() or .load() must be called before predict_batch().")
        e_batch = np.stack([np.asarray(e, dtype="float32") for e, _, _ in inputs], axis=0)
        g_batch = np.stack([np.asarray(g, dtype="float32") for _, g, _ in inputs], axis=0)
        r_batch = np.asarray([[r] for _, _, r in inputs], dtype="float32")
        predictions = self.model.predict([e_batch, g_batch, r_batch], verbose=0)
        return [predictions[i] for i in range(predictions.shape[0])]
