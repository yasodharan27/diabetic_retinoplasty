"""
Attention U-Net model + inference for Lesion Segmentation (pipeline
Stage 04).

Unlike `vessel_segmentation_model.py`/`vessel_segmentation_inference.py`
(Stage 03, a vendored pretrained PyTorch checkpoint, inference only), this
stage is trained within this project, in TensorFlow/Keras, matching the
rest of this repository's stack (`SEGMENTATION_ARCHITECTURE.md` Sec 3/6).
Every `pipeline.SegmentationStage` method here (`train`/`evaluate`/`save`/
`load`/`predict`/`predict_batch`) is fully functional, not a
`NotImplementedError` stub.

4-channel input (Stage 02 processed RGB + Stage 03 vessel probability map,
`SEGMENTATION_ARCHITECTURE.md` Sec 3.1), 4-channel sigmoid output (one
probability map per lesion class -- multi-label, since a pixel can belong
to more than one lesion category, Sec 3.3). Optic Disc is never an output
channel (Sec 3.4/Appendix A.4).

Reuses the existing model-agnostic training framework (`training.Trainer`,
`training.TrainingConfig`, `training.losses.bce_dice_loss`,
`training.metrics.build_metrics("segmentation")`) exactly as
`image_quality_model.py`/`train_image_quality.py` already do for Stage 01 --
no training loop, checkpointing, or metric computation is reimplemented
here.
"""

import os

import numpy as np
import tensorflow as tf
from keras import Input, Model, layers

import config
from lesion_segmentation_dataset import LESION_CLASSES
from pipeline import SegmentationStage
from training import (
    Trainer,
    TrainingConfig,
    bce_dice_loss,
    build_metrics,
    dice_coefficient,
    iou_score,
    weighted_bce_dice_loss,
    weighted_pooled_bce_dice_loss,
)
from vessel_segmentation_inference import _load_rgb_array, predict_vessel_mask

try:
    from skimage.transform import resize as sk_resize
except ImportError:  # pragma: no cover -- scikit-image is already a project dependency
    raise

DEFAULT_INPUT_SHAPE = (512, 512, 4)
DEFAULT_MODEL_PATH = os.path.join(config.LESION_SEG_MODEL_DIR, "best_model.keras")
DEFAULT_RUN_DIR = os.path.join(config.LESION_SEG_MODEL_DIR, "training_run")
DEFAULT_THRESHOLD = 0.5

# --- Experiment 2B: moderate class-balanced loss weights ---
# Hand-chosen, MODERATE relative weights -- explicitly NOT raw inverse
# class-pixel-frequency (measured on all 54 training images: Microaneurysm
# 0.1069%, Haemorrhage 0.9857%, HardExudate 0.8066%, SoftExudate 0.1894% of
# all pixels -- an inverse-frequency Microaneurysm weight would be ~9x
# larger than Haemorrhage's alone, likely destabilizing training). These
# values are an experiment parameter to test whether moderate weighting
# helps the two classes Experiment 2A's unweighted per-channel Dice/BCE
# still could not learn (Microaneurysm, SoftExudate) -- NOT a claim that
# they are optimal, and NOT validated against the real 27-image test set
# until that evaluation is actually run. Order matches LESION_CLASSES
# exactly (Microaneurysm, Haemorrhage, HardExudate, SoftExudate).
#
# Experiment 2C reuses these SAME weight values, unchanged -- see
# `weighted_dice_mode="pooled"` below -- to isolate the pooled-vs-per-
# channel Dice question from the separate question of what the weights
# themselves should be.
EXPERIMENT_2B_CLASS_WEIGHTS = (2.0, 1.0, 1.1, 1.8)


# --- Attention U-Net architecture ---

def _conv_block(x, filters, name):
    x = layers.Conv2D(filters, 3, padding="same", use_bias=False, name=f"{name}_conv1")(x)
    x = layers.BatchNormalization(name=f"{name}_bn1")(x)
    x = layers.Activation("relu", name=f"{name}_relu1")(x)
    x = layers.Conv2D(filters, 3, padding="same", use_bias=False, name=f"{name}_conv2")(x)
    x = layers.BatchNormalization(name=f"{name}_bn2")(x)
    x = layers.Activation("relu", name=f"{name}_relu2")(x)
    return x


def _attention_gate(skip, gate, inter_channels, name):
    """Additive attention gate (Oktay et al., 2018, "Attention U-Net"):
    learns a per-pixel gating coefficient in [0, 1] for `skip`'s features
    from the decoder's own (already-upsampled, same-resolution) `gate`
    signal, so the decoder can down-weight irrelevant encoder activations
    before concatenation rather than combining them unweighted, as a plain
    U-Net's skip connections do."""
    theta_x = layers.Conv2D(inter_channels, 1, padding="same", name=f"{name}_theta_x")(skip)
    phi_g = layers.Conv2D(inter_channels, 1, padding="same", name=f"{name}_phi_g")(gate)
    combined = layers.Activation("relu", name=f"{name}_relu")(layers.add([theta_x, phi_g]))
    psi = layers.Conv2D(1, 1, padding="same", activation="sigmoid", name=f"{name}_psi")(combined)
    return layers.multiply([skip, psi], name=f"{name}_attended")


def _decoder_block(x, skip, filters, name):
    """Upsamples `x` to `skip`'s resolution (also used as the attention
    gate's gating signal, so no separate resize is needed inside
    `_attention_gate`), attention-gates `skip`, concatenates, then applies
    a conv block."""
    gate = layers.Conv2DTranspose(filters, 2, strides=2, padding="same", name=f"{name}_upconv")(x)
    attended_skip = _attention_gate(skip, gate, max(filters // 2, 1), name=f"{name}_attn")
    x = layers.concatenate([gate, attended_skip], name=f"{name}_concat")
    return _conv_block(x, filters, name=name)


def build_attention_unet(input_shape=DEFAULT_INPUT_SHAPE, num_classes=4, base_filters=32,
                          learning_rate=1e-4, class_weights=None, weighted_dice_mode="per_channel"):
    """
    Builds and compiles the Attention U-Net: 4 encoder levels + bottleneck
    + 4 decoder levels with attention-gated skip connections, a 1x1 sigmoid
    output conv (`num_classes` channels, multi-label -- not softmax, since
    lesion classes are not mutually exclusive per pixel). `input_shape`'s
    height/width must each be divisible by 16 (4 pooling levels) for the
    encoder/decoder's spatial dimensions to round-trip exactly.

    Loss: `training.losses.bce_dice_loss()` (unweighted, channel-independent
    per-channel Dice + plain BCE -- Experiment 2A's default) when
    `class_weights` is `None`; `weighted_dice_mode` has no effect in that
    case. Passing a length-`num_classes` sequence (e.g.
    `EXPERIMENT_2B_CLASS_WEIGHTS`, above) switches to one of two weighted
    losses depending on `weighted_dice_mode`:
      - `"per_channel"` (default): `training.losses.weighted_bce_dice_loss
        (class_weights)` (Experiment 2B) -- both the BCE and Dice
        components are combined as a *weighted average of per-channel
        ratios*.
      - `"pooled"`: `training.losses.weighted_pooled_bce_dice_loss
        (class_weights)` (Experiment 2C) -- BCE is combined the same way,
        but the Dice component pools every channel's weighted intersection
        and union into a single ratio instead of averaging per-channel
        ratios (Experiment 2's original pooled-Dice formulation, now
        class-weighted). The two are not equivalent in general.
    This is the only place any of these losses is selected; no other
    stage's compilation is affected by this parameter. Metrics:
    `training.metrics.build_metrics("segmentation")` (`dice_coefficient`,
    `iou_score`, computed over all `num_classes` channels combined -- see
    `per_class_segmentation_metrics` for the per-lesion-class breakdown
    used by `LesionSegmentationStage.evaluate()`).
    """
    if weighted_dice_mode not in ("per_channel", "pooled"):
        raise ValueError(
            f"weighted_dice_mode must be 'per_channel' or 'pooled'; got {weighted_dice_mode!r}."
        )
    if class_weights is not None and len(class_weights) != num_classes:
        raise ValueError(
            f"class_weights must have length num_classes ({num_classes}); "
            f"got {len(class_weights)}."
        )

    height, width = input_shape[0], input_shape[1]
    if height % 16 != 0 or width % 16 != 0:
        raise ValueError(
            f"input_shape {input_shape}: height and width must each be divisible by "
            "16 (this architecture has 4 pooling levels)."
        )

    inputs = Input(shape=input_shape, name="lesion_seg_input")

    e1 = _conv_block(inputs, base_filters, "enc1")
    p1 = layers.MaxPooling2D(2, name="pool1")(e1)

    e2 = _conv_block(p1, base_filters * 2, "enc2")
    p2 = layers.MaxPooling2D(2, name="pool2")(e2)

    e3 = _conv_block(p2, base_filters * 4, "enc3")
    p3 = layers.MaxPooling2D(2, name="pool3")(e3)

    e4 = _conv_block(p3, base_filters * 8, "enc4")
    p4 = layers.MaxPooling2D(2, name="pool4")(e4)

    bottleneck = _conv_block(p4, base_filters * 16, "bottleneck")

    d4 = _decoder_block(bottleneck, e4, base_filters * 8, "dec4")
    d3 = _decoder_block(d4, e3, base_filters * 4, "dec3")
    d2 = _decoder_block(d3, e2, base_filters * 2, "dec2")
    d1 = _decoder_block(d2, e1, base_filters, "dec1")

    # Explicit float32 output: keeps sigmoid numerically stable under a
    # mixed_float16 policy (see training.trainer.enable_mixed_precision),
    # matching image_quality_model.py's identical convention.
    outputs = layers.Conv2D(num_classes, 1, activation="sigmoid", dtype="float32",
                             name="lesion_seg_output")(d1)

    model = Model(inputs=inputs, outputs=outputs, name="lesion_segmentation_attention_unet")
    if class_weights is None:
        loss = bce_dice_loss()
    elif weighted_dice_mode == "pooled":
        loss = weighted_pooled_bce_dice_loss(class_weights)
    else:
        loss = weighted_bce_dice_loss(class_weights)
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
        loss=loss,
        metrics=build_metrics("segmentation"),
    )
    return model


def per_class_segmentation_metrics(y_true, y_pred, class_names=LESION_CLASSES):
    """
    Per-lesion-class Dice/IoU plus their overall mean, reusing
    `training.metrics.dice_coefficient`/`iou_score` unmodified -- applied
    once per channel (the compiled model's own `build_metrics
    ("segmentation")` metrics only ever see all `len(class_names)`
    channels combined, since that's how Keras applies a metric function to
    a multi-channel output).

    `y_true`/`y_pred`: arrays or tensors shaped (..., len(class_names)).
    Returns a flat dict: "dice_<ClassName>"/"iou_<ClassName>" per class,
    plus "dice_mean"/"iou_mean".
    """
    y_true = tf.convert_to_tensor(y_true, dtype=tf.float32)
    y_pred = tf.convert_to_tensor(y_pred, dtype=tf.float32)

    results = {}
    dice_scores, iou_scores = [], []
    for i, name in enumerate(class_names):
        d = float(dice_coefficient(y_true[..., i], y_pred[..., i]).numpy())
        v = float(iou_score(y_true[..., i], y_pred[..., i]).numpy())
        results[f"dice_{name}"] = d
        results[f"iou_{name}"] = v
        dice_scores.append(d)
        iou_scores.append(v)
    results["dice_mean"] = float(np.mean(dice_scores))
    results["iou_mean"] = float(np.mean(iou_scores))
    return results


# --- Inference ---

def load_lesion_model(model_path=DEFAULT_MODEL_PATH):
    """Loads a trained Lesion Segmentation `.keras` model. Loaded with
    `compile=False` then recompiled with the same loss/metrics used at
    training time, rather than relying on Keras to deserialize
    `bce_dice_loss()`'s closure by name -- the same `custom_object_scope`
    class of issue `test_hybrid_model.py` already has to work around
    elsewhere in this repo, avoided here by simply not depending on
    loss/metric deserialization at all."""
    if not os.path.exists(model_path):
        raise FileNotFoundError(
            f"No lesion segmentation checkpoint found at {model_path}. Train the "
            "model first (LesionSegmentationStage.train() / the Stage 04 Colab "
            "notebook) and export the best checkpoint to this location."
        )
    model = tf.keras.models.load_model(model_path, compile=False)
    model.compile(
        optimizer=tf.keras.optimizers.Adam(),
        loss=bce_dice_loss(),
        metrics=build_metrics("segmentation"),
    )
    return model


def predict_lesion_mask(image, vessel_probability_map=None, model=None,
                         model_path=DEFAULT_MODEL_PATH, vessel_model=None,
                         threshold=DEFAULT_THRESHOLD):
    """
    Run Stage 04 lesion segmentation on a single image (a Stage 02
    processed RGB image, in practice -- this function has no opinion on
    what produced it).

    `image` may be a file path, a PIL Image, or an (H, W, 3) uint8 RGB
    numpy array (via `vessel_segmentation_inference._load_rgb_array`, reused
    unmodified). Pass an already-loaded `model` to reuse it across many
    calls; otherwise it is loaded from `model_path` on every call.

    `vessel_probability_map` may be a precomputed (H, W) or (H, W, 1) array
    from `vessel_segmentation_inference.predict_vessel_mask()`; if omitted,
    it is computed here via that same unmodified function (`vessel_model`,
    if given, is passed straight through to it).

    Returns a dict:
      - "probability_maps": (H, W, 4) float32 in [0, 1], channel order ==
        LESION_CLASSES.
      - "binary_masks": (H, W, 4) uint8 {0, 1}, thresholded at `threshold`.
      - "class_names": LESION_CLASSES.
      - "threshold_used": float.
      - "input_shape": (H, W) of `image`, which both mask arrays match exactly.
    """
    if model is None:
        model = load_lesion_model(model_path)

    image_array = _load_rgb_array(image)
    input_shape = image_array.shape[:2]

    if vessel_probability_map is None:
        vessel_probability_map = predict_vessel_mask(image_array, model=vessel_model)["probability_map"]
    vessel_probability_map = np.asarray(vessel_probability_map, dtype=np.float32)
    if vessel_probability_map.ndim == 2:
        vessel_probability_map = vessel_probability_map[..., np.newaxis]
    if vessel_probability_map.shape[:2] != input_shape:
        raise RuntimeError(
            f"Vessel probability map shape {vessel_probability_map.shape[:2]} does not "
            f"match input image shape {input_shape}."
        )

    input_array = np.concatenate([image_array.astype(np.float32) / 255.0, vessel_probability_map], axis=-1)

    model_input_shape = model.input_shape[1:3]
    resized_input = sk_resize(
        input_array, (*model_input_shape, input_array.shape[-1]),
        order=1, mode="reflect", anti_aliasing=True, preserve_range=True,
    ).astype(np.float32)

    prediction = model.predict(resized_input[np.newaxis, ...], verbose=0)[0]

    probability_maps = sk_resize(
        prediction, (*input_shape, prediction.shape[-1]),
        order=1, mode="reflect", anti_aliasing=True, preserve_range=True,
    ).astype(np.float32)
    probability_maps = np.clip(probability_maps, 0.0, 1.0)
    binary_masks = (probability_maps > threshold).astype(np.uint8)

    return {
        "probability_maps": probability_maps,
        "binary_masks": binary_masks,
        "class_names": LESION_CLASSES,
        "threshold_used": float(threshold),
        "input_shape": input_shape,
    }


def predict_lesion_mask_batch(images, vessel_probability_maps=None, model=None,
                               model_path=DEFAULT_MODEL_PATH, vessel_model=None,
                               threshold=DEFAULT_THRESHOLD):
    """
    Runs `predict_lesion_mask` over a list of images, reusing a single
    loaded model (and, if given, a single loaded vessel model) across all
    of them -- mirrors `vessel_segmentation_inference.predict_vessel_mask_batch`'s
    plain per-image loop (this stage's images are similarly few per
    pipeline run).

    `vessel_probability_maps`, if given, must be the same length as
    `images` (one precomputed map, or None, per image).

    Returns a list of `predict_lesion_mask` result dicts, in the same
    order as `images`, each augmented with "filename"/"path" when the
    corresponding input was a file path.
    """
    if model is None:
        model = load_lesion_model(model_path)
    if vessel_probability_maps is None:
        vessel_probability_maps = [None] * len(images)

    results = []
    for image, vessel_map in zip(images, vessel_probability_maps):
        result = predict_lesion_mask(
            image, vessel_probability_map=vessel_map, model=model,
            vessel_model=vessel_model, threshold=threshold,
        )
        if isinstance(image, str):
            result["path"] = image
            result["filename"] = os.path.basename(image)
        results.append(result)
    return results


# --- pipeline.SegmentationStage implementation ---

class LesionSegmentationStage(SegmentationStage):
    """`pipeline.SegmentationStage` implementation for the Attention U-Net.
    Unlike `VesselSegmentationStage` (Stage 03, pretrained/inference-only),
    every method here is fully functional -- Lesion Segmentation trains
    within this project (SEGMENTATION_ARCHITECTURE.md Sec 5/7.0)."""

    def __init__(self, input_shape=DEFAULT_INPUT_SHAPE, class_weights=None,
                 weighted_dice_mode="per_channel"):
        """`class_weights`/`weighted_dice_mode`, if given, are forwarded to
        `build_attention_unet` the first time `train()` builds a model
        (i.e. whenever `self.model` is still `None`) -- `class_weights=
        None` (the default) preserves Experiment 2A's plain unweighted
        `bce_dice_loss()`, and `weighted_dice_mode` then has no effect. A
        length-4 `class_weights` sequence (e.g. `EXPERIMENT_2B_CLASS_
        WEIGHTS`) switches that model to a weighted loss: `weighted_dice_
        mode="per_channel"` (the default) selects `weighted_bce_dice_loss`
        (Experiment 2B); `weighted_dice_mode="pooled"` selects
        `weighted_pooled_bce_dice_loss` (Experiment 2C) instead. Has no
        effect if a model is assigned to `self.model` directly or loaded
        via `load()` -- this only governs how `train()` builds a fresh
        one."""
        self.input_shape = input_shape
        self.class_weights = class_weights
        self.weighted_dice_mode = weighted_dice_mode
        self.model = None

    def train(self, train_data, val_data=None, **kwargs):
        """Trains via `training.Trainer`/`TrainingConfig`, mirroring
        `train_image_quality.py`'s use of the same framework. `train_data`/
        `val_data` must already be built `tf.data.Dataset` pipelines (e.g.
        from `lesion_segmentation_dataset.load_lesion_segmentation_datasets()`)
        -- this method does not build them itself, per
        `pipeline.TrainableStage`'s contract. Accepts an optional
        `trainer_config` (a `training.TrainingConfig`); otherwise builds one
        from `run_dir`/`epochs` kwargs (defaults: `DEFAULT_RUN_DIR`, 50)."""
        if self.model is None:
            self.model = build_attention_unet(
                input_shape=self.input_shape, class_weights=self.class_weights,
                weighted_dice_mode=self.weighted_dice_mode,
            )

        trainer_config = kwargs.get("trainer_config") or TrainingConfig(
            run_dir=kwargs.get("run_dir", DEFAULT_RUN_DIR),
            epochs=kwargs.get("epochs", 50),
            resume=kwargs.get("resume", False),
            save_weights_only=False,  # complete .keras checkpoints, not weights-only
        )
        trainer = Trainer(trainer_config)
        history = trainer.fit(self.model, train_data, val_data)
        self._trainer = trainer
        return history

    def evaluate(self, eval_data, **kwargs):
        """Runs the model over `eval_data` (a `tf.data.Dataset` of
        (input, target) batches -- e.g. from
        `lesion_segmentation_dataset.load_lesion_segmentation_test_dataset()`)
        and returns per-class + mean Dice/IoU via
        `per_class_segmentation_metrics`, per SEGMENTATION_ARCHITECTURE.md
        Sec 5 ("evaluate()'s internals ... use Dice/IoU")."""
        if self.model is None:
            raise RuntimeError("LesionSegmentationStage.load() or .train() must run before evaluate().")

        y_true_batches, y_pred_batches = [], []
        for x_batch, y_batch in eval_data:
            predictions = self.model.predict(x_batch, verbose=0)
            y_true_batches.append(np.asarray(y_batch))
            y_pred_batches.append(predictions)

        y_true = np.concatenate(y_true_batches, axis=0)
        y_pred = np.concatenate(y_pred_batches, axis=0)
        return per_class_segmentation_metrics(y_true, y_pred)

    def save(self, path):
        if self.model is None:
            raise RuntimeError("No trained model to save -- call train() first.")
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        self.model.save(path)
        return path

    def load(self, path=DEFAULT_MODEL_PATH):
        self.model = load_lesion_model(path)
        return self

    def predict(self, input_data):
        if self.model is None:
            raise RuntimeError("LesionSegmentationStage.load() must be called before predict().")
        return predict_lesion_mask(input_data, model=self.model)["probability_maps"]

    def predict_batch(self, inputs):
        if self.model is None:
            raise RuntimeError("LesionSegmentationStage.load() must be called before predict_batch().")
        results = predict_lesion_mask_batch(inputs, model=self.model)
        return [r["probability_maps"] for r in results]
