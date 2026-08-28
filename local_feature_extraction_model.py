"""
Adaptive Multi-Kernel CNN model + inference for Local Feature Extraction
(pipeline Stage 05).

Consumes the 8-channel, segmentation-aware tensor
`local_feature_extraction_dataset.py` builds for each image (Stage 02
processed RGB + Stage 03 vessel probability map + Stage 04 four-class
lesion probability maps, `(512, 512, 8)`) and produces a spatially
resolved local feature map, `(32, 32, 256)` -- never a globally pooled
vector, and never with a classification head of any kind attached. This
is a deliberate scope boundary: Stage 05's only documented consumer is
Stage 07 (Adaptive Cross-Attention), which is not implemented here.

Unlike `lesion_segmentation_model.py`'s `build_attention_unet`, the model
built here is **not compiled** with a loss/optimizer/metrics -- Stage 05
has no standalone ground truth of its own (no "local feature quality"
label exists anywhere in this project). Per the approved design, it will
eventually receive its only learning signal by participating in a future,
not-yet-implemented joint Stage 05-08 + RACAF training graph, through the
downstream CORN ordinal loss. Compiling it here with an invented loss
would misrepresent that.

Architecture: four "multi-kernel blocks" (parallel 3x3 / 5x5 / dilated-3x3
convolution branches, concatenated and fused via a 1x1 conv), each
followed by a 2x2 max-pool -- a direct, unmodified reuse of this
project's existing conv-block idiom (`lesion_segmentation_model.py`'s
`_conv_block`'s BatchNorm+ReLU convention), extended with parallel
branches instead of a single kernel size per stage, per the approved
Stage 05 design's explicit multi-scale requirement. No attention gates,
no decoder, no skip connections -- those are Stage 04's own architecture,
not reused or duplicated here.
"""

import os

import numpy as np
import tensorflow as tf
from keras import Input, Model, layers
from keras.saving import register_keras_serializable

import config
from pipeline import FeatureExtractionStage

DEFAULT_INPUT_SHAPE = (512, 512, 8)
DEFAULT_STAGE_FILTERS = (32, 64, 128, 256)
DEFAULT_MODEL_PATH = os.path.join(config.LOCAL_FEATURE_MODEL_DIR, "best_model.keras")

# Fixed by the approved Stage 05 design: 4 downsampling stages applied to a
# 512x512 input yields exactly this spatial resolution (512 / 2**4 = 32),
# and the last stage's filter count fixes the channel dimension -- neither
# is a free parameter of this module.
OUTPUT_SPATIAL_SIZE = DEFAULT_INPUT_SHAPE[0] // (2 ** len(DEFAULT_STAGE_FILTERS))
OUTPUT_CHANNELS = DEFAULT_STAGE_FILTERS[-1]


@register_keras_serializable(package="local_feature_extraction")
class _StopGradientBoundary(layers.Layer):
    """Explicit stop-gradient boundary layer. A plain `Lambda(tf.stop_gradient)`
    cannot round-trip through `model.save()`/`load_model()` in Keras 3 (a
    bare function reference is not deserializable) -- a registered custom
    `Layer` subclass is the supported way to fold an arbitrary TF op into a
    Functional model's graph while keeping it fully save/load-safe."""

    def call(self, inputs):
        return tf.stop_gradient(inputs)


def _multi_kernel_block(x, filters, name):
    """Three parallel convolutional branches at different effective
    receptive fields -- a plain 3x3 (small), a plain 5x5 (medium), and a
    dilated 3x3 with dilation_rate=3 (large, effective receptive field 7x7
    without the parameter cost of a literal 7x7 kernel) -- concatenated and
    fused back down to `filters` channels via a 1x1 conv. This is what
    makes the block "multi-kernel": every spatial position's output is
    informed by three different receptive-field sizes at once, rather than
    committing to a single one per stage (the approved design's explicit
    rationale -- lesion classes span very different intrinsic scales, from
    Microaneurysm's few-pixel footprint to Hard/Soft Exudate's broader
    patches)."""
    small = layers.Conv2D(filters, 3, padding="same", use_bias=False, name=f"{name}_k3_conv")(x)
    small = layers.BatchNormalization(name=f"{name}_k3_bn")(small)
    small = layers.Activation("relu", name=f"{name}_k3_relu")(small)

    medium = layers.Conv2D(filters, 5, padding="same", use_bias=False, name=f"{name}_k5_conv")(x)
    medium = layers.BatchNormalization(name=f"{name}_k5_bn")(medium)
    medium = layers.Activation("relu", name=f"{name}_k5_relu")(medium)

    large = layers.Conv2D(filters, 3, padding="same", dilation_rate=3, use_bias=False,
                           name=f"{name}_dilated_conv")(x)
    large = layers.BatchNormalization(name=f"{name}_dilated_bn")(large)
    large = layers.Activation("relu", name=f"{name}_dilated_relu")(large)

    fused = layers.concatenate([small, medium, large], name=f"{name}_concat")
    fused = layers.Conv2D(filters, 1, padding="same", use_bias=False, name=f"{name}_fuse_conv")(fused)
    fused = layers.BatchNormalization(name=f"{name}_fuse_bn")(fused)
    fused = layers.Activation("relu", name=f"{name}_fuse_relu")(fused)
    return fused


def build_local_feature_extractor(input_shape=DEFAULT_INPUT_SHAPE, stage_filters=DEFAULT_STAGE_FILTERS):
    """
    Builds the Adaptive Multi-Kernel CNN: `len(stage_filters)` multi-kernel
    blocks, each followed by a 2x2 max-pool, random-initialized (no
    ImageNet weights -- see this module's docstring and the approved design's
    Sec 4 for why no pretrained backbone applies to this 8-channel,
    segmentation-derived input). Returns an **uncompiled** `keras.Model` --
    see this module's docstring for why no loss is attached here.

    `input_shape`'s height/width must each be divisible by
    `2 ** len(stage_filters)` for the downsampling stack to round-trip
    exactly, mirroring `build_attention_unet`'s identical divisibility
    check for its own pooling depth.

    The Stage 03/04 outputs concatenated into this model's input (by
    `local_feature_extraction_dataset.py`) are frozen, inference-only
    dependencies -- the input tensor is wrapped in `tf.stop_gradient`
    immediately after `Input(...)` as an explicit, self-documenting
    boundary marker. In the current dataset pipeline this is defense in
    depth rather than the only safeguard: Stage 03 (PyTorch, run under
    `torch.no_grad()`) and Stage 04 (Keras, run via `.predict()`, never
    inside a `GradientTape`) already produce plain NumPy arrays with no
    gradient history before they ever reach this model's `Input` tensor --
    but once Stage 05 is wired into a single end-to-end differentiable
    graph (a future step, not implemented here), this `stop_gradient` is
    what keeps that guarantee explicit and enforced at the tensor
    boundary itself, exactly as the approved design requires.
    """
    height, width = input_shape[0], input_shape[1]
    depth = len(stage_filters)
    divisor = 2 ** depth
    if height % divisor != 0 or width % divisor != 0:
        raise ValueError(
            f"input_shape {input_shape}: height and width must each be divisible by "
            f"{divisor} ({depth} pooling stages)."
        )

    inputs = Input(shape=input_shape, name="local_feature_input")
    x = _StopGradientBoundary(name="stop_gradient_boundary")(inputs)

    for i, filters in enumerate(stage_filters, start=1):
        x = _multi_kernel_block(x, filters, name=f"stage{i}")
        x = layers.MaxPooling2D(2, name=f"stage{i}_pool")(x)

    # Explicit float32 output: keeps this feature map numerically stable if
    # this model is ever run under a mixed_float16 policy during the future
    # joint Stage 05-08 training graph, matching lesion_segmentation_model.py's
    # / image_quality_model.py's identical convention for their own final layers.
    outputs = layers.Activation("linear", dtype="float32", name="local_features")(x)

    return Model(inputs=inputs, outputs=outputs, name="local_feature_extraction_adaptive_multi_kernel_cnn")


# --- pipeline.FeatureExtractionStage implementation ---

class LocalFeatureExtractionStage(FeatureExtractionStage):
    """`pipeline.FeatureExtractionStage` implementation for the Adaptive
    Multi-Kernel CNN. `predict`/`predict_batch`/`save`/`load` are fully
    functional; `train`/`evaluate` raise `NotImplementedError` -- Stage 05
    has no standalone training or evaluation procedure of its own (see this
    module's docstring), mirroring `VesselSegmentationStage.train()`'s exact
    pattern for a stage whose real training procedure lives outside this
    class entirely (there, a vendored pretrained checkpoint; here, a future
    joint Stage 05-08 + RACAF training script)."""

    def __init__(self, input_shape=DEFAULT_INPUT_SHAPE, stage_filters=DEFAULT_STAGE_FILTERS):
        self.input_shape = input_shape
        self.stage_filters = stage_filters
        self.model = None

    def train(self, train_data, val_data=None, **kwargs):
        raise NotImplementedError(
            "Local Feature Extraction has no standalone training procedure in this project -- "
            "it has no local-feature ground truth of its own and is only ever trained jointly "
            "with Stages 06-08 and RACAF, through the downstream CORN ordinal loss (see the "
            "approved Stage 05 design's Sec 7 and RACAF_ARCHITECTURE.md Sec 7). That joint "
            "training script does not exist yet."
        )

    def evaluate(self, eval_data, **kwargs):
        raise NotImplementedError(
            "Local Feature Extraction has no standalone evaluation metric in this project -- "
            "its usefulness is only observable through the joint downstream pipeline's "
            "classification metrics (see the approved Stage 05 design's Sec 12), not "
            "reproduced here."
        )

    def save(self, path):
        if self.model is None:
            raise RuntimeError("No model to save -- call build() (or assign self.model) first.")
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        self.model.save(path)
        return path

    def load(self, path=DEFAULT_MODEL_PATH):
        self.model = tf.keras.models.load_model(path, compile=False)
        return self

    def build(self):
        """Builds a fresh, uncompiled model from `self.input_shape`/
        `self.stage_filters` and assigns it to `self.model`. Separate from
        `__init__` (mirroring `LesionSegmentationStage`'s lazy-build-on-
        first-use pattern) so a stage instance can be constructed cheaply
        before deciding whether to build, load, or receive a model."""
        self.model = build_local_feature_extractor(
            input_shape=self.input_shape, stage_filters=self.stage_filters,
        )
        return self.model

    def predict(self, input_data):
        if self.model is None:
            raise RuntimeError(
                "LocalFeatureExtractionStage.build() or .load() must be called before predict()."
            )
        batch = np.asarray(input_data, dtype="float32")
        if batch.ndim == len(self.input_shape):
            batch = batch[None, ...]
        return self.model.predict(batch, verbose=0)[0]

    def predict_batch(self, inputs):
        if self.model is None:
            raise RuntimeError(
                "LocalFeatureExtractionStage.build() or .load() must be called before predict_batch()."
            )
        batch = np.stack([np.asarray(x, dtype="float32") for x in inputs], axis=0)
        predictions = self.model.predict(batch, verbose=0)
        return [predictions[i] for i in range(predictions.shape[0])]
