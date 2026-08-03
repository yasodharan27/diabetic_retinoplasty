"""
EfficientNet-B0 Image Quality Assessment model (pipeline Step 1).

Reuses the same ImageNet-pretrained transfer-learning pattern already
established in efficientnet_model.py (partially frozen backbone, GAP, dense
head) applied to a different task and label space: quality tier
(Good / Usable / Reject) instead of DR severity. Kept as its own module
rather than parameterizing efficientnet_model.py, since the two solve
unrelated classification problems with different class counts and inputs
(RGB here vs. grayscale-tiled-to-3-channel there).
"""

import tensorflow as tf
from keras import layers, Model, Input
from keras.applications import EfficientNetB0

from image_quality_dataset import NUM_CLASSES
from training import build_metrics


def build_iqa_model(input_shape=(224, 224, 3), num_classes=NUM_CLASSES,
                     learning_rate=1e-4, freeze_layers=100):
    """Build and compile the EfficientNet-B0 IQA classifier. `freeze_layers`
    mirrors efficientnet_model.py's default (first 100 backbone layers
    frozen, the rest fine-tuned)."""
    inputs = Input(shape=input_shape)
    base_model = EfficientNetB0(include_top=False, weights="imagenet", input_shape=input_shape)
    for layer in base_model.layers[:freeze_layers]:
        layer.trainable = False

    x = base_model(inputs)
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dense(256, activation="relu")(x)
    x = layers.Dropout(0.3)(x)
    # Explicit float32 output: keeps the softmax numerically stable when a
    # mixed_float16 policy is active (see training.trainer.enable_mixed_precision).
    outputs = layers.Dense(num_classes, activation="softmax", dtype="float32")(x)

    model = Model(inputs=inputs, outputs=outputs, name="iqa_efficientnet_b0")
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
        loss="categorical_crossentropy",
        metrics=build_metrics("classification"),
    )
    return model
