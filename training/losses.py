"""
Generic, architecture-independent loss functions reusable across modules
(Image Quality Assessment, Vessel Segmentation, Lesion Segmentation, Final
Classification). Each loss only assumes a `(y_true, y_pred)` tensor pair in
the shape Keras already expects from `model.fit()` -- nothing here is tied
to a specific model architecture.

Module-specific losses that ARE tied to a particular architecture's output
format (e.g. a CORN ordinal-regression loss for the eventual CORN
classification head) are intentionally NOT included here -- add those
alongside that module's model definition when it is implemented and approved.
"""

import tensorflow as tf


def focal_loss(gamma=2.0, alpha=0.25):
    """Focal loss for class-imbalanced classification (Lin et al., 2017).
    Expects one-hot `y_true` and softmax `y_pred`, matching the existing
    usage in train_hybrid_model.py / retrain_efficientnet.py."""

    def _focal_loss(y_true, y_pred):
        epsilon = tf.keras.backend.epsilon()
        y_pred = tf.clip_by_value(y_pred, epsilon, 1 - epsilon)
        y_true = tf.cast(y_true, y_pred.dtype)
        cross_entropy = -y_true * tf.math.log(y_pred)
        focal_weight = tf.pow(1 - y_pred, gamma) * y_true
        if alpha is not None:
            focal_weight = alpha * focal_weight
        loss = focal_weight * cross_entropy
        return tf.reduce_sum(loss, axis=-1)

    return _focal_loss


def dice_loss(smooth=1.0):
    """Soft Dice loss for binary/probability segmentation masks."""

    def _dice_loss(y_true, y_pred):
        y_true = tf.cast(y_true, y_pred.dtype)
        y_true_f = tf.reshape(y_true, [tf.shape(y_true)[0], -1])
        y_pred_f = tf.reshape(y_pred, [tf.shape(y_pred)[0], -1])
        intersection = tf.reduce_sum(y_true_f * y_pred_f, axis=-1)
        union = tf.reduce_sum(y_true_f, axis=-1) + tf.reduce_sum(y_pred_f, axis=-1)
        dice = (2.0 * intersection + smooth) / (union + smooth)
        return 1.0 - dice

    return _dice_loss


def bce_dice_loss(dice_weight=0.5, smooth=1.0):
    """Weighted combination of binary cross-entropy and Dice loss -- a common,
    generic default for segmentation tasks (vessel/lesion masks)."""
    _dice = dice_loss(smooth=smooth)
    bce = tf.keras.losses.BinaryCrossentropy()

    def _combined(y_true, y_pred):
        return (1 - dice_weight) * bce(y_true, y_pred) + dice_weight * _dice(y_true, y_pred)

    return _combined


def weighted_categorical_crossentropy(class_weights):
    """Categorical cross-entropy with explicit per-class weights, as a
    config-driven alternative to Keras' `class_weight=` fit() argument
    (useful when a loss object is required directly, e.g. inside a custom
    training loop rather than `model.fit()`)."""
    weights = tf.constant(class_weights, dtype=tf.float32)

    def _loss(y_true, y_pred):
        epsilon = tf.keras.backend.epsilon()
        y_pred = tf.clip_by_value(y_pred, epsilon, 1 - epsilon)
        y_true = tf.cast(y_true, y_pred.dtype)
        cross_entropy = -y_true * tf.math.log(y_pred)
        weighted = cross_entropy * tf.cast(weights, y_pred.dtype)
        return tf.reduce_sum(weighted, axis=-1)

    return _loss


_LOSS_REGISTRY = {
    "focal": focal_loss,
    "dice": dice_loss,
    "bce_dice": bce_dice_loss,
    "categorical_crossentropy": lambda **kwargs: tf.keras.losses.CategoricalCrossentropy(**kwargs),
    "binary_crossentropy": lambda **kwargs: tf.keras.losses.BinaryCrossentropy(**kwargs),
}


def get_loss(name, **kwargs):
    """Factory: resolve a loss by name for config-driven training scripts."""
    key = name.lower()
    if key not in _LOSS_REGISTRY:
        raise ValueError(f"Unknown loss '{name}'. Available: {sorted(_LOSS_REGISTRY)}")
    return _LOSS_REGISTRY[key](**kwargs)
