"""
Generic, architecture-independent metrics reusable across modules. Includes
standard segmentation overlap metrics (Dice, IoU) and a quadratic weighted
kappa (QWK) implementation for ordinal grading tasks -- the metric
IMPLEMENTATION_PLAN.md flagged as missing anywhere in the current codebase,
despite being the standard metric for DR severity grading. Nothing here
assumes a specific model architecture.
"""

import tensorflow as tf


def dice_coefficient(y_true, y_pred, smooth=1.0):
    y_true = tf.cast(y_true, y_pred.dtype)
    y_true_f = tf.reshape(y_true, [-1])
    y_pred_f = tf.reshape(y_pred, [-1])
    intersection = tf.reduce_sum(y_true_f * y_pred_f)
    return (2.0 * intersection + smooth) / (
        tf.reduce_sum(y_true_f) + tf.reduce_sum(y_pred_f) + smooth
    )


def iou_score(y_true, y_pred, smooth=1.0):
    y_true = tf.cast(y_true, y_pred.dtype)
    y_true_f = tf.reshape(y_true, [-1])
    y_pred_f = tf.reshape(y_pred, [-1])
    intersection = tf.reduce_sum(y_true_f * y_pred_f)
    union = tf.reduce_sum(y_true_f) + tf.reduce_sum(y_pred_f) - intersection
    return (intersection + smooth) / (union + smooth)


class QuadraticWeightedKappa(tf.keras.metrics.Metric):
    """Stateful Keras metric wrapping quadratic weighted kappa, the standard
    metric for ordinal DR-severity grading. Accumulates a confusion matrix
    across batches/epoch (pure TensorFlow ops throughout, so it works under
    `model.fit()`'s default graph-mode execution, not just eagerly) and
    computes kappa from it in `result()`.

    Expects integer class labels for `y_true`; `y_pred` may be either
    per-class probabilities (argmax is taken) or already-integer labels.
    """

    def __init__(self, num_classes, name="quadratic_weighted_kappa", **kwargs):
        super().__init__(name=name, **kwargs)
        self.num_classes = num_classes
        self.confusion = self.add_weight(
            name="confusion", shape=(num_classes, num_classes),
            initializer="zeros", dtype=tf.float32,
        )
        indices = tf.range(num_classes, dtype=tf.float32)
        diff = tf.expand_dims(indices, 1) - tf.expand_dims(indices, 0)
        self.weight_matrix = (diff ** 2) / tf.cast((num_classes - 1) ** 2, tf.float32)

    @staticmethod
    def _flatten_to_labels(y):
        """Accepts either one-hot/probability vectors (rank > 1, last dim > 1)
        or already-integer labels (any other shape) and returns a flat int32
        label tensor. Handles y_true and y_pred identically since both can
        legitimately arrive one-hot (e.g. alongside categorical_crossentropy)."""
        y = tf.convert_to_tensor(y)
        if y.shape.rank is not None and y.shape.rank > 1 and y.shape[-1] not in (None, 1):
            y = tf.argmax(y, axis=-1, output_type=tf.int32)
        return tf.reshape(tf.cast(y, tf.int32), [-1])

    def update_state(self, y_true, y_pred, sample_weight=None):
        y_true = self._flatten_to_labels(y_true)
        y_pred = self._flatten_to_labels(y_pred)
        batch_confusion = tf.math.confusion_matrix(
            y_true, y_pred, num_classes=self.num_classes, dtype=tf.float32
        )
        self.confusion.assign_add(batch_confusion)

    def result(self):
        actual_hist = tf.reduce_sum(self.confusion, axis=1)
        pred_hist = tf.reduce_sum(self.confusion, axis=0)
        total = tf.reduce_sum(self.confusion)

        expected = tf.tensordot(actual_hist, pred_hist, axes=0) / tf.maximum(total, 1.0)
        numerator = tf.reduce_sum(self.weight_matrix * self.confusion)
        denominator = tf.reduce_sum(self.weight_matrix * expected)

        kappa = 1.0 - numerator / tf.maximum(denominator, tf.keras.backend.epsilon())
        return tf.where(total > 0, kappa, tf.constant(0.0))

    def reset_state(self):
        self.confusion.assign(tf.zeros((self.num_classes, self.num_classes)))


def build_metrics(task_type, num_classes=None):
    """
    Factory returning a reasonable default metric list for a task type:
      - "classification": accuracy + AUC
      - "ordinal": accuracy + QuadraticWeightedKappa(num_classes)
      - "segmentation": dice_coefficient + iou_score
    """
    task_type = task_type.lower()
    if task_type == "classification":
        return ["accuracy", tf.keras.metrics.AUC(name="auc")]
    if task_type == "ordinal":
        if num_classes is None:
            raise ValueError("num_classes is required for 'ordinal' metrics")
        return ["accuracy", QuadraticWeightedKappa(num_classes)]
    if task_type == "segmentation":
        return [dice_coefficient, iou_score]
    raise ValueError(f"Unknown task_type '{task_type}'")
