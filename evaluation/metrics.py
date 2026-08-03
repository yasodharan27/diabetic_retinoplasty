"""
Generic, array-based evaluation metrics -- classification and ordinal-grading
focused. Every function takes plain arrays (`y_true`, `y_pred`, and/or
`y_proba`), not a model or a dataset: they operate on predictions a caller
has already produced (e.g. via `model.predict()`), which are supplied later
when this package is wired up to an actual trained model.

`y_true`/`y_pred` may be given as integer class labels or one-hot vectors --
`_to_labels()` normalizes either form. `y_proba` must be per-class
probabilities (shape `(n_samples, n_classes)`, or `(n_samples,)`/
`(n_samples, 1)` for binary positive-class scores).

Segmentation overlap metrics (Dice, IoU) intentionally live in
`training.metrics` instead, since they're computed during training as Keras
metrics on tensors, not post-hoc on numpy prediction arrays.
"""

import numpy as np
from sklearn import metrics as skm


def _to_labels(y):
    """Normalize integer-label or one-hot input to a flat int array of class indices."""
    y = np.asarray(y)
    if y.ndim > 1 and y.shape[-1] > 1:
        return np.argmax(y, axis=-1)
    return y.reshape(-1).astype(int)


def _to_proba(y_proba):
    """Normalize probability input to shape (n_samples, n_classes), expanding
    a single positive-class score column into a 2-class array."""
    y_proba = np.asarray(y_proba, dtype=float)
    if y_proba.ndim == 1:
        y_proba = np.stack([1 - y_proba, y_proba], axis=-1)
    elif y_proba.ndim == 2 and y_proba.shape[-1] == 1:
        y_proba = np.concatenate([1 - y_proba, y_proba], axis=-1)
    return y_proba


# ---------------------------------------------------------------------------
# Core classification metrics
# ---------------------------------------------------------------------------

def accuracy(y_true, y_pred):
    return float(skm.accuracy_score(_to_labels(y_true), _to_labels(y_pred)))


def precision(y_true, y_pred, average="macro", zero_division=0):
    return skm.precision_score(
        _to_labels(y_true), _to_labels(y_pred), average=average, zero_division=zero_division
    )


def recall(y_true, y_pred, average="macro", zero_division=0):
    return skm.recall_score(
        _to_labels(y_true), _to_labels(y_pred), average=average, zero_division=zero_division
    )


def f1_score(y_true, y_pred, average="macro", zero_division=0):
    return skm.f1_score(
        _to_labels(y_true), _to_labels(y_pred), average=average, zero_division=zero_division
    )


def confusion_matrix(y_true, y_pred, num_classes=None, normalize=None):
    """`normalize` is passed through to sklearn: None, 'true', 'pred', or 'all'."""
    labels = list(range(num_classes)) if num_classes else None
    return skm.confusion_matrix(
        _to_labels(y_true), _to_labels(y_pred), labels=labels, normalize=normalize
    )


def per_class_report(y_true, y_pred, class_names=None, zero_division=0):
    """Wraps sklearn's classification_report as a dict: per-class precision/
    recall/f1/support plus macro/weighted averages."""
    return skm.classification_report(
        _to_labels(y_true), _to_labels(y_pred),
        target_names=class_names, output_dict=True, zero_division=zero_division,
    )


# ---------------------------------------------------------------------------
# Sensitivity / Specificity (per class, one-vs-rest, from the confusion matrix)
# ---------------------------------------------------------------------------

def sensitivity_specificity(y_true, y_pred, num_classes=None, class_names=None):
    """Per-class sensitivity (= recall = TP/(TP+FN)) and specificity
    (= TN/(TN+FP)), each computed one-vs-rest from the confusion matrix."""
    y_true_labels = _to_labels(y_true)
    y_pred_labels = _to_labels(y_pred)
    if num_classes is not None:
        labels = list(range(num_classes))
    else:
        labels = sorted(set(y_true_labels.tolist()) | set(y_pred_labels.tolist()))

    cm = skm.confusion_matrix(y_true_labels, y_pred_labels, labels=labels)
    total = cm.sum()

    results = {}
    for i, label in enumerate(labels):
        tp = cm[i, i]
        fn = cm[i, :].sum() - tp
        fp = cm[:, i].sum() - tp
        tn = total - tp - fn - fp
        sens = float(tp / (tp + fn)) if (tp + fn) > 0 else 0.0
        spec = float(tn / (tn + fp)) if (tn + fp) > 0 else 0.0
        key = class_names[i] if class_names else int(label)
        results[key] = {"sensitivity": sens, "specificity": spec}
    return results


# ---------------------------------------------------------------------------
# ROC / AUC
# ---------------------------------------------------------------------------

def roc_curve_data(y_true, y_proba, class_names=None):
    """Per-class (one-vs-rest) ROC curve points + AUC. Returns a dict keyed
    by class name (if given) or class index, each value holding
    fpr/tpr/thresholds/auc -- ready for `evaluation.visualization.plot_roc_curves`."""
    y_true_labels = _to_labels(y_true)
    y_proba = _to_proba(y_proba)
    n_classes = y_proba.shape[-1]

    curves = {}
    for i in range(n_classes):
        y_true_binary = (y_true_labels == i).astype(int)
        fpr, tpr, thresholds = skm.roc_curve(y_true_binary, y_proba[:, i])
        roc_auc = skm.auc(fpr, tpr)
        key = class_names[i] if class_names else i
        curves[key] = {"fpr": fpr, "tpr": tpr, "thresholds": thresholds, "auc": float(roc_auc)}
    return curves


def auc_score(y_true, y_proba, average="macro"):
    """Overall AUC: binary AUC for 2-class probabilities, macro/weighted
    one-vs-rest AUC for multiclass."""
    y_true_labels = _to_labels(y_true)
    y_proba = _to_proba(y_proba)
    n_classes = y_proba.shape[-1]
    if n_classes == 2:
        return float(skm.roc_auc_score(y_true_labels, y_proba[:, 1]))
    return float(skm.roc_auc_score(y_true_labels, y_proba, multi_class="ovr", average=average))


# ---------------------------------------------------------------------------
# Quadratic Weighted Kappa
# ---------------------------------------------------------------------------

def quadratic_weighted_kappa(y_true, y_pred):
    """Standard metric for ordinal DR-severity grading."""
    return float(skm.cohen_kappa_score(_to_labels(y_true), _to_labels(y_pred), weights="quadratic"))


# ---------------------------------------------------------------------------
# Calibration
# ---------------------------------------------------------------------------

def calibration_curve_data(y_true, y_proba, n_bins=10):
    """Top-1-confidence calibration data: for each of `n_bins` confidence
    buckets, the mean predicted confidence and empirical accuracy within
    that bucket. Generalizes to any number of classes (unlike
    `sklearn.calibration.calibration_curve`, which is binary-only), and --
    unlike the reliability diagram currently in `bayesian_inference.py` --
    uses real labels rather than simulated ground truth."""
    y_true_labels = _to_labels(y_true)
    y_proba = _to_proba(y_proba)

    confidences = y_proba.max(axis=-1)
    predictions = y_proba.argmax(axis=-1)
    correct = (predictions == y_true_labels).astype(float)

    bin_edges = np.linspace(0.0, 1.0, n_bins + 1)
    bin_indices = np.clip(np.digitize(confidences, bin_edges[1:-1], right=True), 0, n_bins - 1)

    bin_confidences = np.zeros(n_bins)
    bin_accuracies = np.zeros(n_bins)
    bin_counts = np.zeros(n_bins, dtype=int)
    for b in range(n_bins):
        mask = bin_indices == b
        bin_counts[b] = mask.sum()
        if bin_counts[b] > 0:
            bin_confidences[b] = confidences[mask].mean()
            bin_accuracies[b] = correct[mask].mean()

    return {
        "bin_edges": bin_edges,
        "bin_confidences": bin_confidences,
        "bin_accuracies": bin_accuracies,
        "bin_counts": bin_counts,
    }


def expected_calibration_error(y_true, y_proba, n_bins=10):
    """Expected Calibration Error (ECE): the bin-count-weighted average gap
    between confidence and accuracy across bins (Guo et al., 2017)."""
    data = calibration_curve_data(y_true, y_proba, n_bins=n_bins)
    total = data["bin_counts"].sum()
    if total == 0:
        return 0.0
    weights = data["bin_counts"] / total
    gaps = np.abs(data["bin_confidences"] - data["bin_accuracies"])
    return float(np.sum(weights * gaps))
