"""
Plotting helpers built on top of `evaluation.metrics`'s outputs. Every
function accepts pre-computed data (confusion matrix array, ROC-curve dict,
calibration-curve dict, etc.) and returns the created Matplotlib Figure;
pass `output_path` to also save it to disk. No model or dataset coupling --
purely a rendering layer over arrays/dicts already produced elsewhere.
"""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


def plot_confusion_matrix(cm, class_names=None, normalize=False, ax=None,
                           output_path=None, title="Confusion Matrix"):
    cm = np.asarray(cm, dtype=float)
    if normalize:
        row_sums = cm.sum(axis=1, keepdims=True)
        cm = np.divide(cm, row_sums, out=np.zeros_like(cm), where=row_sums != 0)

    fig, ax = (ax.figure, ax) if ax is not None else plt.subplots(figsize=(8, 6))
    labels = class_names if class_names else list(range(cm.shape[0]))
    sns.heatmap(
        cm, annot=True, fmt=".2f" if normalize else ".0f", cmap="Blues",
        xticklabels=labels, yticklabels=labels, ax=ax,
    )
    ax.set_xlabel("Predicted Label")
    ax.set_ylabel("True Label")
    ax.set_title(title)

    if output_path:
        fig.savefig(output_path, bbox_inches="tight")
    return fig


def plot_roc_curves(roc_data, ax=None, output_path=None, title="ROC Curves"):
    """`roc_data`: output of `evaluation.metrics.roc_curve_data()`."""
    fig, ax = (ax.figure, ax) if ax is not None else plt.subplots(figsize=(8, 6))

    for label, curve in roc_data.items():
        ax.plot(curve["fpr"], curve["tpr"], lw=2, label=f"{label} (AUC = {curve['auc']:.3f})")
    ax.plot([0, 1], [0, 1], "k--", lw=1)
    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.05])
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title(title)
    ax.legend(loc="lower right")
    ax.grid(True, alpha=0.3)

    if output_path:
        fig.savefig(output_path, bbox_inches="tight")
    return fig


def plot_calibration_curve(calibration_data, ax=None, output_path=None, title="Calibration Curve"):
    """`calibration_data`: output of `evaluation.metrics.calibration_curve_data()`."""
    fig, ax = (ax.figure, ax) if ax is not None else plt.subplots(figsize=(7, 7))

    mask = calibration_data["bin_counts"] > 0
    ax.plot([0, 1], [0, 1], "k--", label="Perfectly calibrated")
    ax.plot(
        calibration_data["bin_confidences"][mask],
        calibration_data["bin_accuracies"][mask],
        "o-", label="Model",
    )
    ax.set_xlabel("Confidence")
    ax.set_ylabel("Accuracy")
    ax.set_title(title)
    ax.legend()
    ax.grid(True, alpha=0.3)

    if output_path:
        fig.savefig(output_path, bbox_inches="tight")
    return fig


def plot_metric_bar(values, class_names=None, ylabel="Score",
                     title="Per-Class Metric", ax=None, output_path=None):
    """Generic bar chart for any per-class metric dict (e.g. from
    `sensitivity_specificity()`) or plain array (e.g. per-class F1)."""
    if isinstance(values, dict):
        labels = list(values.keys())
        scores = list(values.values())
    else:
        scores = list(values)
        labels = class_names if class_names else list(range(len(scores)))

    fig, ax = (ax.figure, ax) if ax is not None else plt.subplots(figsize=(8, 5))
    ax.bar([str(label) for label in labels], scores, color="steelblue", alpha=0.85)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.grid(axis="y", alpha=0.3)

    if output_path:
        fig.savefig(output_path, bbox_inches="tight")
    return fig
