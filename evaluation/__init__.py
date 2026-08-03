"""
Reusable, model-agnostic evaluation framework.

Operates purely on prediction arrays (`y_true`, `y_pred`, optional
`y_proba`) -- no model, dataset, or training coupling. Reusable by Image
Quality Assessment, Vessel Segmentation, Lesion Segmentation, and Final
Classification alike, once each module produces predictions to evaluate.
"""

from .evaluator import Evaluator, EvaluationReport
from .metrics import (
    accuracy,
    precision,
    recall,
    f1_score,
    confusion_matrix,
    per_class_report,
    sensitivity_specificity,
    roc_curve_data,
    auc_score,
    quadratic_weighted_kappa,
    calibration_curve_data,
    expected_calibration_error,
)
from .visualization import (
    plot_confusion_matrix,
    plot_roc_curves,
    plot_calibration_curve,
    plot_metric_bar,
)

__all__ = [
    "Evaluator", "EvaluationReport",
    "accuracy", "precision", "recall", "f1_score", "confusion_matrix", "per_class_report",
    "sensitivity_specificity", "roc_curve_data", "auc_score", "quadratic_weighted_kappa",
    "calibration_curve_data", "expected_calibration_error",
    "plot_confusion_matrix", "plot_roc_curves", "plot_calibration_curve", "plot_metric_bar",
]
