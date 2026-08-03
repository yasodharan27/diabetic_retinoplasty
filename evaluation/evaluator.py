"""
Model-agnostic evaluation orchestrator. `Evaluator` composes the functions
in `evaluation.metrics` into a single report, optionally rendering the
`evaluation.visualization` plots alongside it.

Operates purely on already-computed predictions (`y_true`, `y_pred`, and an
optional `y_proba`) -- it does not call a model, run inference, or load a
dataset. Wiring an `Evaluator` up to an actual trained model's
`.predict()` output is left to each module's future evaluation script:

    from evaluation import Evaluator

    evaluator = Evaluator(class_names=["No DR", "Mild", "Moderate", "Severe", "Proliferative"])
    report = evaluator.evaluate_and_visualize(y_true, y_pred, y_proba, output_dir="results/final_classification")
"""

import json
import os
from dataclasses import dataclass
from typing import Optional

import numpy as np

from . import metrics as em
from . import visualization as ev


@dataclass
class EvaluationReport:
    accuracy: float
    precision: float
    recall: float
    f1: float
    confusion_matrix: np.ndarray
    sensitivity_specificity: dict
    per_class_report: dict
    quadratic_weighted_kappa: Optional[float] = None
    auc: Optional[float] = None
    roc_curves: Optional[dict] = None
    calibration: Optional[dict] = None
    expected_calibration_error: Optional[float] = None

    def to_dict(self):
        d = {
            "accuracy": self.accuracy,
            "precision": self.precision,
            "recall": self.recall,
            "f1": self.f1,
            "confusion_matrix": self.confusion_matrix.tolist(),
            "sensitivity_specificity": self.sensitivity_specificity,
            "per_class_report": self.per_class_report,
            "quadratic_weighted_kappa": self.quadratic_weighted_kappa,
            "auc": self.auc,
            "expected_calibration_error": self.expected_calibration_error,
        }
        if self.roc_curves is not None:
            d["roc_curves"] = {
                str(key): {k: (v.tolist() if isinstance(v, np.ndarray) else v) for k, v in curve.items()}
                for key, curve in self.roc_curves.items()
            }
        if self.calibration is not None:
            d["calibration"] = {
                k: (v.tolist() if isinstance(v, np.ndarray) else v) for k, v in self.calibration.items()
            }
        return d

    def save_json(self, path):
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        with open(path, "w") as f:
            json.dump(self.to_dict(), f, indent=2)
        return path


class Evaluator:
    """Computes the standard classification/ordinal evaluation report from
    predictions alone. Reusable as-is by Image Quality Assessment and Final
    Classification (or any other classification-style module); segmentation
    modules should instead use the Dice/IoU metrics in `training.metrics`."""

    def __init__(self, class_names=None, average="macro"):
        self.class_names = class_names
        self.average = average

    def evaluate(self, y_true, y_pred, y_proba=None, num_classes=None) -> EvaluationReport:
        cm = em.confusion_matrix(y_true, y_pred, num_classes=num_classes)

        report = EvaluationReport(
            accuracy=em.accuracy(y_true, y_pred),
            precision=em.precision(y_true, y_pred, average=self.average),
            recall=em.recall(y_true, y_pred, average=self.average),
            f1=em.f1_score(y_true, y_pred, average=self.average),
            confusion_matrix=cm,
            sensitivity_specificity=em.sensitivity_specificity(
                y_true, y_pred, num_classes=num_classes, class_names=self.class_names
            ),
            per_class_report=em.per_class_report(y_true, y_pred, class_names=self.class_names),
        )

        try:
            report.quadratic_weighted_kappa = em.quadratic_weighted_kappa(y_true, y_pred)
        except ValueError:
            report.quadratic_weighted_kappa = None

        if y_proba is not None:
            report.auc = em.auc_score(y_true, y_proba)
            report.roc_curves = em.roc_curve_data(y_true, y_proba, class_names=self.class_names)
            report.calibration = em.calibration_curve_data(y_true, y_proba)
            report.expected_calibration_error = em.expected_calibration_error(y_true, y_proba)

        return report

    def evaluate_and_visualize(self, y_true, y_pred, y_proba=None, num_classes=None, output_dir=None):
        """Same as `evaluate()`, plus renders (and, if `output_dir` is given,
        saves) the confusion matrix, ROC curves, and calibration plots."""
        report = self.evaluate(y_true, y_pred, y_proba=y_proba, num_classes=num_classes)

        cm_path = os.path.join(output_dir, "confusion_matrix.png") if output_dir else None
        ev.plot_confusion_matrix(report.confusion_matrix, class_names=self.class_names, output_path=cm_path)

        if report.roc_curves is not None:
            roc_path = os.path.join(output_dir, "roc_curves.png") if output_dir else None
            ev.plot_roc_curves(report.roc_curves, output_path=roc_path)

        if report.calibration is not None:
            cal_path = os.path.join(output_dir, "calibration_curve.png") if output_dir else None
            ev.plot_calibration_curve(report.calibration, output_path=cal_path)

        if output_dir:
            report.save_json(os.path.join(output_dir, "evaluation_report.json"))

        return report
