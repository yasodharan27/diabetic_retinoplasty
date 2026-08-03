"""
Evaluation entry point for the Image Quality Assessment module.

Loads the exported IQA model and runs it over the held-out EyeQ
`raw/test/` split, reporting the standard evaluation suite via the reusable
evaluation framework (evaluation/): accuracy, precision, recall, F1,
confusion matrix, classification report, and ROC/AUC.
"""

import argparse
import json
import os

import numpy as np

from config import EYEQ_RAW_DIR, IQA_RESULTS_DIR
from evaluation import Evaluator
from image_quality_dataset import QUALITY_CLASSES, load_eyeq_test_split
from image_quality_inference import DEFAULT_MODEL_PATH, load_iqa_model


def evaluate(raw_dir=EYEQ_RAW_DIR, model_path=DEFAULT_MODEL_PATH,
             image_size=(224, 224), batch_size=16, output_dir=IQA_RESULTS_DIR):
    os.makedirs(output_dir, exist_ok=True)

    model = load_iqa_model(model_path)
    test_ds, y_true, filenames = load_eyeq_test_split(
        raw_dir=raw_dir, image_size=image_size, batch_size=batch_size,
    )

    y_proba = model.predict(test_ds, verbose=1)
    y_pred = np.argmax(y_proba, axis=-1)

    evaluator = Evaluator(class_names=QUALITY_CLASSES)
    report = evaluator.evaluate_and_visualize(
        y_true, y_pred, y_proba, num_classes=len(QUALITY_CLASSES), output_dir=output_dir,
    )

    print(f"Evaluated {len(filenames)} images from {raw_dir}/test")
    print(f"Accuracy:  {report.accuracy:.4f}")
    print(f"Precision: {report.precision:.4f}")
    print(f"Recall:    {report.recall:.4f}")
    print(f"F1 score:  {report.f1:.4f}")
    print(f"AUC:       {report.auc:.4f}")
    print(f"Quadratic weighted kappa: {report.quadratic_weighted_kappa:.4f}")
    print(f"Confusion matrix:\n{report.confusion_matrix}")
    print("Classification report:")
    print(json.dumps(report.per_class_report, indent=2))
    print(f"Full report (JSON + plots) saved under {output_dir}")

    return report


def main():
    parser = argparse.ArgumentParser(description="Evaluate the Image Quality Assessment model.")
    parser.add_argument("--raw-dir", default=EYEQ_RAW_DIR)
    parser.add_argument("--model", dest="model_path", default=DEFAULT_MODEL_PATH)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--output-dir", default=IQA_RESULTS_DIR)
    args = parser.parse_args()

    evaluate(
        raw_dir=args.raw_dir,
        model_path=args.model_path,
        batch_size=args.batch_size,
        output_dir=args.output_dir,
    )


if __name__ == "__main__":
    main()
