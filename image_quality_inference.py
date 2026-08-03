"""
Inference for the Image Quality Assessment module.

Loads the exported IQA model (a complete `.keras` model -- see
train_image_quality.py / notebooks/01_image_quality_assessment.ipynb) and
exposes reusable prediction functions -- single-image (`predict_quality`)
and folder-batch (`predict_quality_batch`, `generate_iqa_csv`) -- so later
pipeline stages (the Step 2 preprocessing gate, and eventually the
end-to-end pipeline entry point described in PROJECT_CODE.md's Deployment
Requirement) can call them without re-deriving model-loading logic. No
module should exist only for training; this is this module's inference half.
"""

import argparse
import os

import numpy as np
import pandas as pd
import tensorflow as tf

from config import IQA_MODEL_DIR, IQA_RESULTS_DIR
from image_quality_dataset import QUALITY_CLASSES, _decode_image

DEFAULT_MODEL_PATH = os.path.join(IQA_MODEL_DIR, "best_model.keras")
DEFAULT_CSV_PATH = os.path.join(IQA_RESULTS_DIR, "EyePACS_IQA.csv")
IMAGE_EXTENSIONS = (".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff")


def load_iqa_model(model_path=DEFAULT_MODEL_PATH):
    """Load the complete trained IQA model (architecture + weights) from a
    `.keras` file -- no separate architecture-rebuilding step needed."""
    if not os.path.exists(model_path):
        raise FileNotFoundError(
            f"No IQA model found at {model_path}. Train the model first "
            "(train_image_quality.py or notebooks/01_image_quality_assessment.ipynb) "
            "and export the best checkpoint to this location."
        )
    return tf.keras.models.load_model(model_path)


def preprocess_image(image_path, image_size=(224, 224)):
    """Loads and resizes a single RGB image -- no quality-gate-specific
    preprocessing beyond what the model itself needs, matching the raw-RGB
    training pipeline in image_quality_dataset.py."""
    image = _decode_image(image_path, image_size)
    return tf.expand_dims(image, axis=0)


def _result_from_probabilities(probabilities):
    class_index = int(np.argmax(probabilities))
    return {
        "label": QUALITY_CLASSES[class_index],
        "class_index": class_index,
        "confidence": float(probabilities[class_index]),
        "probabilities": {name: float(p) for name, p in zip(QUALITY_CLASSES, probabilities)},
    }


def predict_quality(image_path, model=None, model_path=DEFAULT_MODEL_PATH,
                     image_size=(224, 224)):
    """
    Predict the quality tier of a single retinal fundus image.

    Returns {"label": "Good"|"Usable"|"Reject", "class_index": int,
    "confidence": float, "probabilities": {class_name: float}}. Pass an
    already-loaded `model` to reuse it across many calls (e.g. inside the
    future end-to-end pipeline); otherwise it is loaded from `model_path`
    on every call.
    """
    if model is None:
        model = load_iqa_model(model_path)

    batch = preprocess_image(image_path, image_size=image_size)
    probabilities = model.predict(batch, verbose=0)[0]
    return _result_from_probabilities(probabilities)


def _list_images(folder_path):
    if not os.path.isdir(folder_path):
        raise NotADirectoryError(f"{folder_path} is not a directory.")
    filenames = sorted(
        f for f in os.listdir(folder_path) if f.lower().endswith(IMAGE_EXTENSIONS)
    )
    if not filenames:
        raise FileNotFoundError(f"No images found in {folder_path}.")
    return filenames


def predict_quality_batch(folder_path, model=None, model_path=DEFAULT_MODEL_PATH,
                           image_size=(224, 224), batch_size=32):
    """
    Run IQA inference on every image directly inside `folder_path` (not
    recursive), batched through a tf.data pipeline for throughput.

    Returns a list of per-image result dicts, in the same order as the
    sorted file listing, each augmented with "filename" and "path" on top
    of `predict_quality`'s usual "label"/"class_index"/"confidence"/
    "probabilities" fields.
    """
    if model is None:
        model = load_iqa_model(model_path)

    filenames = _list_images(folder_path)
    paths = [os.path.join(folder_path, name) for name in filenames]

    ds = (
        tf.data.Dataset.from_tensor_slices(paths)
        .map(lambda p: _decode_image(p, image_size), num_parallel_calls=tf.data.AUTOTUNE)
        .batch(batch_size)
        .prefetch(tf.data.AUTOTUNE)
    )
    probabilities = model.predict(ds, verbose=1)

    results = []
    for filename, path, proba in zip(filenames, paths, probabilities):
        result = _result_from_probabilities(proba)
        result["filename"] = filename
        result["path"] = path
        results.append(result)
    return results


def generate_iqa_csv(folder_path, output_csv=DEFAULT_CSV_PATH, model=None,
                      model_path=DEFAULT_MODEL_PATH, image_size=(224, 224), batch_size=32):
    """
    Run batch IQA inference over every image in `folder_path` and write the
    predictions to a CSV (default: EyePACS_IQA.csv under IQA_RESULTS_DIR).
    Columns mirror EyeQ's own labels.csv convention (image, quality) plus
    the predicted label, confidence, and full per-class probability
    distribution -- ready to gate or review EyePACS (or any other
    unlabeled fundus image folder) before it enters later pipeline stages.
    """
    results = predict_quality_batch(
        folder_path, model=model, model_path=model_path,
        image_size=image_size, batch_size=batch_size,
    )
    df = pd.DataFrame([
        {
            "image": r["filename"],
            "quality": r["class_index"],
            "quality_label": r["label"],
            "confidence": r["confidence"],
            "prob_good": r["probabilities"]["Good"],
            "prob_usable": r["probabilities"]["Usable"],
            "prob_reject": r["probabilities"]["Reject"],
        }
        for r in results
    ])

    os.makedirs(os.path.dirname(output_csv) or ".", exist_ok=True)
    df.to_csv(output_csv, index=False)
    print(f"Wrote {len(df)} predictions to {output_csv}")
    return df


def main():
    parser = argparse.ArgumentParser(
        description="Run Image Quality Assessment inference on a single image, "
                     "or batch-process a folder into an EyePACS_IQA.csv-style report."
    )
    parser.add_argument("path", help="Path to a retinal fundus image, or a folder of images.")
    parser.add_argument("--model", dest="model_path", default=DEFAULT_MODEL_PATH,
                         help="Path to the trained IQA model (.keras).")
    parser.add_argument("--output-csv", default=DEFAULT_CSV_PATH,
                         help="Where to write predictions when `path` is a folder.")
    args = parser.parse_args()

    if os.path.isdir(args.path):
        generate_iqa_csv(args.path, output_csv=args.output_csv, model_path=args.model_path)
    else:
        result = predict_quality(args.path, model_path=args.model_path)
        print(f"Quality: {result['label']} (class {result['class_index']}, "
              f"confidence {result['confidence']:.4f})")
        for name, prob in result["probabilities"].items():
            print(f"  {name}: {prob:.4f}")


if __name__ == "__main__":
    main()
