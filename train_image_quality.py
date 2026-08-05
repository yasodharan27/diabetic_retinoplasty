"""
Training entry point for pipeline Step 1 -- Image Quality Assessment.

Wires the EyeQ dataset loader (image_quality_dataset.py), the
EfficientNet-B0 IQA model (image_quality_model.py), and the reusable
training framework (training/) together -- mixed precision, checkpointing,
early stopping, ReduceLROnPlateau, TensorBoard, and resume support all come
from `training.Trainer` and are not reimplemented here.

Per the project's training policy, real training runs happen in Google
Colab (colab/notebooks/stage01_iqa.ipynb, which calls into the same two
modules). This script is the equivalent local/CLI entry point, useful for
quick local runs or as the reference the notebook mirrors.
"""

import argparse
import os

from config import EYEQ_RAW_DIR, IQA_MODEL_DIR
from image_quality_dataset import load_eyeq_datasets
from image_quality_model import build_iqa_model
from training import Trainer, TrainingConfig, check_gpu

DEFAULT_RUN_DIR = os.path.join(IQA_MODEL_DIR, "training_run")
DEFAULT_EXPORT_PATH = os.path.join(IQA_MODEL_DIR, "best_model.keras")
DEFAULT_FREEZE_LAYERS = 100


def train(raw_dir=EYEQ_RAW_DIR, run_dir=DEFAULT_RUN_DIR, export_path=DEFAULT_EXPORT_PATH,
          image_size=(224, 224), batch_size=16, epochs=50, learning_rate=1e-4,
          freeze_layers=DEFAULT_FREEZE_LAYERS, resume=False):
    check_gpu()

    train_ds, val_ds, class_weights = load_eyeq_datasets(
        raw_dir=raw_dir, image_size=image_size, batch_size=batch_size,
    )
    print(f"Class weights (train split): {class_weights}")

    model = build_iqa_model(
        input_shape=(*image_size, 3), learning_rate=learning_rate, freeze_layers=freeze_layers,
    )
    model.summary()

    # save_weights_only=False: checkpoints/export are complete models (.keras),
    # not weights-only files -- see training.callbacks.checkpoint_paths().
    config = TrainingConfig(run_dir=run_dir, epochs=epochs, resume=resume, save_weights_only=False)
    trainer = Trainer(config)
    history = trainer.fit(model, train_ds, val_ds, class_weight=class_weights)

    exported_path = trainer.export_best_weights(export_path)
    return model, history, exported_path


def main():
    parser = argparse.ArgumentParser(description="Train the Image Quality Assessment model.")
    parser.add_argument("--raw-dir", default=EYEQ_RAW_DIR)
    parser.add_argument("--run-dir", default=DEFAULT_RUN_DIR)
    parser.add_argument("--export-path", default=DEFAULT_EXPORT_PATH)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--freeze-layers", type=int, default=DEFAULT_FREEZE_LAYERS,
                         help="Number of leading EfficientNet-B0 backbone layers to freeze.")
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()

    train(
        raw_dir=args.raw_dir,
        run_dir=args.run_dir,
        export_path=args.export_path,
        batch_size=args.batch_size,
        epochs=args.epochs,
        learning_rate=args.learning_rate,
        freeze_layers=args.freeze_layers,
        resume=args.resume,
    )


if __name__ == "__main__":
    main()
