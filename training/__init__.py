"""
Reusable, model-agnostic training framework.

Provides the shared infrastructure (mixed precision, checkpointing, early
stopping, LR scheduling, TensorBoard, resume support, generic losses/metrics/
optimizers) that every training module -- Image Quality Assessment, Vessel
Segmentation, Lesion Segmentation, and Final Classification -- can reuse.
No model architectures and no dataset loading live here; both are supplied
by the caller (see `training.trainer.Trainer`).
"""

from .trainer import Trainer, TrainingConfig, check_gpu, enable_mixed_precision
from .callbacks import build_callbacks, EpochStateLogger, get_resume_epoch
from .losses import (
    get_loss,
    focal_loss,
    dice_loss,
    bce_dice_loss,
    weighted_bce_dice_loss,
    weighted_categorical_crossentropy,
)
from .metrics import build_metrics, dice_coefficient, iou_score, QuadraticWeightedKappa
from .optimizers import (
    build_optimizer,
    cosine_decay_schedule,
    exponential_decay_schedule,
    warmup_cosine_schedule,
)

__all__ = [
    "Trainer", "TrainingConfig", "check_gpu", "enable_mixed_precision",
    "build_callbacks", "EpochStateLogger", "get_resume_epoch",
    "get_loss", "focal_loss", "dice_loss", "bce_dice_loss", "weighted_bce_dice_loss",
    "weighted_categorical_crossentropy",
    "build_metrics", "dice_coefficient", "iou_score", "QuadraticWeightedKappa",
    "build_optimizer", "cosine_decay_schedule", "exponential_decay_schedule", "warmup_cosine_schedule",
]
