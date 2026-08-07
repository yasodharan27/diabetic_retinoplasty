"""
Abstract base class for the training half of a pipeline module.

Every future trainable module in the target 11-stage architecture (Vessel
Segmentation, Lesion Segmentation, Local/Global Feature Extraction, Feature
Fusion, Ordinal Classification, ...) must expose the same training-lifecycle
contract, so any orchestrator (a Colab notebook, a CLI entry point, or the
eventual end-to-end pipeline) can drive it without knowing its concrete type.
Mirrors the "training implementation" half of PROJECT_CODE.md's Deployment
Requirement -- the "inference implementation" half is `InferenceStage`.

Defines no models and no training logic itself; it only fixes the method
signatures every concrete stage must implement. Concrete subclasses are
expected to use the existing model-agnostic frameworks (`training.Trainer`,
`evaluation.Evaluator`) internally rather than reimplementing them.
"""

from abc import ABC, abstractmethod
from typing import Any, Optional


class TrainableStage(ABC):
    """Training-lifecycle contract for a pipeline stage with a trainable
    model. Concrete subclasses (e.g. a future `VesselSegmentationStage`)
    wrap their own model-building and dataset loading behind these four
    methods."""

    @abstractmethod
    def train(self, train_data: Any, val_data: Optional[Any] = None, **kwargs) -> Any:
        """Train the stage's model on `train_data` (optionally validating
        on `val_data`) and return a training result (e.g. a Keras
        `History`, or whatever the concrete stage's training loop
        produces)."""
        raise NotImplementedError

    @abstractmethod
    def evaluate(self, eval_data: Any, **kwargs) -> Any:
        """Evaluate the trained model against `eval_data` and return an
        evaluation report (e.g. `evaluation.EvaluationReport`, or a
        segmentation-specific Dice/IoU summary)."""
        raise NotImplementedError

    @abstractmethod
    def save(self, path: str) -> str:
        """Persist the trained model to `path`. Returns the path saved to,
        so callers can chain it into `MODEL_SAVE_PATH`-style config."""
        raise NotImplementedError

    @abstractmethod
    def load(self, path: str) -> "TrainableStage":
        """Load a previously trained model from `path` into this stage.
        Returns `self` so calls can be chained (`stage.load(path).evaluate(...)`)."""
        raise NotImplementedError
