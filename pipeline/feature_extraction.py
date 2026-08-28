"""
Abstract base class for spatial feature-extraction pipeline stages --
Local Feature Extraction (Stage 05) and, eventually, Global Feature
Extraction (Stage 06) of PROJECT_CODE.md's target architecture.

Combines `TrainableStage` and `InferenceStage` into the single contract a
feature-extraction module must satisfy, narrowing `predict`/`predict_batch`'s
return type to a spatial feature map -- neither a per-pixel probability/
binary mask (`SegmentationStage`) nor a class label (`ClassificationStage`).
No architecture is implemented here -- this only fixes the interface Stage
05/06 must conform to, so Stage 07 (Adaptive Cross-Attention) can consume
either without knowing its concrete architecture, mirroring exactly what
`segmentation.py`'s own docstring says about its role for Stage 03/04.
"""

from abc import abstractmethod
from typing import Any, List

from .inference import InferenceStage
from .trainable import TrainableStage


class FeatureExtractionStage(TrainableStage, InferenceStage):
    """Contract for a trainable spatial feature-extraction stage. `predict`
    returns a single spatial feature map (e.g. an `(H, W, C)` array, not
    globally pooled to a vector); `predict_batch` returns one such feature
    map per input, in the same order."""

    @abstractmethod
    def predict(self, input_data: Any) -> Any:
        """Run feature extraction on a single input and return its spatial
        feature map."""
        raise NotImplementedError

    @abstractmethod
    def predict_batch(self, inputs: List[Any]) -> List[Any]:
        """Run feature extraction over a batch of inputs, returning one
        spatial feature map per input in the same order."""
        raise NotImplementedError
