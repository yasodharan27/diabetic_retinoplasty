"""
Abstract base class for classification pipeline stages -- Image Quality
Assessment and Ordinal Classification, stages 1 and 8 of PROJECT_CODE.md's
target architecture (EfficientNetB0 quality gate, CORN ordinal DR grading).

Combines `TrainableStage` and `InferenceStage` into the single contract a
classification module must satisfy, and narrows `predict`/`predict_batch`'s
return type to the classification-specific shape already established by
`image_quality_inference.py` (a dict with "label", "class_index",
"confidence", "probabilities"). No classifier architecture (EfficientNet,
CORN head, ...) is implemented here -- this only fixes the interface those
modules must conform to.
"""

from abc import abstractmethod
from typing import Any, Dict, List

from .inference import InferenceStage
from .trainable import TrainableStage


class ClassificationStage(TrainableStage, InferenceStage):
    """Contract for a trainable classification stage. `predict` returns a
    single result dict (`{"label": str, "class_index": int, "confidence":
    float, "probabilities": Dict[str, float]}`, matching the convention
    already used by `image_quality_inference.predict_quality`);
    `predict_batch` returns one such dict per input, in the same order."""

    @abstractmethod
    def predict(self, input_data: Any) -> Dict[str, Any]:
        """Run classification on a single input and return a result dict
        with at least "label", "class_index", "confidence", and
        "probabilities" keys."""
        raise NotImplementedError

    @abstractmethod
    def predict_batch(self, inputs: List[Any]) -> List[Dict[str, Any]]:
        """Run classification over a batch of inputs, returning one result
        dict per input, in the same order."""
        raise NotImplementedError
