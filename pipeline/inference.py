"""
Abstract base class for the inference half of a pipeline module.

Mirrors the "inference implementation" half of PROJECT_CODE.md's Deployment
Requirement ("No module should exist only for training."). Every future
trainable module must also expose this contract so the eventual end-to-end
pipeline (PROJECT_CODE.md: "accepting one retinal fundus image and producing
the complete diabetic retinopathy analysis") can call any stage the same
way, regardless of its concrete model.

Already-implemented modules follow this shape informally today (see
`image_quality_inference.py`'s `load_iqa_model` / `predict_quality` /
`predict_quality_batch`); this class only formalizes the contract for
stages written from now on. Existing modules are not modified to inherit
from it.
"""

from abc import ABC, abstractmethod
from typing import Any, List


class InferenceStage(ABC):
    """Inference-lifecycle contract for a pipeline stage: load a trained
    artifact, then predict on one input or a batch of inputs."""

    @abstractmethod
    def load(self, path: str) -> "InferenceStage":
        """Load a trained model/artifact from `path` so `predict`/
        `predict_batch` can run. Returns `self` so calls can be chained."""
        raise NotImplementedError

    @abstractmethod
    def predict(self, input_data: Any) -> Any:
        """Run inference on a single input and return this stage's result
        (shape/type is defined by the concrete stage, e.g. a class
        probability dict for classification or a mask array for
        segmentation)."""
        raise NotImplementedError

    @abstractmethod
    def predict_batch(self, inputs: List[Any]) -> List[Any]:
        """Run inference over a batch/iterable of inputs, returning one
        result per input in the same order."""
        raise NotImplementedError
