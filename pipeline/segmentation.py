"""
Abstract base class for pixel-wise pipeline stages -- Vessel Segmentation
and Lesion Segmentation, stages 3 and 4 of PROJECT_CODE.md's target
architecture (U-Net, Attention U-Net respectively).

Combines `TrainableStage` and `InferenceStage` into the single contract a
segmentation module must satisfy, and narrows `predict`/`predict_batch`'s
return type to the segmentation-specific shape: a probability/binary mask
per input image, not a class label. No U-Net or Attention U-Net
architecture is implemented here -- this only fixes the interface those
future modules must conform to.
"""

from abc import abstractmethod
from typing import Any, List

from .inference import InferenceStage
from .trainable import TrainableStage


class SegmentationStage(TrainableStage, InferenceStage):
    """Contract for a trainable pixel-wise segmentation stage. `predict`
    returns a single mask (e.g. an `(H, W)` or `(H, W, C)` array of
    per-pixel probabilities); `predict_batch` returns one such mask per
    input, in the same order."""

    @abstractmethod
    def predict(self, input_data: Any) -> Any:
        """Run segmentation on a single preprocessed image and return its
        predicted mask (e.g. a vessel or lesion probability map)."""
        raise NotImplementedError

    @abstractmethod
    def predict_batch(self, inputs: List[Any]) -> List[Any]:
        """Run segmentation over a batch of preprocessed images, returning
        one mask per input in the same order."""
        raise NotImplementedError
