"""
Common pipeline interface layer.

Fixes the abstract contract every future trainable module in the target
11-stage architecture (PROJECT_CODE.md) must conform to, so stages can be
trained, evaluated, and chained into the eventual end-to-end pipeline
without the orchestrator knowing each stage's concrete model. Defines no
models and no pipeline orchestration itself -- see `training/` and
`evaluation/` for the model-agnostic training/evaluation frameworks these
stages are expected to use internally, and PROJECT_CODE.md's Deployment
Requirement for why both a `TrainableStage` and `InferenceStage` half exist.

Existing modules (Image Quality Assessment, preprocessing) are not modified
to inherit from these classes -- this package only establishes the contract
for stages implemented from this point on (Vessel Segmentation, Lesion
Segmentation, Local/Global Feature Extraction, Feature Fusion, Ordinal
Classification).
"""

from .classification import ClassificationStage
from .inference import InferenceStage
from .segmentation import SegmentationStage
from .trainable import TrainableStage

__all__ = [
    "TrainableStage",
    "InferenceStage",
    "SegmentationStage",
    "ClassificationStage",
]
