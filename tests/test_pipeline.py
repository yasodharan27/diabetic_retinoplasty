"""
Lightweight tests for the pipeline/ interface layer (TrainableStage,
InferenceStage, SegmentationStage, ClassificationStage). These classes have
no real implementation to exercise -- there is nothing here to run against
real data -- so the tests instead verify the ABC contract itself: each base
class is non-instantiable until every abstract method is implemented, a
minimal conforming subclass can be instantiated and used, and the
segmentation/classification specializations require the full method set
from both `TrainableStage` and `InferenceStage`.
"""

import unittest
from abc import ABCMeta

from pipeline import (
    ClassificationStage,
    InferenceStage,
    SegmentationStage,
    TrainableStage,
)


class _CompleteTrainableStage(TrainableStage):
    def __init__(self):
        self.trained_on = None
        self.saved_to = None
        self.loaded_from = None

    def train(self, train_data, val_data=None, **kwargs):
        self.trained_on = train_data
        return {"history": "ok"}

    def evaluate(self, eval_data, **kwargs):
        return {"accuracy": 1.0}

    def save(self, path):
        self.saved_to = path
        return path

    def load(self, path):
        self.loaded_from = path
        return self


class _CompleteInferenceStage(InferenceStage):
    def __init__(self):
        self.loaded_from = None

    def load(self, path):
        self.loaded_from = path
        return self

    def predict(self, input_data):
        return {"result": input_data}

    def predict_batch(self, inputs):
        return [self.predict(item) for item in inputs]


class _DummySegmentationStage(SegmentationStage):
    def __init__(self):
        self.loaded_from = None

    def train(self, train_data, val_data=None, **kwargs):
        return {"history": "ok"}

    def evaluate(self, eval_data, **kwargs):
        return {"dice": 1.0}

    def save(self, path):
        return path

    def load(self, path):
        self.loaded_from = path
        return self

    def predict(self, input_data):
        return [[0, 1], [1, 0]]

    def predict_batch(self, inputs):
        return [self.predict(item) for item in inputs]


class _DummyClassificationStage(ClassificationStage):
    def __init__(self):
        self.loaded_from = None

    def train(self, train_data, val_data=None, **kwargs):
        return {"history": "ok"}

    def evaluate(self, eval_data, **kwargs):
        return {"accuracy": 1.0}

    def save(self, path):
        return path

    def load(self, path):
        self.loaded_from = path
        return self

    def predict(self, input_data):
        return {"label": "Good", "class_index": 0, "confidence": 0.9,
                 "probabilities": {"Good": 0.9, "Reject": 0.1}}

    def predict_batch(self, inputs):
        return [self.predict(item) for item in inputs]


class _MissingSaveTrainableStage(TrainableStage):
    """Deliberately incomplete: implements everything except `save`."""

    def train(self, train_data, val_data=None, **kwargs):
        return None

    def evaluate(self, eval_data, **kwargs):
        return None

    def load(self, path):
        return self


class AbstractBaseClassTests(unittest.TestCase):
    def test_all_four_bases_are_abc(self):
        for cls in (TrainableStage, InferenceStage, SegmentationStage, ClassificationStage):
            self.assertIsInstance(cls, ABCMeta)

    def test_trainable_stage_cannot_be_instantiated_directly(self):
        with self.assertRaises(TypeError):
            TrainableStage()

    def test_inference_stage_cannot_be_instantiated_directly(self):
        with self.assertRaises(TypeError):
            InferenceStage()

    def test_segmentation_stage_cannot_be_instantiated_directly(self):
        with self.assertRaises(TypeError):
            SegmentationStage()

    def test_classification_stage_cannot_be_instantiated_directly(self):
        with self.assertRaises(TypeError):
            ClassificationStage()

    def test_incomplete_subclass_cannot_be_instantiated(self):
        with self.assertRaises(TypeError):
            _MissingSaveTrainableStage()


class TrainableStageContractTests(unittest.TestCase):
    def test_complete_subclass_is_instantiable_and_chainable(self):
        stage = _CompleteTrainableStage()
        self.assertIsInstance(stage, TrainableStage)

        stage.train(["sample"])
        self.assertEqual(stage.trained_on, ["sample"])

        report = stage.evaluate(["sample"])
        self.assertEqual(report, {"accuracy": 1.0})

        self.assertEqual(stage.save("models/x.keras"), "models/x.keras")

        returned = stage.load("models/x.keras")
        self.assertIs(returned, stage)
        self.assertEqual(stage.loaded_from, "models/x.keras")


class InferenceStageContractTests(unittest.TestCase):
    def test_complete_subclass_is_instantiable_and_batches_correctly(self):
        stage = _CompleteInferenceStage()
        self.assertIsInstance(stage, InferenceStage)

        returned = stage.load("models/x.keras")
        self.assertIs(returned, stage)

        single = stage.predict("image_1")
        self.assertEqual(single, {"result": "image_1"})

        batch = stage.predict_batch(["image_1", "image_2"])
        self.assertEqual(batch, [{"result": "image_1"}, {"result": "image_2"}])
        self.assertEqual(len(batch), 2)


class SegmentationStageContractTests(unittest.TestCase):
    def test_conforms_to_both_trainable_and_inference(self):
        stage = _DummySegmentationStage()
        self.assertIsInstance(stage, TrainableStage)
        self.assertIsInstance(stage, InferenceStage)
        self.assertIsInstance(stage, SegmentationStage)

        stage.train(["mask_pair"])
        self.assertEqual(stage.evaluate(["mask_pair"]), {"dice": 1.0})
        mask = stage.predict("image_1")
        self.assertEqual(mask, [[0, 1], [1, 0]])
        self.assertEqual(len(stage.predict_batch(["image_1", "image_2"])), 2)


class ClassificationStageContractTests(unittest.TestCase):
    def test_conforms_to_both_trainable_and_inference(self):
        stage = _DummyClassificationStage()
        self.assertIsInstance(stage, TrainableStage)
        self.assertIsInstance(stage, InferenceStage)
        self.assertIsInstance(stage, ClassificationStage)

        result = stage.predict("image_1")
        self.assertIn("label", result)
        self.assertIn("class_index", result)
        self.assertIn("confidence", result)
        self.assertIn("probabilities", result)

        batch = stage.predict_batch(["image_1", "image_2"])
        self.assertEqual(len(batch), 2)
        for item in batch:
            self.assertIn("probabilities", item)


if __name__ == "__main__":
    unittest.main()
