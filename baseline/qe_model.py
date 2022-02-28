from pathlib import Path
from typing import List, Iterable

from kiwi.lib.predict import load_system
from kiwi.runner import Predictions


class PredictionWrapper:

    def __init__(self, predictions: Predictions, index: int):
        self.predictions = predictions
        self.index = index

    @property
    def contains_addition_error(self) -> bool:
        return "BAD" in self.predictions.target_tags_labels[self.index]

    @property
    def contains_omission_error(self) -> bool:
        return "BAD" in self.predictions.source_tags_labels[self.index]


class SupervisedQualityEstimationModel:

    def __init__(self, checkpoint_path: Path, gpu_id=None):
        self.checkpoint_path = checkpoint_path
        self.runner = load_system(self.checkpoint_path, gpu_id=gpu_id)

    def predict(self, src_sentences: List[str], tgt_sentences: List[str], batch_size=16) -> Iterable[PredictionWrapper]:
        assert len(src_sentences) == len(tgt_sentences)
        predictions = self.runner.predict(
            source=src_sentences,
            target=tgt_sentences,
            batch_size=batch_size,
        )
        for i in range(len(src_sentences)):
            yield PredictionWrapper(predictions, i)
