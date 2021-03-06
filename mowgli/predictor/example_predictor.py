import random
from typing import List, Any

from mowgli.classes import Dataset
from mowgli.predictor.predictor import Predictor


class ExamplePredictor(Predictor):

    def preprocess(self, dataset: Dataset, config: Any) -> Any:
        return dataset

    def tune(self, dataset: Dataset, config: Any) -> Any:
        return None

    def predict(self, model: Any, dataset: Dataset, config: Any, partition: str) -> List:
        entries = getattr(dataset, partition)
        all_answers = []
        all_probs = []
        for entry in entries:
            question = entry.question
            answers = entry.answers

            answer = random.choice(entry.labels)
            all_answers.append(answer)

            probs = ['%.2f' % random.random() for i in range(len(answers))]
            all_probs.append(probs)
        return all_answers, all_probs
