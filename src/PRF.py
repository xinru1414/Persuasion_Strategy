from typing import NamedTuple

import torch
import torch.nn.functional as F
from Config import ConversationConfig

CONV_LABEL_NUM = ConversationConfig.conv_label_num
CONV_PAD_LABEL = ConversationConfig.conv_pad_label
SENT_LABEL_NUM = ConversationConfig.sent_label_num


class PRResults:
    def __init__(self, target, prediction, correct):
        self._target = target
        self._prediction = prediction
        self._correct = correct

    @classmethod
    def with_num_of_labels(cls, num_labels):
        return cls(*(list(range(num_labels)) for _ in range(3)))

    def __add__(self, other):
        assert isinstance(other, PRResults)
        assert self.num_labels == other.num_labels, 'Number of labels must match between results if they are being added.'
        new_target = [x+y for x, y in zip(self._target, other._target)]
        new_prediction = [x + y for x, y in zip(self._prediction, other._prediction)]
        new_correct = [x + y for x, y in zip(self._correct, other._correct)]
        return PRResults(new_target, new_prediction, new_correct)

    def add_item(self, target_label: int, predicted_label: int):
        assert 0 <= target_label < self.num_labels, f'Invalid target_label! Must be in range [{0}, {self.num_labels}) was {target_label}.'
        assert 0 <= predicted_label < self.num_labels, f'Invalid predicted_label! Must be in range [{0}, {self.num_labels}) was {predicted_label}.'
        self._target[target_label] += 1
        self._prediction[predicted_label] += 1
        if target_label == predicted_label:
            self._correct[target_label] += 1

    @property
    def num_labels(self):
        return len(self._target)

    @property
    def num_results(self):
        return sum(self._target)

    @property
    def count(self):
        return self.num_results

    @property
    def correct(self):
        return sum(self._correct)

    @property
    def accuracy(self):
        return self.correct / self.count

    @property
    def precision(self):
        return sum(num_targets/num_predictions
                   for num_targets, num_predictions in zip(self._target, self._prediction)
                   if num_predictions != 0) / self.num_labels

    @property
    def recall(self):
        return sum(num_targets/num_correct
                   for num_targets, num_correct in zip(self._target, self._correct)
                   if num_correct != 0) / self.num_labels

    @property
    def f1(self):
        return 2 * (self.recall * self.precision) / (self.recall + self.precision)


def get_item(x):
    return x.item()


def prf(predictions, targets, num_labels) -> PRResults:
    if len(targets.shape) > 1:
        targets = targets.view(targets.shape[0] * targets.shape[1])
    predictions = torch.argmax(F.softmax(predictions, dim=1), dim=1)

    results = PRResults.with_num_of_labels(num_labels)

    for prediction, target in zip(map(get_item, predictions), map(get_item, targets)):
        if target == CONV_PAD_LABEL:
            continue
        else:
            results.add_item(target_label=int(target), predicted_label=int(prediction))

    return results
