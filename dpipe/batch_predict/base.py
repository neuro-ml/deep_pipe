from abc import ABC, abstractmethod
from functools import partial

import numpy as np


class BatchPredict(ABC):
    """
    Interface that realizes the validation and inference logic.
    """

    @abstractmethod
    def validate(self, x, y, *, validate_fn):
        """
        Realizes the validation logic.

        Parameters
        ----------
        x:
            a single input object
        y:
            a single ground truth object
        validate_fn: callable(x, y) -> prediction, loss
            callable, that receives an input batch and a ground truth batch
            and returns the prediction batch and the loss

        Returns
        -------
        prediction:
            prediction for the input
        loss: float
            the validation loss
        """

    @abstractmethod
    def predict(self, x, *, predict_fn):
        """
        Realizes the inference logic.

        Parameters
        ----------
        x:
            a single input object
        predict_fn: callable(x) -> prediction
            callable, that receives an input batch
            and returns the prediction batch

        Returns
        -------
        prediction:
            prediction for the input
        """

    def make_predictor(self, predict_fn):
        return partial(self.predict, predict_fn=predict_fn)

    def make_validator(self, validate_fn):
        return partial(self.validate, validate_fn=validate_fn)


def validate_parts(inputs_iterator, *, validate_fn):
    weights, losses, y_preds = [], [], []
    for inputs in inputs_iterator:
        y_pred, loss = validate_fn(*inputs)
        y_preds.append(y_pred)
        losses.append(loss)
        weights.append(y_pred.size)

    loss = np.average(losses, weights=weights, axis=0)
    return y_preds, loss


def predict_parts(inputs_iterator, *, predict_fn):
    return [predict_fn(inputs) if type(inputs) is np.ndarray else predict_fn(*inputs) for inputs in inputs_iterator]


class DivideCombine(BatchPredict):
    def __init__(self, val_divide, val_combine, test_divide=None, test_combine=None):
        self.val_divide, self.test_divide = val_divide, test_divide or val_divide
        self.val_combine, self.test_combine = val_combine, test_combine or val_combine

    def validate(self, *inputs, validate_fn):
        y_preds, loss = validate_parts(self.val_divide(*inputs), validate_fn=validate_fn)
        return self.val_combine(y_preds), loss

    def predict(self, *inputs, predict_fn):
        return self.test_combine(predict_parts(self.test_divide(*inputs), predict_fn=predict_fn))
