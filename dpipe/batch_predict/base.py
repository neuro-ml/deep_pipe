from abc import ABC, abstractmethod
from functools import partial
from typing import Callable

import numpy as np


class BatchPredict(ABC):
    """
    Interface that realizes the validation and inference logic.
    """

    @abstractmethod
    def validate(self, *inputs, validate_fn: Callable):
        """
        Realizes the validation logic.

        Parameters
        ----------
        inputs
        validate_fn: Callable(*inputs) -> prediction, loss
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
    def predict(self, *inputs, predict_fn):
        """
        Realizes the inference logic.

        Parameters
        ----------
        inputs
        predict_fn: Callable(*inputs) -> prediction
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
    return [predict_fn(*inputs) for inputs in inputs_iterator]


class DivideCombine(BatchPredict):
    """
    Deconstructs an object into batches, feeds them into the network
    and combines the results to create the final prediction.

    Parameters
    ----------
    val_divide: Callable[..., Iterable]
        deconstructs the incoming object into batches. Used during validation.
    val_combine: Callable[Iterable]
        builds the final prediction from the predicted batches. Used during validation.
    test_divide: Callable[..., Iterable], optional
        same as `val_divide`. Used during test. If None - `val_divide` is used.
    test_combine: Callable[Iterable]
        same as `val_combine`. Used during test. If None - `val_combine` is used.
    """

    def __init__(self, val_divide: Callable, val_combine: Callable,
                 test_divide: Callable = None, test_combine: Callable = None):
        self.val_divide, self.test_divide = val_divide, test_divide or val_divide
        self.val_combine, self.test_combine = val_combine, test_combine or val_combine

    def validate(self, *inputs, validate_fn):
        y_preds, loss = validate_parts(self.val_divide(*inputs), validate_fn=validate_fn)
        return self.val_combine(y_preds), loss

    def predict(self, *inputs, predict_fn):
        return self.test_combine(predict_parts(self.test_divide(*inputs), predict_fn=predict_fn))
