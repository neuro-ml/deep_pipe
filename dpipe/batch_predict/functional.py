from typing import Callable

import numpy as np

from dpipe.medim.utils import build_slices
from .base import predict_parts, validate_parts


def make_predictor(divide: Callable, combine: Callable, predict_batch: Callable) -> Callable:
    """
    Builds a function that generates a prediction for a whole object.

    Parameters
    ----------
    divide: Callable(*inputs) -> Iterable
        deconstructs the incoming object into batches.
    combine: Callable(Iterable)
        builds the final prediction from the predicted batches.
    predict_batch: Callable
        predicts a single batch.
    """
    return lambda *inputs: combine(predict_parts(divide(*inputs), predict_fn=predict_batch))


def make_validator(divide, combine, validate_batch):
    """
    Builds a function that generates both prediction and aggregated loss for a whole object.

    See `make_predictor` for details.
    """

    def validate(*inputs):
        predictions, loss = validate_parts(divide(*inputs), validate_fn=validate_batch)
        return combine(predictions), loss

    return validate


def validate_fn_with_shape(y_pred_loss, x_shape, f):
    y_pred, loss = y_pred_loss
    return f(y_pred, x_shape), loss


def restore_shape(y, x_shape, spatial_dims: tuple):
    spatial_dims = list(spatial_dims)
    stop = np.array(y.shape)
    stop[spatial_dims] = np.array(x_shape)[spatial_dims]
    slices = build_slices(np.zeros_like(stop), stop)
    return y[slices]
