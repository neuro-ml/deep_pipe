from typing import Callable, Iterable

import numpy as np

from dpipe.medim.utils import build_slices

from .base import DivideCombine


# TODO: probably should implement DivideCombine methods using these functions instead of other way around
def make_predictor(divide: Callable[..., Iterable], combine: Callable[Iterable], predict_batch: Callable) -> Callable:
    """
    Builds a function that generates predictions for whole objects.

    Parameters
    ----------
    divide: Callable(*inputs) -> Iterable
        deconstructs the incoming object into batches.
    combine: Callable(Iterable)
        builds the final prediction from the predicted batches.
    predict_batch: Callable
        predicts a single batch.
    """
    return DivideCombine(divide, combine).make_predictor(predict_batch)


def make_validator(divide, combine, validate_batch):
    return DivideCombine(divide, combine).make_validator(validate_batch)


def validate_fn_with_shape(y_pred_loss, x_shape, f):
    y_pred, loss = y_pred_loss
    return f(y_pred, x_shape), loss


def add_batch_dim(x):
    return x[None]


def remove_batch_dim(x):
    assert x.shape[0] == 1
    return x[0]


def apply_first(x, f):
    return (f(x[0]), *x[1:])


def restore_shape(y, x_shape, spatial_dims: tuple):
    spatial_dims = list(spatial_dims)
    stop = np.array(y.shape)
    stop[spatial_dims] = np.array(x_shape)[spatial_dims]
    slices = build_slices(np.zeros_like(stop), stop)
    return y[slices]
