from typing import Callable

import numpy as np

from .base import BatchPredict
from ..medim.utils import extract_dims


def add_dims(*data, ndims=1):
    """Increase the dimensionality of each entry in `data` by adding `ndim` leading singleton dimensions."""
    idx = (None,) * ndims
    return tuple(np.asarray(x)[idx] for x in data)


class AddExtractDims(BatchPredict):
    def __init__(self, ndims=1):
        self.ndims = ndims

    def validate(self, *inputs, validate_fn: Callable):
        y_pred, loss = validate_fn(*add_dims(*inputs, ndims=self.ndims))
        return extract_dims(y_pred, ndim=self.ndims), loss

    def predict(self, *inputs, predict_fn):
        return extract_dims(predict_fn(*add_dims(*inputs, ndims=self.ndims)), ndim=self.ndims)
