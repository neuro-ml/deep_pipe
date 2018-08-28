from typing import Callable
from functools import partial

import numpy as np

from .base import BatchPredict


def add_dims(*data, ndims=1):
    """Increase the dimensionality of each entry in `data` by adding `ndim` leading singleton dimensions."""
    idx = (None,) * ndims
    return tuple(np.asarray(x)[idx] for x in data)


def extract_dims(data, ndims=1):
    """Decrease the dimensionality of `data` by extracting `ndim` leading singleton dimensions."""
    for _ in range(ndims):
        assert len(data) == 1
        data = data[0]
    return data


class AddExtractDims(BatchPredict):
    def __init__(self, ndims=1):
        self.ndims = ndims

    def validate(self, *inputs, validate_fn: Callable):
        y_pred, loss = validate_fn(*add_dims(*inputs, ndims=self.ndims))
        return extract_dims(y_pred, ndims=self.ndims), loss

    def predict(self, *inputs, predict_fn):
        return extract_dims(predict_fn(*add_dims(*inputs, ndims=self.ndims)), ndims=self.ndims)


# Deprecated
# ----------

Simple = AddExtractDim = AddExtractDims
add_dimension = add_dims
extract_dimension = partial(extract_dims, ndims=2)
