from functools import partial

import numpy as np

from .base import DivideCombine


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


class AddExtractDims(DivideCombine):
    def __init__(self, ndims=1):
        super().__init__(partial(add_dims, ndims=ndims + 1), partial(extract_dims, ndims=ndims + 1))


# class MultiClass(DivideCombine):
#     def __init__(self):
#         super().__init__(lambda *xs: [add_dimension(*xs)], np.argmax(extract_dimension(x), axis=0))


# Deprecated
# ----------

Simple = AddExtractDim = AddExtractDims
add_dimension = add_dims
extract_dimension = partial(extract_dims, ndims=2)
