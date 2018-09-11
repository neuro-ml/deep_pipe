import numpy as np


def add_dims(*data, ndims=1):
    """Increase the dimensionality of each entry in `data` by adding `ndim` leading singleton dimensions."""
    idx = (None,) * ndims
    return tuple(np.asarray(x)[idx] for x in data)
