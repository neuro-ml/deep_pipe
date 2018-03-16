import itertools

import numpy as np

from .utils import build_slices


def get_start_stop(mask: np.ndarray):
    """Find indices of a box that contains all true values of mask, so that if
     you use them in slice() you will extract box with all true values."""
    start, stop = [], []
    for ax in itertools.combinations(range(mask.ndim), mask.ndim - 1):
        nonzero = np.any(mask, axis=ax)
        left, right = np.where(nonzero)[0][[0, -1]]
        start.insert(0, left)
        stop.insert(0, right + 1)
    return np.array(start), np.array(stop)


def get_slice(mask: np.ndarray):
    """Find slices of a box that contains all true values of mask, so that if
     you use them in mask[slices] you will extract box with all true values."""
    return build_slices(*get_start_stop(mask))


def extract(arrays, mask: np.ndarray):
    """Extract bounding boxes from the last dims of all arrays according to the
     mask."""
    s = [...] + get_slice(mask)
    return [np.array(a[s]) for a in arrays]
