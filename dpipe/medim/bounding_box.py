import itertools

import numpy as np


def get_idx(mask: np.ndarray):
    """Find indices of a box that contains all true values of mask, so that if
     you use them in slice() you will extract box with all true values."""
    n = mask.ndim
    out = []
    for ax in itertools.combinations(range(n), n - 1):
        nonzero = np.any(mask, axis=ax)
        left, right = np.where(nonzero)[0][[0, -1]]
        out.append([left, right + 1])
    return np.array(list(reversed(out)))


def get_slice(mask: np.ndarray):
    """Find slices of a box that contains all true values of mask, so that if
     you use them in mask[slices] you will extract box with all true values."""
    limits = get_idx(mask)
    return [slice(*l) for l in limits]


def extract(arrays, mask: np.ndarray):
    """Extract bounding boxes from the last dims of all arrays according to the
     mask."""
    s = [...] + get_slice(mask)
    return [np.array(a[s]) for a in arrays]


def extract_fixed(*arrays, mask: np.array, size):
    """Extract bounding boxes of size=size from the last dims of all arrays
    according to the mask."""
    size = np.array(size)
    limits = get_idx(mask)

    s = limits[:, 1] - limits[:, 0]

    dif = size - s
    assert np.all(dif >= 0)
    limits[:, 0] = limits[:, 0] - dif
    limits[:, 1] = limits[:, 1] + np.maximum(-limits[:, 0], 0)
    limits[:, 0] = np.maximum(limits[:, 0], 0)

    slices = [...] + [slice(l, r) for l, r in limits]
    return [np.array(a[slices]) for a in arrays]
