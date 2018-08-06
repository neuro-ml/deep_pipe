import numpy as np


def iterate_slices(*data: np.ndarray, axis: int = -1):
    """Iterate over slices of a series of tensors along a given axis."""

    size = data[0].shape[axis]
    if any(x.shape[axis] != size for x in data):
        raise ValueError('All the tensors must have the same size along the given axis')

    for idx in range(size):
        yield tuple(x.take(idx, axis=axis) for x in data)


def iterate_axis(x: np.ndarray, axis: int):
    assert axis < x.ndim
    for i in range(x.shape[axis]):
        yield x.take(i, axis=axis)
