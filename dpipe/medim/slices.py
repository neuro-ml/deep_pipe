import numpy as np


def iterate_slices(*data: np.ndarray, axis: int = -1):
    """Iterate over slices of a series of tensors along a given axis."""

    size = data[0].shape[axis]
    if any(x.shape[axis] != size for x in data):
        raise ValueError('All the tensors must have the same size along the given axis')

    for idx in range(size):
        result = tuple(x.take(idx, axis=axis) for x in data)
        if len(result) == 1:
            result = result[0]
        yield result
