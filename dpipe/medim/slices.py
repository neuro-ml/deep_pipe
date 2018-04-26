import numpy as np


def iterate_slices(*data: np.ndarray, axis: int = -1):
    """
    Iterate over slices of a series of tensors along a given axis.

    ----------
    data: Sequence
    axis: int, optional
    """
    size = data[0].shape[axis]
    assert all(x.shape[axis] == size for x in data)

    for idx in range(size):
        result = tuple(x.take(idx, axis=axis) for x in data)
        if len(result) == 1:
            result = result[0]
        yield result
