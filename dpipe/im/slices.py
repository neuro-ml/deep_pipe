import numpy as np

from ..checks import check_shape_along_axis


def iterate_slices(*data: np.ndarray, axis: int):
    """Iterate over slices of a series of tensors along a given axis."""
    check_shape_along_axis(*data, axis=axis)

    for idx in range(data[0].shape[axis]):
        yield tuple(x.take(idx, axis=axis) for x in data)


def iterate_axis(x: np.ndarray, axis: int):
    return np.moveaxis(x, axis, 0)
