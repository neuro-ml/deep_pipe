import numpy as np
from scipy.ndimage.interpolation import map_coordinates
from scipy.ndimage.filters import gaussian_filter

from .box import get_boxes_grid
from .preprocessing import slice_to_shape, pad_to_shape
from .shape_utils import fill_remaining_axes, get_axes


def elastic_transform(x: np.ndarray, amplitude: float, axes=None, order: int = 1):
    """Apply a gaussian elastic distortion with a given amplitude to a tensor along the given axes."""
    axes = get_axes(axes, x.ndim)
    grid_shape = np.array(x.shape)[axes]
    deltas = [gaussian_filter(np.random.uniform(-amplitude, amplitude, grid_shape), 1) for _ in grid_shape]
    grid = np.mgrid[tuple(map(slice, grid_shape))] + deltas

    result = np.empty_like(x)
    kernel_size = fill_remaining_axes([1] * x.ndim, grid_shape, axes)
    for start, _ in get_boxes_grid(x.shape, kernel_size, 1):
        slc = list(start)
        for ax in axes:
            slc[ax] = slice(None)
        result[slc] = map_coordinates(x[slc], grid, order=order)

    return result


def pad_slice(x: np.ndarray, shape, axes=None, padding_values=0):
    return pad_to_shape(
        slice_to_shape(x, np.minimum(shape, x.shape), axes), np.maximum(shape, x.shape), axes, padding_values
    )
