from functools import partial

import numpy as np
from scipy.ndimage.interpolation import map_coordinates
from scipy.ndimage.filters import gaussian_filter

from dpipe.itertools import extract
from .utils import apply_along_axes
from .axes import AxesLike, axis_from_dim


def elastic_transform(x: np.ndarray, amplitude: float, axis: AxesLike = None, order: int = 1):
    """Apply a gaussian elastic distortion with a given amplitude to a tensor along the given axes."""
    axis = axis_from_dim(axis, x.ndim)
    grid_shape = extract(x.shape, axis)
    deltas = [gaussian_filter(np.random.uniform(-amplitude, amplitude, grid_shape), 1) for _ in grid_shape]
    grid = np.mgrid[tuple(map(slice, grid_shape))] + deltas

    return apply_along_axes(partial(map_coordinates, coordinates=grid, order=order), x, axis)
