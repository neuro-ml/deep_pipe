import warnings
from functools import partial

import numpy as np
from scipy.ndimage.interpolation import map_coordinates
from scipy.ndimage.filters import gaussian_filter

from dpipe.itertools import extract
from .utils import apply_along_axes
from .axes import expand_axes, AxesLike


def elastic_transform(x: np.ndarray, amplitude: float, axis: AxesLike = None, order: int = 1, *, axes: AxesLike = None):
    """Apply a gaussian elastic distortion with a given amplitude to a tensor along the given axes."""
    if axes is not None:
        assert axis is None
        warnings.warn('`axes` has been renamed to `axis`', UserWarning)
        axis = axes

    axis = expand_axes(axis, x.shape)
    grid_shape = extract(x.shape, axis)
    deltas = [gaussian_filter(np.random.uniform(-amplitude, amplitude, grid_shape), 1) for _ in grid_shape]
    grid = np.mgrid[tuple(map(slice, grid_shape))] + deltas

    return apply_along_axes(partial(map_coordinates, coordinates=grid, order=order), x, axis)
