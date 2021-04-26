"""
Tools for patch extraction and generation.
"""
from typing import Callable

import numpy as np

from .shape_ops import crop_to_box
from .box import returns_box
from .axes import fill_by_indices, AxesLike, resolve_deprecation, check_axes
from .shape_utils import shape_after_convolution, shape_after_full_convolution
from ..checks import check_shape_along_axis
from dpipe.itertools import squeeze_first, extract, lmap


def sample_box_center_uniformly(shape, box_size: np.array):
    """Returns the center of a sampled uniformly box of size ``box_size``, contained in the array of shape ``shape``."""
    return get_random_box(shape, box_size)[0] + box_size // 2


def uniform(shape):
    return np.array(lmap(np.random.randint, np.atleast_1d(shape)))


def get_random_patch(*arrays: np.ndarray, patch_size: AxesLike, axis: AxesLike = None,
                     distribution: Callable = uniform):
    """
    Get a random patch of size ``path_size`` along the ``axes`` for each of the ``arrays``.
    The patch position is equal for all the ``arrays``.

    Parameters
    ----------
    arrays
    patch_size
    axis
    distribution: Callable(shape)
        function that samples a random number in the range ``[0, n)`` for each axis. Defaults to a uniform distribution.
    """
    if not arrays:
        raise ValueError('No arrays given.')

    if axis is None:
        dims = lmap(np.ndim, arrays)
        if len(set(dims)) != 1:
            raise ValueError(f'Must pass the axes explicitly, because the arrays have different ndims: {dims}.')

        axis = resolve_deprecation(axis, arrays[0].ndim, patch_size)
    axis = check_axes(axis)
    check_shape_along_axis(*arrays, axis=axis)

    shape = extract(arrays[0].shape, axis)
    start = distribution(shape_after_convolution(shape, patch_size))
    box = np.array([start, start + patch_size])

    return squeeze_first(tuple(crop_to_box(arr, box, axis) for arr in arrays))


# TODO: what to do if axis != None?
@returns_box
def get_random_box(shape: AxesLike, box_shape: AxesLike, axis: AxesLike = None, distribution: Callable = uniform):
    """Get a random box of shape ``box_shape`` that fits in the ``shape`` along the given ``axes``."""
    axis = resolve_deprecation(axis, len(shape), box_shape)
    start = distribution(shape_after_full_convolution(shape, box_shape, axis))
    return start, start + fill_by_indices(shape, box_shape, axis)
