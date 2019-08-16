"""
Tools for patch extraction and generation.
"""
import numpy as np

from .shape_ops import crop_to_box
from .box import returns_box
from .axes import expand_axes, fill_by_indices, AxesLike
from .shape_utils import shape_after_full_convolution, shape_after_convolution
from .checks import check_len, check_shape_along_axis
from .utils import lmap, name_changed
from .itertools import squeeze_first, extract

extract_patch = name_changed(crop_to_box, 'extract_patch', '27.06.2019')


@np.deprecate
def extract_patch_spatial_box(x: np.ndarray, spatial_box: np.ndarray, spatial_dims, padding_values=None):
    check_len(*spatial_box, spatial_dims)
    return crop_to_box(x, spatial_box, axes=spatial_dims, padding_values=padding_values)


def sample_box_center_uniformly(shape, box_size: np.array):
    """Returns the center of a sampled uniformly box of size ``box_size``, contained in the array of shape ``shape``."""
    return get_random_box(shape, box_size)[0] + box_size // 2


def get_random_patch(*arrays: np.ndarray, patch_size: AxesLike, axes: AxesLike = None):
    """
    Get a random patch of size ``path_size`` along the ``axes`` for each of the ``arrays``.
    The patch position is equal for all the arrays.
    """
    if not arrays:
        raise ValueError('No arrays given.')

    axes = expand_axes(axes, patch_size)
    check_shape_along_axis(*arrays, axis=axes)

    shape = extract(arrays[0].shape, axes)
    start = np.array(lmap(np.random.randint, shape_after_convolution(shape, patch_size, axes)))
    box = np.array([start, start + patch_size])

    return squeeze_first(tuple(crop_to_box(arr, box, axes) for arr in arrays))


# TODO: what to do if axis != None?
@returns_box
def get_random_box(shape: AxesLike, box_shape: AxesLike, axes: AxesLike = None):
    """Get a random box of shape ``box_shape`` that fits in the ``shape`` along the given ``axes``."""
    start = np.stack(lmap(np.random.randint, shape_after_full_convolution(shape, box_shape, axes)))
    return start, start + fill_by_indices(shape, box_shape, axes)
