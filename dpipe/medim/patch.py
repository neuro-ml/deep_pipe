"""
Tools for patch extraction and generation.
"""
import numpy as np

from .shape_ops import pad
from .box import returns_box
from .axes import expand_axes, fill_by_indices, AxesLike
from .shape_utils import shape_after_full_convolution
from .checks import check_len, check_shape_along_axis
from .box import limit_box, get_box_padding, broadcast_box
from .utils import build_slices, lmap
from .itertools import squeeze_first


# TODO: combine with crop_to_box
@np.deprecate
def extract_patch(x: np.ndarray, *, box: np.array, padding_values=None) -> np.array:
    check_len(x.shape, *box)

    limited_box, padding = limit_box(box, x.shape), get_box_padding(box, x.shape)
    x_slice = x[build_slices(*limited_box)]
    if padding_values is None:
        np.testing.assert_array_equal(padding, 0, "Box is outside of the x and no padding_values.")
        patch = np.array(x_slice)
    else:
        patch = pad(x_slice, padding, padding_values=padding_values)

    return patch


@np.deprecate
def extract_patch_spatial_box(x: np.ndarray, spatial_box: np.ndarray, spatial_dims, padding_values=None):
    check_len(*spatial_box, spatial_dims)
    return extract_patch(x, box=broadcast_box(spatial_box, x.shape, spatial_dims),
                         padding_values=padding_values)


# TODO: use get_random_box
def sample_box_center_uniformly(shape, box_size: np.array):
    """Returns `center` of a sampled uniformly box of size `box_size`, contained in the array of shape `shape`."""
    np.testing.assert_array_less(box_size - 1, shape, f'No box of size {box_size} can fit in the shape {shape}')

    max_center = shape - box_size + 1
    center = [np.random.randint(m) for m in max_center]
    return center + box_size // 2


def get_random_patch(*arrays: np.ndarray, patch_size: AxesLike, axes: AxesLike = None):
    """
    Get a random patch of size ``path_size`` along the ``axes`` for each of the ``arrays``.
    The patch position is equal for all the arrays.
    """
    check_shape_along_axis(*arrays, axis=expand_axes(axes, patch_size))

    slc = (..., *build_slices(*get_random_box(arrays[0].shape, patch_size, axes)))
    return squeeze_first(tuple(arr[slc] for arr in arrays))


@returns_box
def get_random_box(shape: AxesLike, box_shape: AxesLike, axes: AxesLike = None):
    """Get a random box of shape ``box_shape`` that fits in the ``shape`` along the given ``axes``."""
    start = np.stack(lmap(np.random.randint, shape_after_full_convolution(shape, box_shape, axes)))
    return start, start + fill_by_indices(shape, box_shape, axes)
