"""
Tools for patch extraction and generation.
"""
import numpy as np

from .checks import check_len
from .box import limit_box, get_box_padding, broadcast_spatial_box
from .utils import build_slices, get_axes, pad
from .shape_utils import shape_after_convolution, compute_shape_from_spatial


def extract_patch(x: np.ndarray, *, box: np.array, padding_values=None) -> np.array:
    """Returns patch, contained by the `box` from the `x` array. Will add padding, if `padding_values` were provided."""
    check_len(x.shape, *box)

    limited_box, padding = limit_box(box, x.shape), get_box_padding(box, x.shape)
    x_slice = x[build_slices(*limited_box)]
    if padding_values is None:
        np.testing.assert_array_equal(padding, 0, "Box is outside of the x and no padding_values.")
        patch = np.array(x_slice)
    else:
        patch = pad(x_slice, padding, padding_values)

    return patch


def extract_patch_spatial_box(x: np.ndarray, spatial_box: np.ndarray, spatial_dims, padding_values=None):
    check_len(*spatial_box, spatial_dims)
    return extract_patch(x, box=broadcast_spatial_box(x.shape, spatial_box, spatial_dims),
                         padding_values=padding_values)


def sample_box_center_uniformly(shape, box_size: np.array):
    """Returns `center` of a sampled uniformly box of size `box_size`, contained in the array of shape `shape`."""
    np.testing.assert_array_less(box_size - 1, shape, f'No box of size {box_size} can fit in the shape {shape}')

    max_center = shape - box_size + 1
    center = [np.random.randint(m) for m in max_center]
    return center + box_size // 2


def get_random_patch_start_stop(shape, patch_size, spatial_dims=None):
    spatial_dims = get_axes(spatial_dims, len(patch_size))

    start = np.zeros_like(shape)
    stop = np.array(shape)
    spatial = shape_after_convolution(stop[spatial_dims], patch_size)
    start[spatial_dims] = list(map(np.random.randint, spatial))
    stop[spatial_dims] = start[spatial_dims] + patch_size

    return start, stop


def get_random_patch(x: np.ndarray, patch_size, spatial_dims=None) -> np.ndarray:
    start, stop = get_random_patch_start_stop(x.shape, patch_size, spatial_dims)
    return x[build_slices(start, stop)]
