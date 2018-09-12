from typing import Iterable

import numpy as np

from .axes import expand_axes, broadcast_to_axes, fill_by_indices, AxesLike
from .box import make_box_
from .itertools import zip_equal, extract, peek
from .shape_utils import shape_after_full_convolution
from .utils import build_slices


def get_boxes_grid(shape: AxesLike, box_size: AxesLike, stride: AxesLike, axes: AxesLike = None, valid: bool = True):
    """
    Yield boxes appropriate for a tensor of shape ``shape`` in a convolution-like fashion.

    Parameters
    ----------
    shape
        the input tensor's shape.
    box_size
    axes
        axes along which the slices will be taken.
    stride
        the stride (step-size) of the slice.
    valid
        whether boxes of size smaller than ``box_size`` should be left out.
    """
    final_shape = shape_after_full_convolution(shape, box_size, axes, stride, valid=valid)
    box_size, stride = np.broadcast_arrays(box_size, stride)

    full_box = fill_by_indices(shape, box_size, axes)
    full_stride = fill_by_indices(np.ones_like(shape), stride, axes)

    for start in np.ndindex(*final_shape):
        start = np.asarray(start) * full_stride
        yield make_box_([start, np.minimum(start + full_box, shape)])


def grid_patch(x: np.ndarray, patch_size: AxesLike, stride: AxesLike, axes: AxesLike = None,
               valid: bool = False) -> Iterable[np.ndarray]:
    """
    A convolution-like approach to generating patches from a tensor.

    Parameters
    ----------
    x
    patch_size
    axes
        dimensions along which the slices will be taken.
    stride
        the stride (step-size) of the slice.
    valid
        whether patches of size smaller than ``patch_size`` should be left out.
    """
    for box in get_boxes_grid(x.shape, patch_size, stride, axes, valid=valid):
        yield x[build_slices(*box)]


def combine(x_parts: Iterable[np.ndarray], x_shape):
    """Combines parts of one big array of shape x_shape back into one array by simple concatenation."""
    return combine_grid_patches(x_parts, x_shape)


# TODO: better doc
def combine_grid_patches(patches: Iterable[np.ndarray], output_shape: AxesLike, stride: AxesLike = None,
                         axes: AxesLike = None) -> np.ndarray:
    """
    Build a tensor of shape ``output_shape`` from ``patches`` obtained in a convolution-like approach
    with corresponding parameters. The overlapping parts are averaged.

    References
    ----------
    `grid_patch` `get_boxes_grid`
    """
    patch, patches = peek(patches)
    axes = expand_axes(axes, output_shape)
    if stride is None:
        stride = extract(patch.shape, axes)
    else:
        _, stride = broadcast_to_axes(axes, stride)
    output_shape = fill_by_indices(patch.shape, output_shape, axes)

    result = np.zeros(output_shape, patch.dtype)
    counts = np.zeros(output_shape, int)
    for box, patch in zip_equal(
            get_boxes_grid(output_shape, extract(patch.shape, axes), stride, axes, valid=False), patches):
        slc = build_slices(*box)
        result[slc] += patch
        counts[slc] += 1

    return result / np.maximum(1, counts)
