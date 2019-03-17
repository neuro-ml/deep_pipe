from typing import Iterable

import numpy as np

from .axes import broadcast_to_axes, fill_by_indices, AxesLike
from .box import make_box_, Box
from .itertools import zip_equal, extract, peek
from .shape_utils import shape_after_full_convolution
from .utils import build_slices

__all__ = 'get_boxes', 'divide', 'combine'


def get_boxes(shape: AxesLike, box_size: AxesLike, stride: AxesLike, axes: AxesLike = None,
              valid: bool = True) -> Iterable[Box]:
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


def divide(x: np.ndarray, patch_size: AxesLike, stride: AxesLike, axes: AxesLike = None,
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
    for box in get_boxes(x.shape, patch_size, stride, axes, valid=valid):
        yield x[build_slices(*box)]


# TODO: better doc
def combine(patches: Iterable[np.ndarray], output_shape: AxesLike, stride: AxesLike,
            axes: AxesLike = None, valid: bool = False) -> np.ndarray:
    """
    Build a tensor of shape ``output_shape`` from ``patches`` obtained in a convolution-like approach
    with corresponding parameters. The overlapping parts are averaged.

    References
    ----------
    `divide` `get_boxes`
    """
    axes, stride = broadcast_to_axes(axes, stride)
    patch, patches = peek(patches)
    if len(output_shape) != patch.ndim:
        output_shape = fill_by_indices(patch.shape, output_shape, axes)

    result = np.zeros(output_shape, patch.dtype)
    counts = np.zeros(output_shape, int)
    for box, patch in zip_equal(
            get_boxes(output_shape, extract(patch.shape, axes), stride, axes, valid), patches):
        slc = build_slices(*box)
        result[slc] += patch
        counts[slc] += 1

    return result / np.maximum(1, counts)
