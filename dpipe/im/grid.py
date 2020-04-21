"""
Function for working with patches from tensors.
See the :doc:`tutorials/patches` tutorial for more details.
"""
from typing import Iterable

import numpy as np

from .shape_ops import crop_to_box
from .axes import broadcast_to_axes, fill_by_indices, AxesLike
from .box import make_box_, Box
from dpipe.itertools import zip_equal, peek
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

    References
    ----------
    See the :doc:`tutorials/patches` tutorial for more details.
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

    References
    ----------
    See the :doc:`tutorials/patches` tutorial for more details.
    """
    for box in get_boxes(x.shape, patch_size, stride, axes, valid=valid):
        yield crop_to_box(x, box)


def combine(patches: Iterable[np.ndarray], output_shape: AxesLike, stride: AxesLike,
            axes: AxesLike = None, valid: bool = False) -> np.ndarray:
    """
    Build a tensor of shape ``output_shape`` from ``patches`` obtained in a convolution-like approach
    with corresponding parameters. The overlapping parts are averaged.

    References
    ----------
    See the :doc:`tutorials/patches` tutorial for more details.
    """
    axes, stride = broadcast_to_axes(axes, stride)
    patch, patches = peek(patches)
    patch_size = np.array(patch.shape)[list(axes)]
    if len(np.atleast_1d(output_shape)) != patch.ndim:
        output_shape = fill_by_indices(patch.shape, output_shape, axes)

    dtype = patch.dtype
    if not np.issubdtype(dtype, np.floating):
        dtype = float

    result = np.zeros(output_shape, dtype)
    counts = np.zeros(output_shape, int)
    for box, patch in zip_equal(get_boxes(output_shape, patch_size, stride, axes, valid), patches):
        slc = build_slices(*box)
        result[slc] += patch
        counts[slc] += 1

    np.true_divide(result, counts, out=result, where=counts > 0)
    return result
