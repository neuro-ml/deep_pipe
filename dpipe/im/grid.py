"""
Function for working with patches from tensors.
See the :doc:`tutorials/patches` tutorial for more details.
"""
from typing import Iterable, Type, Tuple, Callable

import numpy as np

from .shape_ops import crop_to_box
from .axes import fill_by_indices, AxesLike, resolve_deprecation, axis_from_dim, broadcast_to_axis
from .box import make_box_, Box
from dpipe.itertools import zip_equal, peek
from .shape_utils import shape_after_convolution
from .utils import build_slices

__all__ = 'get_boxes', 'divide', 'combine', 'PatchCombiner', 'Average'


def get_boxes(shape: AxesLike, box_size: AxesLike, stride: AxesLike, axis: AxesLike = None,
              valid: bool = True) -> Iterable[Box]:
    """
    Yield boxes appropriate for a tensor of shape ``shape`` in a convolution-like fashion.

    Parameters
    ----------
    shape
        the input tensor's shape.
    box_size
    axis
        axes along which the slices will be taken.
    stride
        the stride (step-size) of the slice.
    valid
        whether boxes of size smaller than ``box_size`` should be left out.

    References
    ----------
    See the :doc:`tutorials/patches` tutorial for more details.
    """
    axis = resolve_deprecation(axis, len(shape), box_size, stride)
    box_size, stride = broadcast_to_axis(axis, box_size, stride)
    box_size = fill_by_indices(shape, box_size, axis)
    stride = fill_by_indices(np.ones_like(shape), stride, axis)

    final_shape = shape_after_convolution(shape, box_size, stride, valid=valid)
    for start in np.ndindex(*final_shape):
        start = np.asarray(start) * stride
        yield make_box_([start, np.minimum(start + box_size, shape)])


def divide(x: np.ndarray, patch_size: AxesLike, stride: AxesLike, axis: AxesLike = None,
           valid: bool = False, get_boxes: Callable = get_boxes) -> Iterable[np.ndarray]:
    """
    A convolution-like approach to generating patches from a tensor.

    Parameters
    ----------
    x
    patch_size
    axis
        dimensions along which the slices will be taken.
    stride
        the stride (step-size) of the slice.
    valid
        whether patches of size smaller than ``patch_size`` should be left out.
    get_boxes
        function that yields boxes, for signature see ``get_boxes``

    References
    ----------
    See the :doc:`tutorials/patches` tutorial for more details.
    """
    for box in get_boxes(x.shape, patch_size, stride, axis, valid=valid):
        yield crop_to_box(x, box)


class PatchCombiner:
    def __init__(self, shape: Tuple[int, ...], dtype: np.dtype):
        self.dtype = dtype
        self.shape = shape

    def update(self, box: Box, patch: np.ndarray):
        raise NotImplementedError

    def build(self) -> np.ndarray:
        raise NotImplementedError


class Average(PatchCombiner):
    def __init__(self, shape: Tuple[int, ...], dtype: np.dtype):
        super().__init__(shape, dtype)
        self._result = np.zeros(shape, dtype)
        self._counts = np.zeros(shape, int)

    def update(self, box: Box, patch: np.ndarray):
        slc = build_slices(*box)
        self._result[slc] += patch
        self._counts[slc] += 1

    def build(self):
        np.true_divide(self._result, self._counts, out=self._result, where=self._counts > 0)
        return self._result


def combine(patches: Iterable[np.ndarray], output_shape: AxesLike, stride: AxesLike,
            axis: AxesLike = None, valid: bool = False,
            combiner: Type[PatchCombiner] = Average, get_boxes: Callable = get_boxes) -> np.ndarray:
    """
    Build a tensor of shape ``output_shape`` from ``patches`` obtained in a convolution-like approach
    with corresponding parameters.
    The overlapping parts are aggregated using the strategy from ``combiner`` - Average by default.

    References
    ----------
    See the :doc:`tutorials/patches` tutorial for more details.
    """
    patch, patches = peek(patches)
    patch = np.asarray(patch)
    patch_size = patch.shape

    axis = axis_from_dim(axis, patch.ndim)
    output_shape, stride = broadcast_to_axis(axis, output_shape, stride)
    output_shape = fill_by_indices(patch_size, output_shape, axis)
    stride = fill_by_indices(patch_size, stride, axis)

    if np.greater(patch_size, output_shape).any():
        raise ValueError(f'output_shape {output_shape} is smaller than the patch size {patch_size}')

    dtype = patch.dtype
    if not np.issubdtype(dtype, np.floating):
        dtype = float

    combiner = combiner(output_shape, dtype)
    for box, patch in zip_equal(get_boxes(output_shape, patch_size, stride, valid=valid), patches):
        combiner.update(box, patch)

    return combiner.build()
