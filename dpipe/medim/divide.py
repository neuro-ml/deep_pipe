import numpy as np

from dpipe.medim.box import get_boxes_grid
from .types import AxesLike
from .utils import build_slices


def divide(x: np.ndarray, patch_size: AxesLike, stride: AxesLike, axes: AxesLike = None):
    """
    A convolution-like approach to generating patches from a sequence of tensors.
    """
    return [x[(..., *build_slices(*box))] for box in get_boxes_grid(x.shape, patch_size, stride=stride, axes=axes)]


def combine(x_parts, x_shape):
    """Combines parts of one big array of shape x_shape back into one array."""
    x = np.zeros(x_shape, dtype=x_parts[0].dtype)
    part_shape = x_parts[0].shape
    for i, box in enumerate(get_boxes_grid(x_shape, box_size=part_shape, stride=part_shape)):
        x[build_slices(*box)] = x_parts[i]
    return x


def grid_patches(*arrays: np.ndarray, patch_size: AxesLike, stride: AxesLike, axes: AxesLike = None) -> np.ndarray:
    """
    A convolution-like approach to generating patches from a sequence of tensors.

    Parameters
    ----------
    arrays
    patch_size
    axes
        dimensions along which the slices will be taken.
    stride
        the stride (step-size) of the slice.
    """
    for box in get_boxes_grid(arrays[0].shape, patch_size, stride=stride, axes=axes):
        slc = (..., *build_slices(*box))
        yield tuple(array[slc] for array in arrays)
