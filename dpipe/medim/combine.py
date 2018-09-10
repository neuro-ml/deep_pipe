from typing import Sequence

import numpy as np

from .box import get_boxes_grid
from .utils import build_slices
from .itertools import zip_equal, extract
from .axes import fill_by_indices, expand_axes, broadcast_to_axes
from .types import AxesLike


def combine_grid_patches(patches: Sequence[np.ndarray], output_shape: AxesLike, stride: AxesLike,
                         axes: AxesLike = None) -> np.ndarray:
    """
    Build a tensor of shape ``output_shape`` from ``patches`` obtained in a convolution-like approach
    with corresponding parameters. The overlapping parts are averaged.

    References
    ----------
    `grid_patch` `get_boxes_grid`
    """
    patch = patches[0]
    axes = expand_axes(axes, output_shape)
    axes, stride = broadcast_to_axes(axes, stride)
    output_shape = fill_by_indices(patch.shape, output_shape, axes)

    result = np.zeros(output_shape, patch.dtype)
    counts = np.zeros(output_shape, int)
    for box, patch in zip_equal(
            get_boxes_grid(output_shape, extract(patch.shape, axes), stride, axes, valid=False), patches):
        slc = build_slices(*box)
        result[slc] += patch
        counts[slc] += 1

    return result / np.maximum(1, counts)
