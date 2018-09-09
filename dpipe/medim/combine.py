from typing import Sequence

import numpy as np

from .box import get_boxes_grid
from .utils import build_slices
from .itertools import zip_equal
from .shape_utils import fill_by_indices
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
    patch_shape = patches[0].shape
    output_shape = fill_by_indices(patch_shape, output_shape, axes)

    result = np.zeros(output_shape, patches[0].dtype)
    counts = np.zeros(output_shape, int)
    for box, patch in zip_equal(get_boxes_grid(output_shape, patch_shape, stride, axes, valid=False), patches):
        slc = build_slices(*box)
        result[slc] += patch
        counts[slc] += 1

    return result / np.maximum(1, counts)
