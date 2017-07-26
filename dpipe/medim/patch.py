# Tools for patch extraction and generation
from typing import List

import numpy as np

from .utils import extract


def extract_patch(x: np.ndarray, *, center_idx: np.array, patch_size: np.array,
                  spatial_dims: list) -> np.array:
    """Returns extracted patch of specific spatial size and with specified 
    center from x.
    Parameters
    ----------
    x
        Array with data. Some of it's dimensions are spatial. We extract 
        spatial patch specified by spatial location and spatial size. If
        available patch is smaller than required, we pad with zeroes.
    center_idx
        Location of the center of the patch. Components
        correspond to spatial dimensions. If some of patch size components was
        even, patch center is supposed to be on the right center pixel.
    patch_size
        Spatial patch size. Output will have original shape for
        non-spatial dimensions and patch_size shape for spatial dimensions.
    spatial_dims
        Which of x's dimensions consider as spatial. Accepts
        negative parameters.
    
    Returns
    -------
    :
        Patch extracted from x, padded, if necessary. 
 
    """
    start = center_idx - patch_size // 2
    end = start + patch_size

    padding = np.zeros((x.ndim, 2), dtype=int)
    spatial_shape = np.array(extract(x.shape, spatial_dims))

    padding[spatial_dims, 0] = -start
    padding[spatial_dims, 1] = end - spatial_shape
    padding[spatial_dims] = np.maximum(0, padding[spatial_dims])

    start = np.maximum(start, 0)
    end = np.minimum(end, spatial_shape)

    slices = [slice(None)] * x.ndim
    for i, s in enumerate(spatial_dims):
        slices[s] = slice(start[i], end[i])

    x = x[slices]
    if np.any(padding > 0):
        x = np.pad(x, padding, mode='constant')
    return x


def extract_patches(xs: List[np.ndarray], *, center_idx: np.array,
                    patch_sizes: List[np.array],
                    spatial_dims: list) -> List[np.array]:
    """Applies extract_patch for each object in xs with corresponing patch
     size."""
    patches = []
    for i, x in enumerate(xs):
        patch = extract_patch(
            x, center_idx=center_idx, spatial_dims=spatial_dims,
            patch_size=patch_sizes[i])
        patches.append(patch)
    return patches


def get_uniform_center_index(x_shape: np.array, patch_size: np.array,
                             spatial_dims: list):
    """
    Returns spatial center coordinates for the patch, chosen randomly.
    We assume that patch have to belong to the object boundaries.
    
    Parameters
    ----------
    x_shape:
        Object shape.
    patch_size:
        Size of the required patch
    spatial_dims:
        Elements from x_shape that correspond to spatial dims. Can be negative. 

    Returns
    -------
    :
        Center indices for spatial dims. If patch size was even, center index
        is shifted to the right. 

    """
    max_spatial_center_idx = x_shape[spatial_dims] - patch_size + 1

    start_idx = np.random.rand(len(spatial_dims)) * max_spatial_center_idx
    start_idx = np.int32(start_idx)
    center_idx = start_idx + patch_size // 2
    return center_idx


def get_conditional_center_indices(
        spatial_mask: np.array, patch_size: np.array, spatial_dims: list):
    """Returns array with spatial center indices for patches that completely 
    belong to spatial_mask and spatial voxel mask is activated."""
    c = np.argwhere(spatial_mask)

    l_bound = c - patch_size // 2
    r_bound = c + patch_size // 2 + patch_size % 2

    # Remove centers that are too left and too right
    c = c[np.all((l_bound >= 0) &
                 (r_bound <= np.array(spatial_mask.shape)), axis=1)]
    return c
