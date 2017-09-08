# Tools for patch extraction and generation
import numpy as np

from .utils import extract

# FIXME consider what happens if central_idx is outside of x, error is likely
# Probably need to rewrite it to support it
# FIXME even patch size is not supported

def find_patch_start_end_padding(
        shape: np.ndarray, *, spatial_center_idx: np.array,
        spatial_patch_size: np.array, spatial_dims: list):
    assert (np.all((spatial_patch_size % 2) == 1),
            'even patch size is not supported')

    spatial_start = spatial_center_idx - spatial_patch_size // 2
    spatial_end = spatial_start + spatial_patch_size

    padding = np.zeros((len(shape), 2), dtype=int)
    spatial_shape = shape[spatial_dims]

    padding[spatial_dims, 0] = -spatial_start
    padding[spatial_dims, 1] = spatial_end - spatial_shape
    padding[spatial_dims] = np.maximum(0, padding[spatial_dims])

    spatial_start = np.maximum(spatial_start, 0)
    spatial_end = np.minimum(spatial_end, spatial_shape)

    start = np.zeros(len(shape))
    start[spatial_dims] = spatial_start
    end = np.array(shape)
    end[spatial_dims] = spatial_end

    return start, end, padding


def extract_patch(x: np.ndarray, *, spatial_center_idx: np.array,
                  spatial_patch_size: np.array, spatial_dims: list) -> np.array:
    """Returns extracted patch of specific spatial size and with specified 
    center from x.
    Parameters
    ----------
    x
        Array with data. Some of it's dimensions are spatial. We extract 
        spatial patch specified by spatial location and spatial size. If
        available patch is smaller than required, we pad with zeroes.
    spatial_center_idx
        Location of the center of the patch. Components
        correspond to spatial dimensions. If some of patch size components was
        even, patch center is supposed to be on the right center pixel.
    spatial_patch_size
        Spatial patch size. Output will have original shape for
        non-spatial dimensions and patch_size shape for spatial dimensions.
    spatial_dims
        Which of x's dimensions consider as spatial. Accepts
        negative parameters.
    padding_mode
        Defines operation, used to calculate values in padding. By default
        padding is not allowed.
    
    Returns
    -------
    :
        Patch extracted from x, padded, if necessary. 
 
    """
    slice, padding = find_patch_start_end_padding(
        np.array(x.shape), spatial_center_idx=spatial_center_idx,
        spatial_patch_size=spatial_patch_size, spatial_dims=spatial_dims)

    if padding_mode is None:
        if np.all(padding != 0):
            raise ValueError(
                "Padding mode was set to None, which doesn't allow padding "
                "but patch size doesn't fit with current centre.\n"
                f"Required padding: {padding}"
            )
        x_patch = np.array(x_slice)
    else:
        x_patch = np.zeros(np.sum(padding, axis=1) + np.sha)

    patch = np.pad(x[slices], padding, mode='constant')
    assert np.all([ps == ts for ps, ts in
                   zip(extract(patch.shape, spatial_dims), spatial_patch_size)])
    return patch


# FIXME consider what happens if central_idx is outside of x, error is likely
# Probably need to rewrite it to support it
def extract_patch(x: np.ndarray, *, spatial_center_idx: np.array,
                  spatial_patch_size: np.array, spatial_dims: list,
                  padding_mode: str = None) -> np.array:
    """Returns extracted patch of specific spatial size and with specified
    center from x.
    Parameters
    ----------
    x
        Array with data. Some of it's dimensions are spatial. We extract
        spatial patch specified by spatial location and spatial size. If
        available patch is smaller than required, we pad with zeroes.
    spatial_center_idx
        Location of the center of the patch. Components
        correspond to spatial dimensions. If some of patch size components was
        even, patch center is supposed to be on the right center pixel.
    spatial_patch_size
        Spatial patch size. Output will have original shape for
        non-spatial dimensions and patch_size shape for spatial dimensions.
    spatial_dims
        Which of x's dimensions consider as spatial. Accepts
        negative parameters.
    padding_mode
        Defines operation, used to calculate values in padding. By default
        padding is not allowed.

    Returns
    -------
    :
        Patch extracted from x, padded, if necessary.

    """
    slice, padding = find_patch_start_end_padding(
        np.array(x.shape), spatial_center_idx=spa)

    if padding_mode is None:
        if np.all(padding != 0):
            raise ValueError(
                "Padding mode was set to None, which doesn't allow padding "
                "but patch size doesn't fit with current centre.\n"
                f"Required padding: {padding}"
            )
        x_patch = np.array(x_slice)
    else:
        x_patch = np.zeros(np.sum(padding, axis=1) + np.sha)

    patch = np.pad(x[slices], padding, mode='constant')
    assert np.all([ps == ts for ps, ts in
                   zip(extract(patch.shape, spatial_dims), spatial_patch_size)])
    return patch



def get_uniform_center_index(x_shape: np.array, patch_size: np.array,
                             spatial_dims: list) -> np.array:
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
