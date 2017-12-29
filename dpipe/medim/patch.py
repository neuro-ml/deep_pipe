"""
Tools for patch extraction and generation
For even patch sizes, center is always considered to be close to the right border
"""
import numpy as np

from .utils import build_slices, pad, get_axes


# FIXME consider what happens if central_idx is outside of x, error is likely
# Probably need to rewrite to support it


def find_patch_start_end_padding(shape: np.ndarray, *, spatial_center_idx: np.array, spatial_patch_size: np.array,
                                 spatial_dims: list):
    spatial_start = spatial_center_idx - spatial_patch_size // 2
    spatial_end = spatial_start + spatial_patch_size

    padding = np.zeros((len(shape), 2), dtype=int)
    spatial_shape = shape[spatial_dims]

    padding[spatial_dims, 0] = -spatial_start
    padding[spatial_dims, 1] = spatial_end - spatial_shape
    padding[spatial_dims] = np.maximum(0, padding[spatial_dims])

    spatial_start = np.maximum(spatial_start, 0)
    spatial_end = np.minimum(spatial_end, spatial_shape)

    start = np.zeros(len(shape), dtype=int)
    start[spatial_dims] = spatial_start
    end = np.array(shape)
    end[spatial_dims] = spatial_end

    return start, end, padding


def extract_patch(x: np.ndarray, *, spatial_center_idx: np.array, spatial_patch_size: np.array, spatial_dims: list,
                  padding_values: np.array = None) -> np.array:
    """Returns extracted patch of specific spatial size and with specified
    center from x.
    Parameters
    ----------
    x
        Array with data. Some of it's dimensions are spatial. We extract
        spatial patch specified by spatial location and spatial size.
    spatial_center_idx
        Location of the center of the patch. Components
        correspond to spatial dimensions. If some of patch size components was
        even, patch center is supposed to be on the right center pixel.
    spatial_patch_size
        Spatial patch size. Output will have original shape for
        non-spatial dimensions and patch_size shape for spatial dimensions.
    spatial_dims
        Which of x's dimensions be consider as spatial. Accepts
        negative parameters.
    padding_values
        Defines values, that will fill padding. Have to be broadcastable to
        the resulting patch shape. By default no broadcasting is allowed.

    Returns
    -------
    :
        Patch extracted from x, padded, if necessary.

    """
    start, end, padding = find_patch_start_end_padding(
        np.array(x.shape), spatial_center_idx=spatial_center_idx, spatial_patch_size=spatial_patch_size,
        spatial_dims=spatial_dims
    )
    x_slice = x[build_slices(start, end)]
    if padding_values is None:
        np.testing.assert_array_equal(padding, 0, "patch is outside of the x")
        patch = np.array(x_slice)
    else:
        patch = pad(x_slice, padding, padding_values)

    return patch


def sample_uniform_center_index(x_shape: np.array, spatial_patch_size: np.array,
                                spatial_dims: list) -> np.array:
    """
    Returns spatial center coordinates for the patch, chosen randomly.
    We assume that patch have to belong to the object boundaries.

    Parameters
    ----------
    x_shape:
        Object shape.
    spatial_patch_size:
        Size of the required patch
    spatial_dims:
        Elements from x_shape that correspond to spatial dims. Can be negative.

    Returns
    -------
    :
        Center indices for spatial dims. If patch size was even, center index
        is shifted to the right.

    """
    max_spatial_center_idx = x_shape[spatial_dims] - spatial_patch_size + 1

    np.testing.assert_array_less(0, max_spatial_center_idx, 'x_shape is small')

    start_idx = np.random.rand(len(spatial_dims)) * max_spatial_center_idx
    start_idx = np.int32(start_idx)
    center_idx = start_idx + spatial_patch_size // 2
    return center_idx


def find_masked_patch_center_indices(mask: np.array, patch_size: np.array):
    """Returns array with spatial center indices for patches that completely 
    belong to spatial_mask and spatial voxel mask is activated."""
    c = np.argwhere(mask)

    l_bound = c - patch_size // 2
    r_bound = l_bound + patch_size

    # Remove centers that are too left and too right
    c = c[np.all((l_bound >= 0) & (r_bound <= np.array(mask.shape)), axis=1)]
    return c


def shape_after_convolution(shape, kernel_size, padding=0, stride=1, dilation=1) -> tuple:
    """
    Get the shape of a tensor after applying a the convolution with
    corresponding parameters.

    Parameters
    ----------
    shape
        input shape
    kernel_size
        convolution kernel size
    padding
        padding sizes
    stride
        stride of the convolution
    dilation
        dilation of the convolution kernel

    Returns
    -------
    output_shape: tuple
    """
    padding = np.asarray(padding)
    shape = np.asarray(shape)
    dilation = np.asarray(dilation)
    kernel_size = np.asarray(kernel_size)

    result = np.floor((shape + 2 * padding - dilation * (kernel_size - 1) - 1) / stride + 1)
    return result.astype(int)


def get_random_patch(x, patch_size, spatial_dims=None) -> np.ndarray:
    if spatial_dims is None:
        spatial_dims = list(range(-len(patch_size), 0))

    start = np.zeros_like(x.shape)
    stop = np.array(x.shape)
    spatial = shape_after_convolution(stop[spatial_dims], patch_size)
    start[spatial_dims] = [np.random.uniform(i) for i in spatial]
    stop[spatial_dims] = start[spatial_dims] + patch_size

    return x[build_slices(start, stop)]


def slices_conv(shape, kernel_size, spatial_dims=None, stride=None):
    """
    A convolution-like approach to generating slices from a tensor.

    Parameters
    ----------
    shape
        the input tensor's shape
    kernel_size
    spatial_dims
        dimensions along which the slices will be taken
    stride
        the stride (step-size) of the slice. If None, the stride is assumed
        to be equal to kernel_size.

    Yields
    ------
    start,stop: tuple
        coordinates of a slice's start and stop
    """
    spatial_dims = get_axes(spatial_dims, len(kernel_size))
    if stride is None:
        stride = kernel_size

    final_shape = np.array(shape).copy()
    spatial_shape = final_shape[spatial_dims]
    final_shape[:] = 1
    final_shape[spatial_dims] = shape_after_convolution(spatial_shape, kernel_size, stride=stride)

    whole_patch = np.array(shape).copy()
    whole_patch[spatial_dims] = kernel_size

    for i in np.ndindex(*final_shape):
        i = np.asarray(i)
        i[spatial_dims] *= stride
        yield tuple(i), tuple(i + whole_patch)


def patch_conv(x, patch_size, spatial_dims=None, stride=None, padding=0) -> np.ndarray:
    """
    A convolution-like approach to generating patches from a tensor.

    Parameters
    ----------
    x: np.array
        the input tensor
    patch_size
    spatial_dims
        dimensions along which the slices will be taken
    stride
        the stride (step-size) of the slice
    padding
        padding sizes of the input tensor

    Yields
    ------
    x_patch: np.ndarray
        patches from the input tensor
    """
    if padding:
        x = np.pad(x, pad_width=padding, mode='constant')

    for start, stop in slices_conv(x.shape, patch_size, spatial_dims, stride):
        yield x[build_slices(start, stop)]
