from itertools import product

import numpy as np

from .shape_utils import shape_after_convolution
from .utils import get_axes, build_slices, pad


def compute_n_parts_per_axis(x_shape, patch_size):
    return np.ceil(np.asarray(x_shape) / np.asarray(patch_size)).astype(int)


def divide_no_padding(x: np.ndarray, patch_size, intersection_size):
    """Divides padded x (should be padded beforehand) into multiple patches of
    patch_size shape with intersections of size intersection_size."""
    assert len(x.shape) == len(patch_size) == len(intersection_size)
    patch_size = np.array(patch_size)
    intersection_size = np.array(intersection_size)

    n_parts_per_axis = compute_n_parts_per_axis(
        np.array(x.shape) - 2 * intersection_size,
        patch_size - 2 * intersection_size
    )

    x_parts = []
    for idx in product(*map(range, n_parts_per_axis)):
        lb = np.array(idx) * (patch_size - 2 * intersection_size)
        slices = [*map(slice, lb, lb + patch_size)]
        x_parts.append(x[slices])

    return x_parts


def divide(x: np.ndarray, patch_size, intersection_size, padding_values=None):
    """Divides x into multiple patches of patch_size shape with intersections of
    size intersection_size. To provide values on boarders, perform padding with
    values from padding_values, which has to be broadcastable to x shape."""
    intersection_size = np.array(intersection_size)
    if padding_values is not None:
        x = pad(x, padding=np.repeat(intersection_size[:, None], 2, axis=1),
                padding_values=padding_values)
    else:
        assert np.all(intersection_size == 0)
    return divide_no_padding(x, patch_size=patch_size,
                             intersection_size=intersection_size)


def divide_spatial(x: np.ndarray, *, spatial_patch_size, spatial_intersection_size, padding_values=None,
                   spatial_dims: list):
    patch_size = np.array(x.shape)
    patch_size[spatial_dims] = spatial_patch_size

    intersection_size = np.zeros(x.ndim, dtype=int)
    intersection_size[spatial_dims] = spatial_intersection_size

    return divide(x, patch_size=patch_size, intersection_size=intersection_size,
                  padding_values=padding_values)


def combine(x_parts, x_shape):
    """Combines parts of one big array of shape x_shape back into one array."""
    patch_size = np.array(x_parts[0].shape)
    n_parts_per_axis = compute_n_parts_per_axis(x_shape, patch_size)
    assert len(x_shape) == len(patch_size)

    x = np.zeros(x_shape, dtype=x_parts[0].dtype)
    for i, idx in enumerate(product(*map(range, n_parts_per_axis))):
        lb = np.array(idx) * patch_size
        slices = [*map(slice, lb, lb + patch_size)]
        x[slices] = x_parts[i]
    return x


def get_grid_patch_start_stop(shape, kernel_size, spatial_dims=None, stride=None):
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
    # TODO: simplify logic
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


def divide_grid_patch(x, patch_size, spatial_dims=None, stride=None, padding=0) -> np.ndarray:
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

    for start, stop in get_grid_patch_start_stop(x.shape, patch_size, spatial_dims, stride):
        yield x[build_slices(start, stop)]
