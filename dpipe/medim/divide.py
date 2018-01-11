from itertools import product

import numpy as np

from .utils import pad


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
