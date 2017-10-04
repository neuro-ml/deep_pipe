from itertools import product

import numpy as np

from .utils import pad


def compute_n_parts_per_axis(x_shape: np.array, patch_size: np.array):
    return np.ceil(x_shape / patch_size).astype(int)


def divide_no_padding(x: np.ndarray, patch_size: np.array,
                      intersection_size: np.array):
    """Divides padded x (should be padded beforehand) into multiple patches of
    patch_size shape with intersection."""
    assert len(x.shape) == len(patch_size) == len(intersection_size)

    n_parts_per_axis = compute_n_parts_per_axis(
        np.array(x.shape) - 2 * intersection_size, patch_size
    )

    x_parts = []
    for idx in product(*map(range, n_parts_per_axis)):
        lb = np.array(idx) * patch_size
        slices = [*map(slice, lb, lb + patch_size + 2 * intersection_size)]
        x_parts.append(x[slices])

    return x_parts


def divide(x: np.ndarray, patch_size: np.array, intersection_size: np.array,
           padding_values):
    """Divides x into multiple patches of patch_size shape with intersection. To
    provide values on boarders, perform padding with values from padding_values,
    which has to be broadcastable to x shape."""
    x_padded = pad(x, padding=np.repeat(intersection_size[:, None], 2, axis=1),
                   padding_values=padding_values)
    return divide_no_padding(x_padded, patch_size=patch_size,
                             intersection_size=intersection_size)


def combine(x_parts, x_shape: np.array):
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
