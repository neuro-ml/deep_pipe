from itertools import product

import numpy as np


def _get_steps(shape: np.ndarray, n_parts_per_axis):
    n_parts_per_axis = np.array(n_parts_per_axis)
    steps = shape // n_parts_per_axis
    steps += (shape % n_parts_per_axis) > 0
    return steps


def divide(x: np.ndarray, padding, n_parts_per_axis):
    """Divides padded x (should be padded beforehand)
    into multiple parts of about the same shape according to
     n_parts_per_axis list.
    and padding."""
    padding = np.array(padding)
    steps = _get_steps(np.array(x.shape) - 2 * padding, n_parts_per_axis)

    x_parts = []
    for idx in product(*map(range, n_parts_per_axis)):
        lb = np.array(idx) * steps
        slices = [*map(slice, lb, lb + steps + 2 * padding)]
        x_parts.append(x[slices])

    return x_parts


def combine(x_parts, n_parts_per_axis):
    """Combines parts of one big array back into one array, according to
     n_parts_per_axis."""
    assert x_parts[0].ndim == len(n_parts_per_axis)
    shape = _build_shape(x_parts, n_parts_per_axis)
    x = _combine_with_shape(x_parts, n_parts_per_axis, shape)
    return x


def _build_shape(x_parts, n_parts_per_axis):
    n_dims = len(n_parts_per_axis)
    n_parts = len(x_parts)
    shape = []
    for i in range(n_dims):
        step = np.prod(n_parts_per_axis[i + 1:], dtype=int)
        s = sum([x_parts[j*step].shape[i] for j in range(n_parts_per_axis[i])])
        shape.append(s)
    return shape


def _combine_with_shape(x_parts, n_parts_per_axis, shape):
    steps = _get_steps(np.array(shape), n_parts_per_axis)
    x = np.zeros(shape, dtype=x_parts[0].dtype)
    for i, idx in enumerate(product(*map(range, n_parts_per_axis))):
        lb = np.array(idx) * steps
        slices = [*map(slice, lb, lb + steps)]
        x[slices] = x_parts[i]

    return x
