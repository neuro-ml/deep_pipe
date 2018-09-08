import numpy as np

from .itertools import extract
from .types import AxesLike
from .checks import check_len


def compute_shape_from_spatial(complete_shape, spatial_shape, spatial_dims):
    check_len(spatial_shape, spatial_dims)
    shape = np.array(complete_shape)
    shape[list(spatial_dims)] = spatial_shape
    return tuple(shape)


def fill_by_indices(target, values, indices):
    """Replace the values in ``target`` located at ``indices`` by the ones from ``values``."""
    indices = expand_axes(indices, values)
    target = np.array(target)
    target[list(indices)] = values
    return tuple(target)


def broadcast_shape_nd(shape, n):
    if len(shape) > n:
        raise ValueError(f'len({shape}) > {n}')
    return (1,) * (n - len(shape)) + tuple(shape)


def broadcast_shape(x_shape, y_shape):
    max_n = max(len(x_shape), len(y_shape))

    x_shape = broadcast_shape_nd(x_shape, max_n)
    y_shape = broadcast_shape_nd(y_shape, max_n)

    shape = []
    for i, j in zip(reversed(x_shape), reversed(y_shape)):
        if i == j or i == 1 or j == 1:
            shape.append(max(i, j))
        else:
            raise ValueError(f'shapes are not broadcastable: {x_shape} {y_shape}')
    return tuple(reversed(shape))


def check_axes(axes) -> tuple:
    axes = np.atleast_1d(axes)
    if axes.ndim != 1:
        raise ValueError(f'Axes must be 1D, but {axes.ndim}D provided.')
    if not np.issubdtype(axes.dtype, np.integer):
        raise ValueError(f'Axes must be integer, but {axes.dtype} provided.')
    axes = tuple(axes)
    if len(axes) != len(set(axes)):
        raise ValueError(f'Axes contain duplicates: {axes}.')
    return axes


def expand_axes(axes, values) -> tuple:
    if axes is None:
        axes = list(range(-len(np.atleast_1d(values)), 0))
    return check_axes(axes)


def shape_after_convolution(shape: AxesLike, kernel_size: AxesLike, stride: AxesLike = 1, padding: AxesLike = 0,
                            dilation: AxesLike = 1, valid: bool = True) -> tuple:
    """Get the shape of a tensor after applying a convolution with corresponding parameters."""
    padding, shape, dilation, kernel_size = map(np.asarray, [padding, shape, dilation, kernel_size])

    result = (shape + 2 * padding - dilation * (kernel_size - 1) - 1) / stride + 1
    to_int = np.floor if valid else np.ceil

    result = to_int(result).astype(int)
    new_shape = tuple(result)
    if (result <= 0).any():
        raise ValueError(f'Such a convolution is not possible. Output shape: {new_shape}.')
    return new_shape


def shape_after_full_convolution(shape: AxesLike, kernel_size: AxesLike, axes: AxesLike = None, stride: AxesLike = 1,
                                 padding: AxesLike = 0, dilation: AxesLike = 1, valid: bool = True) -> tuple:
    """
    Get the shape of a tensor after applying a convolution with corresponding parameters along the given axes.
    The dimensions along the remaining axes will become singleton.
    """
    axes = expand_axes(axes, np.broadcast_arrays(kernel_size, stride, padding, dilation)[0])

    return fill_by_indices(
        np.ones_like(shape),
        shape_after_convolution(extract(shape, axes), kernel_size, stride, padding, dilation, valid), axes
    )
