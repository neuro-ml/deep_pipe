import numpy as np
import warnings

from .checks import check_len

from dpipe.medim.utils import get_axes


def compute_shape_from_spatial(complete_shape, spatial_shape, spatial_dims):
    if spatial_dims is None:
        warnings.warn("Deprecated call, `spatial_dims` cannot be `None`.")
        spatial_dims = range(-len(spatial_shape), 0)
    check_len(spatial_shape, spatial_dims)
    shape = np.array(complete_shape)
    shape[list(spatial_dims)] = spatial_shape
    return tuple(shape)


def compute_full_shape(shape, shape_along_axes, axes):
    return compute_shape_from_spatial(shape, shape_along_axes, tuple(get_axes(axes, len(shape_along_axes))))


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


def shape_after_convolution(shape, kernel_size, padding=0, stride=1, dilation=1) -> tuple:
    """Get the shape of a tensor after applying a convolution with corresponding parameters."""
    padding, shape, dilation, kernel_size = map(np.asarray, [padding, shape, dilation, kernel_size])

    result = (shape + 2 * padding - dilation * (kernel_size - 1) - 1) / stride + 1
    new_shape = tuple(np.floor(result).astype(int))
    if (result < 1).any():
        raise ValueError(f'Such a convolution is not possible. Output shape: {new_shape}.')
    return new_shape
