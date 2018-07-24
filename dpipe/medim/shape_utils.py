import numpy as np

from .checks import check_len


def compute_shape_from_spatial(complete_shape, spatial_shape, spatial_dims):
    check_len(spatial_shape, spatial_dims)
    shape = np.array(complete_shape)
    shape[list(spatial_dims)] = spatial_shape
    return tuple(shape)


def fill_remaining_axes(reference, values_along_axes, axes):
    """Replace the values in `reference` located at `axes` by the ones from `values_along_axes`."""
    reference = np.array(reference)
    values_along_axes = np.atleast_1d(values_along_axes)
    axes = get_axes(axes, len(values_along_axes))

    assert len(values_along_axes) == len(axes) or len(values_along_axes) == 1, f'{values_along_axes}, {axes}'
    reference[axes] = values_along_axes
    return tuple(reference)


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


def get_axes(axes, ndim):
    if axes is None:
        axes = list(range(-ndim, 0))
    return list(np.atleast_1d(axes))


def shape_after_convolution(shape, kernel_size, stride=1, padding=0, dilation=1) -> tuple:
    """Get the shape of a tensor after applying a convolution with corresponding parameters."""
    padding, shape, dilation, kernel_size = map(np.asarray, [padding, shape, dilation, kernel_size])
    # TODO: add ceil_mode?

    result = (shape + 2 * padding - dilation * (kernel_size - 1) - 1) / stride + 1
    new_shape = tuple(np.floor(result).astype(int))
    if (result < 1).any():
        raise ValueError(f'Such a convolution is not possible. Output shape: {new_shape}.')
    return new_shape


def shape_after_full_convolution(shape, kernel_size, axes=None, stride=1, padding=0, dilation=1) -> tuple:
    """
    Get the shape of a tensor after applying a convolution with corresponding parameters along the given axes.
    The dimensions along the remaining axes will become singleton.
    """
    axes = get_axes(axes, max(len(np.atleast_1d(x)) for x in [kernel_size, stride, padding, dilation]))

    return fill_remaining_axes(
        np.ones_like(shape),
        shape_after_convolution(np.array(shape)[axes], kernel_size, stride, padding, dilation), axes
    )
