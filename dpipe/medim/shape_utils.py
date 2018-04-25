import numpy as np


def compute_shape_from_spatial(complete_shape, spatial_shape, spatial_dims):
    if spatial_dims is None:
        spatial_dims = range(-len(spatial_shape), 0)
    shape = np.array(complete_shape)
    shape[list(spatial_dims)] = spatial_shape
    return tuple(shape)


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
            raise ValueError('shapes are not broadcastable:\n'
                             f'{x_shape};{y_shape}')
    return tuple(reversed(shape))


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

    # TODO: raise if division is not even
    result = np.floor((shape + 2 * padding - dilation * (kernel_size - 1) - 1) / stride + 1)
    return result.astype(int)
