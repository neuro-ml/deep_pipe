import numpy as np


def compute_shape_from_spatial(complete_shape, spatial_shape, spatial_dims):
    shape = np.array(complete_shape)
    shape[list(spatial_dims)] = spatial_shape
    return shape


def broadcast_shape_nd(shape, n):
    if len(shape) > n:
        raise ValueError(f'len({shape}) > {n}')
    return [1] * (n - len(shape)) + list(shape)


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
