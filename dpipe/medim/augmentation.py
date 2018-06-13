import numpy as np
from scipy import ndimage
from scipy.ndimage.interpolation import map_coordinates, zoom
from scipy.ndimage.filters import gaussian_filter


# TODO: simplify
def elastic_transform(x, alpha, sigma, axes=None, order=1):
    """
    Apply a gaussian elastic transform to a np.array along given axes.
    """

    if axes is None:
        axes = range(x.ndim)
    axes = list(sorted(axes))
    x = np.array(x)
    shape = np.array(x.shape)
    grid_shape = shape[axes]
    shape[axes] = 1

    dr = [gaussian_filter(np.random.rand(*grid_shape) * 2 - 1, sigma, mode="constant") * alpha
          for _ in range(len(grid_shape))]
    r = np.meshgrid(*[np.arange(i) for i in grid_shape], indexing='ij')

    indices = [np.reshape(k + dk, (-1, 1)) for k, dk in zip(r, dr)]

    result = np.empty_like(x)
    for idx in np.ndindex(*shape):
        idx = list(idx)
        for ax in axes:
            idx[ax] = slice(None)

        z = x[idx]
        result[idx] = map_coordinates(z, indices, order=order).reshape(z.shape)
    return result


# TODO: rewrite
def pad_slice(x, shape, pad_value=None):
    delta = np.array(shape) - x.shape

    d_pad = np.maximum(0, delta)
    d_pad = list(zip(d_pad // 2, (d_pad + 1) // 2))
    d_slice = np.maximum(0, -delta)
    d_slice = zip(d_slice // 2, x.shape - (d_slice + 1) // 2)
    d_slice = [slice(x, y) for x, y in d_slice]

    x = x[d_slice]
    if pad_value is None:
        pad_value = x.take(0)
    x = np.pad(x, d_pad, mode='constant', constant_values=pad_value)
    return x
