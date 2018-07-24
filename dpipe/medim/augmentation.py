import numpy as np
from scipy.ndimage.interpolation import map_coordinates
from scipy.ndimage.filters import gaussian_filter

from dpipe.medim.preprocessing import slice_to_shape, pad_to_shape


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


def pad_slice(x: np.ndarray, shape, axes=None, padding_values=0):
    return pad_to_shape(
        slice_to_shape(x, np.minimum(shape, x.shape), axes), np.maximum(shape, x.shape), axes, padding_values
    )
