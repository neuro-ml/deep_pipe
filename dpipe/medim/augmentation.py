from typing import Sequence

import numpy as np
from scipy import ndimage
from scipy.ndimage.interpolation import map_coordinates, zoom
from scipy.ndimage.filters import gaussian_filter

from .utils import pad, get_axes


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


def spacial_augmentation(x, y, axes=None, order=1):
    if axes is None:
        axes = range(x.ndim)
    axes = list(sorted(axes))
    y = y.astype(float)
    # multithreading
    np.random.seed()

    # scale
    sigmas = np.zeros(x.ndim)
    sigmas[axes] = .15
    scale_factor = np.random.normal(1, sigmas)
    x = pad_slice(zoom(x, scale_factor, order=order), x.shape)
    y = pad_slice(zoom(y, scale_factor, order=order), y.shape)

    # rotate
    cval = x.take(0)
    angles = np.random.normal(0, 5, len(axes))
    for angle, *axis in zip(angles, axes, axes[1:]):
        x = ndimage.rotate(x, angle, axes=axis, reshape=False, order=order, cval=cval)
        y = ndimage.rotate(y, angle, axes=axis, reshape=False, order=order)

    stack = np.concatenate((x, y))
    stack = elastic_transform(stack, alpha=1, sigma=1, axes=axes)
    x, y = stack[:len(x)], stack[-len(y):]
    return x, y > .5


def spatial_augmentation_strict(x, y, axes=None, order=1):
    if axes is None:
        axes = range(x.ndim)
    axes = list(sorted(axes))
    y = y.astype(np.float32)
    # multithreading
    np.random.seed()

    # scale
    sigmas = np.zeros(x.ndim)
    sigmas[axes] = .15
    scale_factor = np.random.normal(1, sigmas)
    x = zoom(x, scale_factor, order=order)
    y = zoom(y, scale_factor, order=order)

    # rotate
    cval = x.take(0)
    angles = np.random.normal(0, 5, len(axes))
    for angle, *axis in zip(angles, axes, axes[1:]):
        x = ndimage.rotate(x, angle, axes=axis, reshape=False, order=order, cval=cval)
        y = ndimage.rotate(y, angle, axes=axis, reshape=False, order=order)

    stack = np.concatenate((x, y))
    stack = elastic_transform(stack, alpha=1, sigma=1, axes=axes, order=order)
    x, y = stack[:len(x)], stack[-len(y):]
    return x, y


def random_flip(x, y, axes):
    # multithreading
    np.random.seed()

    for axis, flip in zip(axes, np.random.binomial(1, .5, len(axes))):
        if flip:
            x = np.flip(x, axis)
            y = np.flip(y, axis)
    return x, y
