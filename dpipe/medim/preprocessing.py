from typing import Sequence

import numpy as np
from scipy import ndimage

from dpipe.medim.utils import get_axes, build_slices


def normalize_scan(scan, mean=True, std=True, drop_percentile: int = None):
    """Normalize scan to make mean and std equal to (0, 1) if stated.
    Supports robust estimation with drop_percentile."""
    if drop_percentile is not None:
        bottom = np.percentile(scan, drop_percentile)
        top = np.percentile(scan, 100 - drop_percentile)

        mask = (scan > bottom) & (scan < top)
        vals = scan[mask]
    else:
        vals = scan.flatten()

    assert vals.ndim == 1

    if mean:
        scan = scan - vals.mean()

    if std:
        scan = scan / vals.std()

    return np.array(scan, dtype=np.float32)


def normalize_mscan(mscan, mean=True, std=True, drop_percentile: int = None):
    """Normalize mscan to make mean and std equal to (0, 1) if stated.
    Supports robust estimation with drop_percentile."""
    new_mscan = np.zeros_like(mscan, dtype=np.float32)
    for i in range(len(mscan)):
        new_mscan[i] = normalize_scan(mscan[i], mean=mean, std=std,
                                      drop_percentile=drop_percentile)
    return new_mscan


def rotate_image(image: np.ndarray, angles: Sequence, axes: Sequence = None, order: int = 1, reshape=False):
    """
    Rotate an image along the given axes.

    Parameters
    ----------
    image: np.ndarray
        image to rotate
    angles: Sequence
    axes: Sequence
        axes along which the image will be rotated. The length must be equal to len(angles) + 1
    order: int, optional
        interpolation order
    reshape: bool, optional
        whether to reshape the resulting image

    Returns
    -------
    rotated_image: np.ndarray
    """
    axes = get_axes(axes, len(angles) + 1)
    assert len(axes) == len(angles) + 1
    result = image.copy()
    for angle, *axis in zip(angles, axes, axes[1:]):
        result = ndimage.rotate(result, angle, axes=axis, reshape=reshape, order=order)
    return result


def scale(image: np.array, spatial_shape: list, order: int = 3, axes: list = None) -> np.ndarray:
    """
    Rescale image to `spacial_shape` along the `axes`.

    Parameters
    ----------
    image: np.array
        image to rescale
    spatial_shape: Sequence
        final image shape
    order: int, optional
        order of interpolation
    axes: Sequence, optional
        axes along which the image will be scaled.
        If None - the last `len(spacial_shape)` axes are used.

    Returns
    -------
    reshaped image: np.ndarray
    """
    axes = get_axes(axes, len(spatial_shape))
    old_shape = np.array(image.shape)[axes].astype('float64')
    new_shape = np.array(spatial_shape).astype('float64')

    scale_factor = np.ones_like(image.shape, 'float64')
    scale_factor[axes] = new_shape / old_shape

    return ndimage.zoom(image, scale_factor, order=order)


def pad_to_shape(x: np.array, spatial_shape: Sequence, axes: Sequence = None, strict: bool = True) -> np.ndarray:
    """
    Pad a tensor to `spacial_shape` along the `axes`.

    Parameters
    ----------
    x: np.array
        tensor to pad
    spatial_shape: Sequence
        final image shape
    axes: Sequence, optional
        axes along which the image will be padded.
        If None - the last `len(spacial_shape)` axes are used.
    strict: bool, optional
        If True, the output shape cannot be smaller than the input shape.

    Returns
    -------
    padded_tensor: np.ndarray
    """
    axes = get_axes(axes, len(spatial_shape))
    old_shape = np.array(x.shape)[axes]
    new_shape = np.array(spatial_shape)

    if strict:
        assert (old_shape <= new_shape).all()

    delta = new_shape - old_shape
    delta = np.maximum(0, delta)
    padding_width = np.array((delta // 2, (delta + 1) // 2)).T

    padding = np.zeros((x.ndim, 2), int)
    padding[axes] = padding_width.astype(int)

    return np.pad(x, padding, mode='constant')


def slice_to_shape(x: np.array, spatial_shape: Sequence, axes: Sequence = None, strict: bool = True) -> np.ndarray:
    """
    Slice a tensor to `spacial_shape` along the `axes`.

    Parameters
    ----------
    x: np.array
        tensor to pad
    spatial_shape: Sequence
        final image shape
    axes: Sequence, optional
        axes along which the image will be padded.
        If None - the last `len(spacial_shape)` axes are used.
    strict: bool, optional
        If True, the output shape cannot be smaller than the input shape.

    Returns
    -------
    tensor_slice: np.ndarray
    """
    axes = get_axes(axes, len(spatial_shape))
    old_shape = np.array(x.shape)[axes]
    new_shape = np.array(spatial_shape)

    if strict:
        assert (old_shape >= new_shape).all()

    delta = old_shape - new_shape
    delta = np.maximum(0, delta)
    start, stop = np.zeros(x.ndim, dtype=int), np.array(x.shape)
    start[axes], stop[axes] = delta // 2, old_shape - (delta + 1) // 2

    return x[build_slices(start, stop)]
