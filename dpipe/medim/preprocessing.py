from typing import Sequence

import numpy as np
from scipy import ndimage


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


def scale(image: np.array, spacial_shape: list, order: int = 3,
          axes: list = None) -> np.array:
    """
    Rescale image to `spacial_shape` along the `axes`.

    Parameters
    ----------
    image: np.array
        image to rescale
    spacial_shape: Sequence
        final image shape
    order: int, optional
        order of interpolation
    axes: Sequence, optional
        axes along which the image will be scaled.
        If None - the last `len(spacial_shape)` axes are used.

    Returns
    -------
    reshaped image: np.array
    """
    if axes is None:
        axes = range(-len(spacial_shape), 0)
    axes = list(sorted(axes))

    old_shape = np.array(image.shape)[axes].astype('float64')
    new_shape = np.array(spacial_shape).astype('float64')

    scale_factor = np.ones_like(image.shape, 'float64')
    scale_factor[axes] = new_shape / old_shape

    return ndimage.zoom(image, scale_factor, order=order)


def pad(image: np.array, spacial_shape: Sequence,
        axes: Sequence = None) -> np.array:
    """
    Pad image to `spacial_shape` along the `axes`.

    Parameters
    ----------
    image: np.array
        image to pad
    spacial_shape: Sequence
        final image shape
    axes: Sequence, optional
        axes along which the image will be padded.
        If None - the last `len(spacial_shape)` axes are used.

    Returns
    -------
    padded_image: np.array
    """
    if axes is None:
        axes = range(-len(spacial_shape), 0)
    axes = list(sorted(axes))

    old_shape = np.array(image.shape)[axes]
    new_shape = np.array(spacial_shape)

    assert (old_shape <= new_shape).all()

    delta = new_shape - old_shape
    pad = np.array((delta // 2, (delta + 1) // 2)).T

    padding = np.zeros((image.ndim, 2), int)
    padding[axes] = pad.astype(int)

    return np.pad(image, padding, mode='constant')
