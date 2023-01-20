from typing import Callable, Union

import numpy as np
from imops.crop import crop_to_box
from imops.pad import pad, pad_to_divisible, pad_to_shape, restore_crop
from imops.zoom import zoom, zoom_to_shape

from ..itertools import extract
from .axes import AxesLike, AxesParams, broadcast_to_axis, fill_by_indices, resolve_deprecation
from .utils import build_slices

__all__ = [
    'zoom', 'zoom_to_shape', 'proportional_zoom_to_shape',
    'crop_to_shape', 'crop_to_box', 'restore_crop',
    'pad', 'pad_to_shape', 'pad_to_divisible',
]


def crop_to_shape(x: np.ndarray, shape: AxesLike, axis: AxesLike = None, ratio: AxesParams = 0.5) -> np.ndarray:
    """
    Crop ``x`` to match ``shape`` along ``axes``.
    Parameters
    ----------
    x
    shape
        final shape.
    axis
        axes along which ``x`` will be padded. If None - the last ``len(shape)`` axes are used.
    ratio
        the fraction of the crop that will be applied to the left, ``1 - ratio`` will be applied to the right.
    """
    if not hasattr(x, 'ndim') or not hasattr(x, 'shape'):
        x = np.asarray(x)

    axis = resolve_deprecation(axis, x.ndim, shape)
    shape, ratio = broadcast_to_axis(axis, shape, ratio)

    old_shape, new_shape = np.array(x.shape), np.array(fill_by_indices(x.shape, shape, axis))
    if (old_shape < new_shape).any():
        raise ValueError(f'The resulting shape cannot be greater than the original one: {old_shape} vs {new_shape}')

    ndim = len(x.shape)
    ratio = fill_by_indices(np.zeros(ndim), ratio, axis)
    start = ((old_shape - new_shape) * ratio).astype(int)

    return x[build_slices(start, start + new_shape)]


def proportional_zoom_to_shape(
    x: np.ndarray,
    shape: AxesLike,
    axis: AxesLike = None,
    padding_values: Union[AxesParams, Callable] = 0,
    order: int = 1,
) -> np.ndarray:
    """
    Proportionally rescale ``x`` to fit ``shape`` along ``axes`` then pad it to that shape.
    Parameters
    ----------
    x
    shape
        final shape.
    axis
        axes along which ``x`` will be padded. If None - the last ``len(shape)`` axes are used.
    padding_values
        values to pad with.
    order
        order of interpolation.
    """
    x = np.asarray(x)
    axis = resolve_deprecation(axis, x.ndim, shape, padding_values)
    scale_factor = (np.array(shape, 'float64') / extract(x.shape, axis)).min()

    return pad_to_shape(zoom(x, scale_factor, axis, order), shape, axis, padding_values)
