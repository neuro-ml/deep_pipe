import warnings
from typing import Callable, Union, Sequence

import numpy as np
from scipy import ndimage

from .box import Box
from ..itertools import extract
from .axes import fill_by_indices, expand_axes, AxesLike, AxesParams, broadcast_to_axes
from .utils import build_slices

__all__ = [
    'zoom', 'zoom_to_shape', 'proportional_zoom_to_shape',
    'crop_to_shape', 'crop_to_box', 'restore_crop',
    'pad', 'pad_to_shape', 'pad_to_divisible',
]


def zoom(x: np.ndarray, scale_factor: AxesParams, axis: AxesLike = None, order: int = 1,
         fill_value: Union[float, Callable] = 0, *, axes: AxesLike = None) -> np.ndarray:
    """
    Rescale ``x`` according to ``scale_factor`` along the ``axes``.

    Parameters
    ----------
    x
    scale_factor
    axis
        axis along which the tensor will be scaled. If None - the last ``len(scale_factor)`` axes are used.
    order
        order of interpolation.
    fill_value
        value to fill past edges. If Callable (e.g. `numpy.min`) - ``fill_value(x)`` will be used.
    """
    axis = _resolve_deprecation(axis, axes, x.ndim, scale_factor)
    scale_factor = fill_by_indices(np.ones(x.ndim, 'float64'), scale_factor, axis)
    if callable(fill_value):
        fill_value = fill_value(x)

    # remove an annoying warning
    with warnings.catch_warnings():
        warnings.simplefilter('ignore', UserWarning)
        return ndimage.zoom(x, scale_factor, order=order, cval=fill_value)


def zoom_to_shape(x: np.ndarray, shape: AxesLike, axis: AxesLike = None, order: int = 1,
                  fill_value: Union[float, Callable] = 0, *, axes: AxesLike = None) -> np.ndarray:
    """
    Rescale ``x`` to match ``shape`` along the ``axes``.

    Parameters
    ----------
    x
    shape
        final shape.
    axis
        axes along which the tensor will be scaled. If None - the last ``len(shape)`` axes are used.
    order
        order of interpolation.
    fill_value
        value to fill past edges. If Callable (e.g. `numpy.min`) - ``fill_value(x)`` will be used.
    """
    axis = _resolve_deprecation(axis, axes, x.ndim, shape)
    old_shape = np.array(x.shape, 'float64')
    new_shape = np.array(fill_by_indices(x.shape, shape, axis), 'float64')
    return zoom(x, new_shape / old_shape, order=order, fill_value=fill_value)


def proportional_zoom_to_shape(x: np.ndarray, shape: AxesLike, axis: AxesLike = None,
                               padding_values: Union[AxesParams, Callable] = 0, order: int = 1, *,
                               axes: AxesLike = None) -> np.ndarray:
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
    axis = _resolve_deprecation(axis, axes, x.ndim, shape)
    axis = expand_axes(axis, shape)
    scale_factor = (np.array(shape, 'float64') / extract(x.shape, axis)).min()
    return pad_to_shape(zoom(x, scale_factor, axis, order), shape, axis, padding_values)


def pad(x: np.ndarray, padding: Union[AxesLike, Sequence[Sequence[int]]], axis: AxesLike = None,
        padding_values: Union[AxesParams, Callable] = 0, *, axes: AxesLike = None) -> np.ndarray:
    """
    Pad ``x`` according to ``padding`` along the ``axes``.

    Parameters
    ----------
    x
        tensor to pad.
    padding
        if 2D array [[start_1, stop_1], ..., [start_n, stop_n]] - specifies individual padding
        for each axis from ``axes``. The length of the array must either be equal to 1 or match the length of ``axes``.
        If 1D array [val_1, ..., val_n] - same as [[val_1, val_1], ..., [val_n, val_n]].
        If scalar (val) - same as [[val, val]].
    padding_values
        values to pad with, must be broadcastable to the resulting array.
        If Callable (e.g. `numpy.min`) - ``padding_values(x)`` will be used.
    axis
        axes along which ``x`` will be padded. If None - the last ``len(padding)`` axes are used.
    """
    padding = np.asarray(padding)
    if padding.ndim < 2:
        padding = padding.reshape(-1, 1)
    axis = _resolve_deprecation(axis, axes, x.ndim, padding)
    padding = np.asarray(fill_by_indices(np.zeros((x.ndim, 2), dtype=int), np.atleast_2d(padding), axis))
    if (padding < 0).any():
        raise ValueError(f'Padding must be non-negative: {padding.tolist()}.')
    if callable(padding_values):
        padding_values = padding_values(x)

    new_shape = np.array(x.shape) + np.sum(padding, axis=1)
    new_x = np.array(padding_values, dtype=x.dtype)
    new_x = np.broadcast_to(new_x, new_shape).copy()

    start = padding[:, 0]
    end = np.where(padding[:, 1] != 0, -padding[:, 1], None)
    new_x[build_slices(start, end)] = x
    return new_x


def pad_to_shape(x: np.ndarray, shape: AxesLike, axis: AxesLike = None, padding_values: Union[AxesParams, Callable] = 0,
                 ratio: AxesParams = 0.5, *, axes: AxesLike = None) -> np.ndarray:
    """
    Pad ``x`` to match ``shape`` along the ``axes``.

    Parameters
    ----------
    x
    shape
        final shape.
    padding_values
        values to pad with. If Callable (e.g. `numpy.min`) - ``padding_values(x)`` will be used.
    axis
        axes along which ``x`` will be padded. If None - the last ``len(shape)`` axes are used.
    ratio
        the fraction of the padding that will be applied to the left, ``1 - ratio`` will be applied to the right.
    """
    axis = _resolve_deprecation(axis, axes, x.ndim, shape)
    axis, shape, ratio = broadcast_to_axes(axis, shape, ratio)
    old_shape = np.array(x.shape)[list(axis)]
    if (old_shape > shape).any():
        shape = fill_by_indices(x.shape, shape, axis)
        raise ValueError(f'The resulting shape cannot be smaller than the original: {x.shape} vs {shape}')

    delta = shape - old_shape
    start = (delta * ratio).astype(int)
    padding = np.array((start, delta - start)).T.astype(int)
    return pad(x, padding, axis, padding_values=padding_values)


def pad_to_divisible(x: np.ndarray, divisor: AxesLike, axis: AxesLike = None,
                     padding_values: Union[AxesParams, Callable] = 0, ratio: AxesParams = 0.5,
                     remainder: AxesLike = 0, *, axes: AxesLike = None):
    """
    Pads ``x`` to be divisible by ``divisor`` along the ``axes``.

    Parameters
    ----------
    x
    divisor
        a value an incoming array should be divisible by.
    remainder
        ``x`` will be padded such that its shape gives the remainder ``remainder`` when divided by ``divisor``.
    axis
        axes along which the array will be padded. If None - the last ``len(divisor)`` axes are used.
    padding_values
        values to pad with. If Callable (e.g. `numpy.min`) - ``padding_values(x)`` will be used.
    ratio
        the fraction of the padding that will be applied to the left, ``1 - ratio`` will be applied to the right.

    References
    ----------
    `pad_to_shape`
    """
    axis = _resolve_deprecation(axis, axes, x.ndim, divisor)
    axis, divisor, remainder, ratio = broadcast_to_axes(axis, divisor, remainder, ratio)
    assert np.all(remainder >= 0)
    shape = np.maximum(np.array(x.shape)[list(axis)], remainder)
    return pad_to_shape(x, shape + (remainder - shape) % divisor, axis, padding_values, ratio)


def crop_to_shape(x: np.ndarray, shape: AxesLike, axis: AxesLike = None, ratio: AxesParams = 0.5, *,
                  axes: AxesLike = None) -> np.ndarray:
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
    axis = _resolve_deprecation(axis, axes, x.ndim, shape)
    axis, shape, ratio = broadcast_to_axes(axis, shape, ratio)
    old_shape, new_shape = np.array(x.shape), np.array(fill_by_indices(x.shape, shape, axis))
    if (old_shape < new_shape).any():
        raise ValueError(f'The resulting shape cannot be greater than the original one: {old_shape} vs {new_shape}')

    ndim = len(x.shape)
    ratio = fill_by_indices(np.zeros(ndim), ratio, axis)
    start = ((old_shape - new_shape) * ratio).astype(int)
    return x[build_slices(start, start + new_shape)]


def crop_to_box(x: np.ndarray, box: Box, axis: AxesLike = None, padding_values: AxesParams = None, *,
                axes: AxesLike = None) -> np.ndarray:
    """
    Crop ``x`` according to ``box`` along ``axes``.

    If axes is None - the last ``box.shape[-1]`` axes are used.
    """
    start, stop = box
    axis = _resolve_deprecation(axis, axes, x.ndim, start)
    axis = expand_axes(axis, start)

    slice_start = np.maximum(start, 0)
    slice_stop = np.minimum(stop, extract(x.shape, axis))
    padding = np.array([slice_start - start, stop - slice_stop], dtype=int).T
    if padding_values is None and padding.any():
        raise ValueError(f"The box {box} exceeds the input's limits {x.shape}.")

    slice_start = fill_by_indices(np.zeros(x.ndim, int), slice_start, axis)
    slice_stop = fill_by_indices(x.shape, slice_stop, axis)
    x = x[build_slices(slice_start, slice_stop)]

    if padding_values is not None and padding.any():
        x = pad(x, padding, axis, padding_values)
    return x


def restore_crop(x: np.ndarray, box: Box, shape: AxesLike, padding_values: AxesParams = 0) -> np.ndarray:
    """
    Pad ``x`` to match ``shape``. The left padding is taken equal to ``box``'s start.
    """
    assert len(shape) == x.ndim
    start, stop = box

    if (stop > shape).any() or (stop - start != x.shape).any():
        raise ValueError(f"The input array (of shape {x.shape}) was not obtained by cropping a "
                         f"box {start, stop} from the shape {shape}.")

    padding = np.array([start, shape - stop], dtype=int).T
    x = pad(x, padding, padding_values=padding_values)
    assert all(np.array(x.shape) == shape)
    return x


def _resolve_deprecation(axis, axes, ndim, values):
    values = len(np.atleast_1d(values))

    if axes is not None:
        assert axis is None
        msg = 'The argument `axes` is deprecated. Use `axis` instead.'
        warnings.warn(msg, UserWarning, 3)
        warnings.warn(msg, DeprecationWarning, 3)
        axis = axes

    if axis is None and ndim != values:
        msg = ('In the future the last axes will not be used to infer `axis`. '
               'Pass the appropriate `axis` to suppress this warning.')
        warnings.warn(msg, UserWarning, 3)
        warnings.warn(msg, DeprecationWarning, 3)

    return axis
