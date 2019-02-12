import numpy as np
from scipy import ndimage

from .box import Box
from .itertools import extract
from .axes import fill_by_indices, expand_axes, AxesLike, AxesParams
from .utils import build_slices, pad as pad_, name_changed

__all__ = [
    'zoom', 'zoom_to_shape', 'proportional_zoom_to_shape',
    'crop_to_shape', 'crop_to_box', 'restore_crop',
    'pad', 'pad_to_shape',
]


def zoom(x: np.ndarray, scale_factor: AxesParams, axes: AxesLike = None, order: int = 1) -> np.ndarray:
    """
    Rescale ``x`` according to ``scale_factor`` along the ``axes``.

    Parameters
    ----------
    x
    scale_factor:
    axes
        axes along which the tensor will be scaled. If None - the last `len(shape)` axes are used.
    order
        order of interpolation.
    """
    return ndimage.zoom(x, fill_by_indices(np.ones(x.ndim), scale_factor, axes), order=order)


def zoom_to_shape(x: np.ndarray, shape: AxesLike, axes: AxesLike = None, order: int = 1) -> np.ndarray:
    """
    Rescale ``x`` to match ``shape`` along the ``axes``.

    Parameters
    ----------
    x
    shape
        final shape.
    axes
        axes along which the tensor will be scaled. If None - the last `len(shape)` axes are used.
    order
        order of interpolation.
    """
    old_shape = np.array(x.shape, 'float64')
    new_shape = np.array(fill_by_indices(x.shape, shape, axes), 'float64')

    return ndimage.zoom(x, new_shape / old_shape, order=order)


def proportional_zoom_to_shape(x: np.ndarray, shape: AxesLike, axes: AxesLike = None, padding_values: AxesParams = 0,
                               order: int = 1) -> np.ndarray:
    """
    Proportionally rescale ``x`` to fit ``shape`` along ``axes`` then pad it to that shape.

    Parameters
    ----------
    x
    shape
        final shape.
    axes
        axes along which ``x`` will be padded. If None - the last `len(shape)` axes are used.
    padding_values
        values to pad with.
    order
        order of interpolation.
    """
    axes = expand_axes(axes, shape)
    scale_factor = (np.array(shape, 'float64') / extract(x.shape, axes)).min()
    return pad_to_shape(zoom(x, scale_factor, axes, order), shape, axes, padding_values)


def pad(x: np.ndarray, padding: AxesLike, axes: AxesLike = None, padding_values: AxesParams = 0) -> np.ndarray:
    """
    Pad ``x`` according to ``padding`` along the ``axes``.

    Parameters
    ----------
    x
        tensor to pad.
    padding
        padding in a format compatible with `numpy.pad`
    padding_values
        values to pad with.
    axes
        axes along which ``x`` will be padded. If None - the last `len(padding)` axes are used.
    """
    padding = fill_by_indices(np.zeros((x.ndim, 2), dtype=int), np.atleast_2d(padding), axes)
    return pad_(x, padding, padding_values)


def pad_to_shape(x: np.ndarray, shape: AxesLike, axes: AxesLike = None, padding_values: AxesParams = 0) -> np.ndarray:
    """
    Pad ``x`` to match ``shape`` along the ``axes``.

    Parameters
    ----------
    x
    shape
        final shape.
    padding_values
        values to pad with.
    axes
        axes along which ``x`` will be padded. If None - the last `len(shape)` axes are used.
    """
    old_shape, new_shape = np.array(x.shape), np.array(fill_by_indices(x.shape, shape, axes))
    if (old_shape > new_shape).any():
        raise ValueError(f'The resulting shape cannot be smaller than the original: {old_shape} vs {new_shape}')

    delta = new_shape - old_shape
    padding = np.array((delta // 2, (delta + 1) // 2)).T.astype(int)

    return pad(x, padding, padding_values=padding_values)


def crop_to_shape(x: np.ndarray, shape: AxesLike, axes: AxesLike = None) -> np.ndarray:
    """
    Crop ``x`` to match ``shape`` along ``axes``.

    Parameters
    ----------
    x
    shape
        final shape.
    axes
        axes along which ``x`` will be padded. If None - the last `len(shape)` axes are used.
    """
    old_shape, new_shape = np.array(x.shape), np.array(fill_by_indices(x.shape, shape, axes))
    if (old_shape < new_shape).any():
        raise ValueError(f'The resulting shape cannot be greater than the original one: {old_shape} vs {new_shape}')

    start = ((old_shape - new_shape) // 2).astype(int)
    return x[build_slices(start, start + new_shape)]


def crop_to_box(x: np.ndarray, box: Box, axes: AxesLike = None) -> np.ndarray:
    """
    Crop ``x`` according to ``box`` along ``axes``.

    If axes is None - the last ``box.shape[-1]`` axes are used.
    """
    start, stop = box
    if (stop > x.shape).any():
        raise ValueError(f"The box's stop {stop} is bigger than the input shape {x.shape}.")

    start = fill_by_indices(np.zeros(x.ndim, int), start, axes)
    stop = fill_by_indices(x.shape, stop, axes)

    return x[build_slices(start, stop)]


def restore_crop(x: np.ndarray, box: Box, shape: AxesLike, padding_values: AxesParams = 0):
    """
    Pad ``x`` to match ``shape``. The left padding is taken equal to ``box``'s start.
    """
    assert len(shape) == x.ndim
    start, stop = box

    if (stop > shape).any() or (stop - start != x.shape).any():
        raise ValueError(f"The input array (of shape {x.shape}) was not obtained by cropping a "
                         f"box {start, stop} from the shape {shape}.")

    padding = np.array([start, shape - stop], dtype=int).T
    return pad(x, padding, padding_values=padding_values)


slice_to_shape = name_changed(crop_to_shape, 'slice_to_shape', '14.01.2019')
scale_to_shape = name_changed(zoom_to_shape, 'scale_to_shape', '14.01.2019')
proportional_scale_to_shape = name_changed(proportional_zoom_to_shape, 'scale', '14.01.2019')
scale = name_changed(zoom, 'scale', '14.01.2019')
