import numpy as np
from scipy import ndimage

from .itertools import extract
from .types import AxesParams, AxesLike
from .shape_utils import fill_by_indices, expand_axes
from .utils import build_slices, pad, apply_along_axes, scale as _scale


def normalize_image(image: np.ndarray, mean: bool = True, std: bool = True, drop_percentile: int = None):
    """
    Normalize an ``image`` to make mean and std equal to 0 and 1 respectively (if required).
    Supports robust estimation with drop_percentile.
    """
    if drop_percentile is not None:
        bottom = np.percentile(image, drop_percentile)
        top = np.percentile(image, 100 - drop_percentile)

        mask = (image > bottom) & (image < top)
        vals = image[mask]
    else:
        vals = image.flatten()

    assert vals.ndim == 1

    if mean:
        image = image - vals.mean()
    if std:
        image = image / vals.std()

    return image


def normalize_multichannel_image(image: np.ndarray, mean: bool = True, std: bool = True, drop_percentile: int = None):
    """
    Normalize an ``image`` to make mean and std equal to 0 and 1 respectively (if required)
    for each channel separately. Supports robust estimation with drop_percentile.
    """
    return np.array([normalize_image(channel, mean, std, drop_percentile) for channel in image], np.float32)


def min_max_scale(x: np.ndarray, axes: AxesLike = None):
    """Scale ``x``'s values so that its minimum and maximum along ``axes`` become 0 and 1 respectively."""
    return apply_along_axes(_scale, x, expand_axes(axes, x.shape))


def scale(x: np.ndarray, scale_factor: AxesParams, axes: AxesLike = None, order: int = 1) -> np.ndarray:
    """
    Rescale a tensor according to ``scale_factor``.

    Parameters
    ----------
    x
        tensor to rescale.
    scale_factor:
        scale factor for each axis.
    axes
        axes along which the tensor will be scaled. If None - the last `len(shape)` axes are used.
    order
        order of interpolation.
    """
    return ndimage.zoom(x, fill_by_indices(np.ones(x.ndim), scale_factor, axes), order=order)


def scale_to_shape(x: np.ndarray, shape: AxesLike, axes: AxesLike = None, order: int = 1) -> np.ndarray:
    """
    Rescale a tensor to ``shape`` along the ``axes``.

    Parameters
    ----------
    x
        tensor to rescale.
    shape
        final tensor shape.
    axes
        axes along which the tensor will be scaled. If None - the last `len(shape)` axes are used.
    order
        order of interpolation.
    """
    old_shape = np.array(x.shape, 'float64')
    new_shape = np.array(fill_by_indices(x.shape, shape, axes), 'float64')

    return ndimage.zoom(x, new_shape / old_shape, order=order)


def pad_to_shape(x: np.ndarray, shape: AxesLike, axes: AxesLike = None, padding_values: AxesParams = 0) -> np.ndarray:
    """
    Pad a tensor to `shape` along the `axes`.

    Parameters
    ----------
    x
        tensor to pad.
    shape
        final tensor shape.
    padding_values
        values to pad the tensor with.
    axes
        axes along which the tensor will be padded. If None - the last `len(shape)` axes are used.
    """
    old_shape, new_shape = np.array(x.shape), np.array(fill_by_indices(x.shape, shape, axes))
    if (old_shape > new_shape).any():
        raise ValueError(f'The resulting shape cannot be smaller than the original: {old_shape} vs {new_shape}')

    delta = new_shape - old_shape
    padding_width = np.array((delta // 2, (delta + 1) // 2)).T.astype(int)

    return pad(x, padding_width, padding_values)


def slice_to_shape(x: np.ndarray, shape: AxesLike, axes: AxesLike = None) -> np.ndarray:
    """
    Slice a tensor to ``shape`` along ``axes``.

    Parameters
    ----------
    x
        tensor to pad.
    shape
        final tensor shape.
    axes
        axes along which the tensor will be padded. If None - the last `len(shape)` axes are used.
    """
    old_shape, new_shape = np.array(x.shape), np.array(fill_by_indices(x.shape, shape, axes))
    if (old_shape < new_shape).any():
        raise ValueError(f'The resulting shape cannot be greater than the original one: {old_shape} vs {new_shape}')

    start = ((old_shape - new_shape) // 2).astype(int)
    return x[build_slices(start, start + new_shape)]


def proportional_scale_to_shape(x: np.ndarray, shape: AxesLike, axes: AxesLike = None, padding_values: AxesParams = 0,
                                order: int = 1) -> np.ndarray:
    """
    Proportionally scale a tensor to fit ``shape`` along ``axes`` then pad it to that shape.

    Parameters
    ----------
    x
        tensor to pad.
    shape
        final tensor shape.
    axes
        axes along which the tensor will be padded. If None - the last `len(shape)` axes are used.
    padding_values
        values to pad the tensor with.
    order
        order of interpolation.
    """
    axes = expand_axes(axes, shape)
    scale_factor = (np.array(shape, 'float64') / extract(x.shape, axes)).min()
    return pad_to_shape(scale(x, scale_factor, axes, order), shape, axes, padding_values)
