import numpy as np

from .axes import expand_axes, AxesLike
from .utils import apply_along_axes, scale as _scale

# TODO: deprecated 14.01.2019
from .shape_ops import *
from .shape_ops import slice_to_shape, scale_to_shape, scale, proportional_scale_to_shape


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
