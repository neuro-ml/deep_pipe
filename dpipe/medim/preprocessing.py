import numpy as np
from skimage.measure import label

from .itertools import negate_indices
from .axes import AxesLike, check_axes

from .shape_ops import *


# TODO: docstring
def normalize(x: np.ndarray, mean: bool = True, std: bool = True, percentiles: AxesLike = None,
              axes: AxesLike = None) -> np.ndarray:
    """
    Normalize ``x``'s values to make mean and std independently along ``axes`` equal to 0 and 1 respectively
    (if specified).
    """
    if axes is not None:
        axes = tuple(negate_indices(check_axes(axes), x.ndim))

    robust_values = x
    if percentiles is not None:
        if np.size(percentiles) == 1:
            percentiles = [percentiles, 100 - percentiles]

        bottom, top = np.percentile(x, percentiles, axes, keepdims=True)
        mask = (x < bottom) | (x >= top)
        robust_values = np.ma.masked_array(x, mask=mask)

    if mean:
        x = x - robust_values.mean(axes, keepdims=True)
    if std:
        x = x / robust_values.std(axes, keepdims=True)

    return x


def min_max_scale(x: np.ndarray, axes: AxesLike = None) -> np.ndarray:
    """Scale ``x``'s values so that its minimum and maximum become 0 and 1 respectively
    independently along ``axes``."""
    if axes is not None:
        axes = tuple(negate_indices(check_axes(axes), x.ndim))

    x_min, x_max = x.min(axis=axes, keepdims=True), x.max(axis=axes, keepdims=True)
    return (x - x_min) / (x_max - x_min)


def bytescale(x: np.ndarray) -> np.ndarray:
    """
    Scales ``x``'s values so that its minimum and maximum become 0 and 255 respectively.
    Afterwards converts it to ``uint8``.
    """
    return np.uint8(np.round(255 * min_max_scale(x)))


def describe_connected_components(mask: np.ndarray, background: int = 0, drop_background: bool = True):
    """
    Get the connected components of ``mask`` as well as their labels and volumes.

    Parameters
    ----------
    mask
    background
        the label of the background. The pixels with this label will be marked as the background component
        (even if it is not connected).
    drop_background:
        whether to exclude the background from the returned components' descriptions.

    Returns
    -------
    labeled_mask
        array of the same shape as ``mask``.
    labels
        a list of labels from the ``labeled_mask``. The background label is always 0.
        The labels are sorted according to their corresponding volumes.
    volumes
        a list of corresponding labels' volumes.
    """
    label_map = label(mask, background=background)
    labels, volumes = np.unique(label_map, return_counts=True)
    idx = volumes.argsort()[::-1]
    labels, volumes = labels[idx], volumes[idx]
    if drop_background:
        foreground = labels != 0
        labels, volumes = labels[foreground], volumes[foreground]
    return label_map, labels, volumes


def get_greatest_component(mask: np.ndarray, background: int = 0, drop_background: bool = True) -> np.ndarray:
    """Get the greatest connected component from ``mask``. See `describe_connected_components` for details."""
    label_map, labels, volumes = describe_connected_components(mask, background, drop_background)
    return label_map == labels[0]


# 27.07.2019
@np.deprecate(new_name='normalize')
def normalize_image(image: np.ndarray, mean: bool = True, std: bool = True, drop_percentile: int = None) -> np.ndarray:
    return normalize(image, mean, std, drop_percentile)


@np.deprecate(new_name='normalize')
def normalize_multichannel_image(image: np.ndarray, mean: bool = True, std: bool = True, drop_percentile: int = None):
    return normalize(image, mean, std, drop_percentile, 0)
