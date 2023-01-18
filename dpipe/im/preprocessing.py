import numpy as np
from imops.measure import label

from dpipe.itertools import negate_indices
from .axes import AxesLike, check_axes, AxesParams

__all__ = [
    'normalize', 'min_max_scale', 'bytescale',
    'describe_connected_components', 'get_greatest_component',
]


def normalize(x: np.ndarray, mean: bool = True, std: bool = True, percentiles: AxesParams = None,
              axis: AxesLike = None, dtype=None) -> np.ndarray:
    """
    Normalize ``x``'s values to make mean and std independently along ``axes`` equal to 0 and 1 respectively
    (if specified).

    Parameters
    ----------
    x
    mean
        whether to make mean == zero
    std
        whether to make std == 1
    percentiles
        if pair (a, b) - the percentiles between which mean and/or std will be estimated
        if scalar (s) - same as (s, 100 - s)
        if None - same as (0, 100).
    axis
        axes along which mean and/or std will be estimated independently.
        If None - the statistics will be estimated globally.
    dtype
        the dtype of the output.
    """
    if axis is not None:
        axis = tuple(negate_indices(check_axes(axis), x.ndim))

    robust_values = x
    if percentiles is not None:
        if np.size(percentiles) == 1:
            percentiles = [percentiles, 100 - percentiles]

        bottom, top = np.percentile(x, percentiles, axis, keepdims=True)
        mask = (x < bottom) | (x > top)
        robust_values = np.ma.masked_array(x, mask=mask)

    if mean:
        x = x - robust_values.mean(axis, keepdims=True)
    if std:
        x = x / robust_values.std(axis, keepdims=True)

    x = np.ma.filled(x, np.nan)
    if dtype is not None:
        x = x.astype(dtype)
    return x


def min_max_scale(x: np.ndarray, axis: AxesLike = None) -> np.ndarray:
    """
    Scale ``x``'s values so that its minimum and maximum become 0 and 1 respectively
    independently along ``axes``.
    """
    if axis is not None:
        axis = tuple(negate_indices(check_axes(axis), x.ndim))

    x_min, x_max = x.min(axis=axis, keepdims=True), x.max(axis=axis, keepdims=True)
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
    label_map, labels, volumes = label(mask, background=background, return_labels=True, return_sizes=True)

    if not drop_background:
        # background's label is always 0
        labels = np.append(labels, 0)
        volumes = np.append(volumes, label_map.size - volumes.sum(dtype=int))

    idx = volumes.argsort()[::-1]
    labels, volumes = labels[idx], volumes[idx]

    return label_map, labels, volumes


def get_greatest_component(mask: np.ndarray, background: int = 0, drop_background: bool = True) -> np.ndarray:
    """Get the greatest connected component from ``mask``. See `describe_connected_components` for details."""
    label_map, labels, volumes = describe_connected_components(mask, background, drop_background)
    if not len(labels):
        raise ValueError('Argument ``mask`` should contain non-background values if ``drop_background`` is True.')

    return label_map == labels[0]
