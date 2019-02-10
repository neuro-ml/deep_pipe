import numpy as np
from skimage.measure import label

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
        whether to exclude the background from the returned components descriptions.

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


def get_greatest_component(mask: np.ndarray, background: int = 0, drop_background: bool = True):
    """Get the greatest connected component from ``mask``. See `describe_connected_components` for details."""
    label_map, labels, volumes = describe_connected_components(mask, background, drop_background)
    return label_map == labels[0]
