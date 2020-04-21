import numpy as np
from matplotlib.colors import hsv_to_rgb

from dpipe.im.preprocessing import min_max_scale


def hsv_image(hue, saturation, value):
    """Creates image in HSV format from HSV data."""
    shaped = [field for field in (hue, saturation, value) if hasattr(field, 'shape')]
    shape = shaped[0].shape

    hsv = np.zeros((*shape, 3))
    hsv[..., 0] = hue
    hsv[..., 1] = saturation
    hsv[..., 2] = value
    return hsv


def rgb_from_hsv_data(hue, saturation, value):
    """Creates image in RGB format from HSV data."""
    return hsv_to_rgb(hsv_image(hue, saturation, value))


def gray_image_colored_mask(gray_image, mask, hue):
    """Creates gray image with colored mask. Keeps intensities intact,
    so dark areas on gray image will be hard to see even after colorization."""
    return rgb_from_hsv_data(hue, np.where(mask, 1, 0), min_max_scale(gray_image))


def gray_image_bright_colored_mask(gray_image, mask, hue):
    """Creates gray image with colored mask. Changes mask intensities,
    so dark areas on gray image will be easy to see after colorization."""
    return rgb_from_hsv_data(hue, np.where(mask, 1, 0), np.where(mask, 1, min_max_scale(gray_image)))


def segmentation_probabilities(image, probabilities, hue):
    return hsv_to_rgb(hsv_image(hue, probabilities, image))


def masked_segmentation_probabilities(image, probabilities, hue, mask):
    return hsv_to_rgb(hsv_image(hue, np.where(mask, probabilities, 0), np.where(mask, 1, image)))
