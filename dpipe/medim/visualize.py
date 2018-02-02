import numpy as np
from matplotlib import pyplot as plt
from ipywidgets import interact, IntSlider
from matplotlib.colors import hsv_to_rgb


def slice3d(*data, axis: int = -1, fig_size: int = 5, max_columns: int = None,
            colorbar: bool = False, cmap: str = None, vlim=(None, None)):
    """
    Creates an interactive plot, simultaneously showing slices along a given
    axis for all the passed images.

    Parameters
    ----------
    data : list of numpy arrays
    axis : the axis along which the slices will be taken
    fig_size : the size of the image of a single slice
    max_columns : the maximal number of figures in a row.
                    None - all figures will be in the same row.
    colorbar : Whether to display a colorbar.
    cmap : matplotlib cmap
    """
    size = data[0].shape[axis]
    for x in data:
        assert x.shape[axis] == size
    if max_columns is None:
        rows, columns = 1, len(data)
    else:
        columns = min(len(data), max_columns)
        rows = (len(data) - 1) // columns + 1

    def update(idx):
        fig, axes = plt.subplots(rows, columns, figsize=(fig_size * columns, fig_size * rows))
        axes = np.array(axes).flatten()
        for ax, x in zip(axes, data):
            im = ax.imshow(x.take(idx, axis=axis), cmap=cmap, vmin=vlim[0], vmax=vlim[1])
            if colorbar:
                fig.colorbar(im, ax=ax, orientation='horizontal')
        plt.tight_layout()
        plt.show()

    interact(update, idx=IntSlider(min=0, max=size - 1, continuous_update=False))


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
    return rgb_from_hsv_data(hue, np.where(mask, 1, 0), gray_image)


def gray_image_bright_colored_mask(gray_image, mask, hue):
    """Creates gray image with colored mask. Changes mask intensities,
    so dark areas on gray image will be easy to see after colorization."""
    return rgb_from_hsv_data(hue, np.where(mask, 1, 0), np.where(mask, 1, gray_image))


def segmentation_probabilities(image, probabilities, hue):
    return hsv_to_rgb(hsv_image(hue, probabilities, image))
