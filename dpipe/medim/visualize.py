import numpy as np
from ipywidgets import interact, IntSlider
from matplotlib import pyplot as plt

from .hsv import gray_image_colored_mask, gray_image_bright_colored_mask, segmentation_probabilities


def slice3d(*data: np.ndarray, axis: int = -1, figsize: int = 5, max_columns: int = None,
            colorbar: bool = False, show_axes: bool = True, cmap: str = None, vlim=(None, None)):
    """
    Creates an interactive plot, simultaneously showing slices along a given
    axis for all the passed images.

    Parameters
    ----------
    data : np.ndarray
    axis : the axis along which the slices will be taken
    figsize : the size of the image of a single slice
    max_columns : the maximal number of figures in a row.
                    None - all figures will be in the same row.
    colorbar : Whether to display a colorbar.
    show_axes: Whether to do display grid on the image.
    cmap,vlim : parameters passed to matplotlib.pyplot.imshow
    """
    size = data[0].shape[axis]
    assert all(x.shape[axis] == size for x in data)

    if max_columns is None:
        rows, columns = 1, len(data)
    else:
        columns = min(len(data), max_columns)
        rows = (len(data) - 1) // columns + 1

    def update(idx):
        fig, axes = plt.subplots(rows, columns, figsize=(figsize * columns, figsize * rows))
        axes = np.array(axes).flatten()
        for ax, x in zip(axes, data):
            im = ax.imshow(x.take(idx, axis=axis), cmap=cmap, vmin=vlim[0], vmax=vlim[1])
            if colorbar:
                fig.colorbar(im, ax=ax, orientation='horizontal')
            if not show_axes:
                ax.set_axis_off()
        plt.tight_layout()
        plt.show()

    interact(update, idx=IntSlider(min=0, max=size - 1, continuous_update=False))
