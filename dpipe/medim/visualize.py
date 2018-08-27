import numpy as np
from ipywidgets import interact, IntSlider
from matplotlib import pyplot as plt
from matplotlib.animation import FuncAnimation

from .hsv import gray_image_colored_mask, gray_image_bright_colored_mask, segmentation_probabilities


def slice3d(*data: np.ndarray, axis: int = -1, figsize: int = 5, max_columns: int = None,
            colorbar: bool = False, show_axes: bool = False, cmap: str = None, vlim=(None, None)):
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
    if any(x.shape[axis] != size for x in data):
        raise ValueError('All the tensors must have the same size along the given axis')

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


def gif3d(*data: np.ndarray, path_to_save: str, axis: int = -1, figsize: int = 5, max_columns: int = None,
          colorbar: bool = False, show_axes: bool = False, cmap: str = None, vlim=(None, None),
          interval = 20, fps: int = 30):
    """
    Creates a GIF image of interactive plot (see slice3d)

    Parameters
    ----------
    data : np.ndarray
    path_to_save: str
    axis : the axis along which the slices will be taken
    figsize : the size of the image of a single slice
    max_columns : the maximal number of figures in a row.
                    None - all figures will be in the same row.
    colorbar : Whether to display a colorbar.
    show_axes: Whether to do display grid on the image.
    cmap,vlim : parameters passed to matplotlib.pyplot.imshow
    interval: parameter passed to matplotlib.animation.FuncAnimation
    fps: int frames per second, passed to matplotlib.animation.FuncAnimation.save
    """

    size = data[0].shape[axis]
    if any(x.shape[axis] != size for x in data):
        raise ValueError('All the tensors must have the same size along the given axis')

    if max_columns is None:
        rows, columns = 1, len(data)
    else:
        columns = min(len(data), max_columns)
        rows = (len(data) - 1) // columns + 1

    fig, axes = plt.subplots(rows, columns, figsize=(figsize * columns, figsize * rows))
    axes = np.array(axes).flatten()

    ims = []
    for ax, x in zip(axes, data):
        im = ax.imshow(x.take(0, axis=axis), cmap=cmap, vmin=vlim[0], vmax=vlim[1], animated=True)
        if colorbar and all(x is not None for x in vlim):
            fig.colorbar(im, ax=ax, orientation='horizontal')
        if not show_axes:
            ax.set_axis_off()
        ims.append(im)

    def update(idx):
        for im, x in zip(ims, data):
            im.set_data(x.take(idx, axis=axis))

        return ims

    plt.tight_layout()

    ani = FuncAnimation(fig, func=update, frames=data[0].shape[axis], interval=interval, blit=True)
    ani.save(path_to_save, writer='imagemagick', fps=fps)