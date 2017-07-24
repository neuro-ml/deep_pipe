from matplotlib import pyplot as plt
from ipywidgets import interact, IntSlider
import numpy as np


def slice3d(*data, axis: int = -1, fig_size: int = 5, max_columns=None):
    """
    Creates an interactive plot, simultaneously showing slices along a given
    axis for all the passes images.

    :param data: list of numpy arrays
    :param axis: the axis along which the slices will be taken
    :param fig_size: the size of the image of a single slice
    :param max_columns: the maximal number of figures in a row.
                    None - all figures will be in the same row.
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
        fig, axes = plt.subplots(rows, columns,
                                 figsize=(fig_size * columns, fig_size * rows))
        axes = np.array(axes).flatten()
        for ax, x in zip(axes, data):
            im = ax.imshow(x.take(idx, axis=axis))
            fig.colorbar(im, ax=ax, orientation='horizontal')
        plt.tight_layout()
        plt.show()

    interact(update, idx=IntSlider(min=0, max=size-1, continuous_update=False))
