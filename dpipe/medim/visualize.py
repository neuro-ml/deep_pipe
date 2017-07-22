from matplotlib import pyplot as plt
import numpy as np
from ipywidgets import interact


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
            ax.imshow(x.take(idx, axis=axis))
        plt.show()

    interact(update, idx=(0, size - 1))
