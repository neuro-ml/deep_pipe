from matplotlib import pyplot as plt
import numpy as np


def slice3d(*data, axis: int = -1, fig_size: int = 5, max_columns: int = None,
            colorbar: bool = False, cmap: str = None, vlim = (None, None)):
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
    # Dirty hack to solve problem between new registry system that scan
    # everything and ipywidgets
    from ipywidgets import interact, IntSlider

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

    interact(update, idx=IntSlider(min=0, max=size-1, continuous_update=False))
