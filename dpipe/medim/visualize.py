from matplotlib import pyplot as plt
from ipywidgets import interact


def slice3d(*data, axis: int = -1, fig_size: int = 5):
    """
    Creates an interactive plot, simultaneously showing slices along a given
    axis for all the passes images.

    :param data: list of numpy arrays
    :param axis: the axis along which the slices will be taken
    :param fig_size: the size of the image of a single slice
    """
    size = data[0].shape[axis]
    for x in data:
        assert x.shape[axis] == size
    plots = len(data)

    def update(idx):
        fig, axes = plt.subplots(1, plots, figsize=(fig_size * plots, fig_size))
        if plots == 1:
            axes = [axes]
        for ax, x in zip(axes, data):
            ax.imshow(x.take(idx, axis=axis))
        plt.show()

    interact(update, idx=(0, size - 1))
