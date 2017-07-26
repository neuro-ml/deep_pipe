import numpy as np

from ..datasets import Dataset
from dpipe.medim.slices import iterate_slices
from .utils import combine_batch
import dpipe.external.pdp.pdp as pdp


# TODO: this is remarkably terrible ^_^
def iterate_multiple_slices(x, y=None, num_slices=1):
    if num_slices == 1:
        if y is None:
            yield from iterate_slices(x)
        else:
            yield from iterate_slices(x, y)
        return

    output_shape = list(x.shape)
    size = output_shape.pop()
    output_shape[0] *= num_slices
    for i in range(size):
        begin = max(0, i - num_slices // 2)
        end = min(size, i + (num_slices + 1) // 2)
        stack = []
        output = np.zeros(output_shape)
        for idx in range(begin, end):
            stack.extend(x[..., idx])
        stack = np.stack(stack)
        #         choosing where to pad:
        if begin != 0:
            output[:len(stack)] = stack
        else:
            output[-len(stack):] = stack
        if y is None:
            yield output
        else:
            yield output, y[..., i]


def shuffle_ids(ids):
    return np.random.permutation(ids)


def make_slices_iter(
        ids, dataset: Dataset, batch_size, *, shuffle=False, empty_slice=True):
    if shuffle:
        ids = shuffle_ids(ids)

    def slicer(ids):
        for id in ids:
            x = dataset.load_x(id)
            y = dataset.load_y(id)
            yield from iterate_slices(x, y, empty=empty_slice)

    # @pdp.pack_args
    # def flip(x, y):
    #     x = x.copy()
    #     y = y.copy()
    #     if np.random.randint(0, 2):
    #         x = np.flip(x, 1)
    #         y = np.flip(y, 1)
    #     if np.random.randint(0, 2):
    #         x = np.flip(x, 2)
    #         y = np.flip(y, 2)
    #     # randomly swap channels because why not?
    #     x = x[np.random.permutation(np.arange(len(x)))]
    #     return x, y

    return pdp.Pipeline(
        pdp.Source(slicer(ids), buffer_size=30),
        # pdp.LambdaTransformer(flip, buffer_size=3),
        pdp.Chunker(chunk_size=batch_size, buffer_size=2),
        pdp.LambdaTransformer(combine_batch, buffer_size=3)
    )


def make_multiple_slices_iter(
        ids, dataset: Dataset, batch_size, *, num_slices, shuffle=False):
    if shuffle:
        ids = shuffle_ids(ids)

    def slicer(ids):
        for id in ids:
            x = dataset.load_x(id)
            y = dataset.load_y(id)

            yield from iterate_multiple_slices(x, y, num_slices)

    # @pdp.pack_args
    # def flip(x, y):
    #     x = x.copy()
    #     y = y.copy()
    #     # if np.random.randint(0, 2):
    #     #     x = np.flip(x, 1)
    #     #     y = np.flip(y, 1)
    #     # swap direction
    #     if np.random.randint(0, 2):
    #         x = np.flip(x, 2)
    #         y = np.flip(y, 2)
    #     # randomly swap channels because why not?
    #     # x = x[np.random.permutation(np.arange(len(x)))]
    #     return x, y

    return pdp.Pipeline(
        pdp.Source(slicer(ids), buffer_size=30),
        # pdp.LambdaTransformer(flip, buffer_size=3),
        pdp.Chunker(chunk_size=batch_size, buffer_size=2),
        pdp.LambdaTransformer(combine_batch, buffer_size=3)
    )
