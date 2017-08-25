import numpy as np

import dpipe.externals.pdp.pdp as pdp
from dpipe.medim.slices import iterate_slices


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
        ids, load_x, load_y, batch_size, *, shuffle=False, empty_slice=True):
    if shuffle:
        ids = shuffle_ids(ids)

    def slicer(ids):
        for id_ in ids:
            x, y = load_x(id_), load_y(id_)
            yield from iterate_slices(x, y, empty=empty_slice)

    # @pdp.pack_args
    # def flip(mscan, segm):
    #     mscan = mscan.copy()
    #     segm = segm.copy()
    #     if np.random.randint(0, 2):
    #         mscan = np.flip(mscan, 1)
    #         segm = np.flip(segm, 1)
    #     if np.random.randint(0, 2):
    #         mscan = np.flip(mscan, 2)
    #         segm = np.flip(segm, 2)
    #     # randomly swap channels because why not?
    #     mscan = mscan[np.random.permutation(np.arange(len(mscan)))]
    #     return mscan, segm

    return pdp.Pipeline(
        pdp.Source(slicer(ids), buffer_size=30),
        # pdp.LambdaTransformer(flip, buffer_size=3),
        pdp.Chunker(chunk_size=batch_size, buffer_size=2),
        pdp.LambdaTransformer(pdp.combine_batches, buffer_size=3)
    )


def make_multiple_slices_iter(
        ids, load_x, load_y, batch_size, *, num_slices, shuffle=False):
    if shuffle:
        ids = shuffle_ids(ids)

    def slicer(ids):
        for id_ in ids:
            x, y = load_x(id_), load_y(id_)

            yield from iterate_multiple_slices(x, y, num_slices)

    # @pdp.pack_args
    # def flip(mscan, segm):
    #     mscan = mscan.copy()
    #     segm = segm.copy()
    #     # if np.random.randint(0, 2):
    #     #     mscan = np.flip(mscan, 1)
    #     #     segm = np.flip(segm, 1)
    #     # swap direction
    #     if np.random.randint(0, 2):
    #         mscan = np.flip(mscan, 2)
    #         segm = np.flip(segm, 2)
    #     # randomly swap channels because why not?
    #     # mscan = mscan[np.random.permutation(np.arange(len(mscan)))]
    #     return mscan, segm

    return pdp.Pipeline(
        pdp.Source(slicer(ids), buffer_size=30),
        # pdp.LambdaTransformer(flip, buffer_size=3),
        pdp.Chunker(chunk_size=batch_size, buffer_size=2),
        pdp.LambdaTransformer(pdp.combine_batches, buffer_size=3)
    )
