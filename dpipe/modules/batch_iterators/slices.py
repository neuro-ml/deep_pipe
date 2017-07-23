import numpy as np

from ..datasets import Dataset
from dpipe.medim.slices import iterate_slices
from .utils import combine_batch
import dpipe.external.pdp.pdp as pdp


def shuffle_ids(ids):
    return np.random.permutation(ids)


def make_slices_iter(
        ids, dataset: Dataset, batch_size, *, shuffle=False):
    if shuffle:
        ids = shuffle_ids(ids)
        print(ids)

    def slicer(ids):
        for id in ids:
            x = dataset.load_x(id)
            y = dataset.load_y(id)
            yield from iterate_slices(x, y, empty=False)

    return pdp.Pipeline(
        pdp.Source(slicer(ids), buffer_size=30),
        pdp.Chunker(chunk_size=batch_size, buffer_size=2),
        pdp.LambdaTransformer(combine_batch, buffer_size=3)
    )
