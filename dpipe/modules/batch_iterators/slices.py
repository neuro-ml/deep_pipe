import numpy as np

from dpipe.modules.dl.model_cores.enet import iterate_slices
from ..datasets import Dataset
from .utils import combine_batch

from bdp import Pipeline, LambdaTransformer, Source, Chunker


def shuffle_ids(ids):
    return np.random.permutation(ids)


def make_slices_iter(
        ids, dataset: Dataset, batch_size, *, shuffle=False):
    if shuffle:
        ids = shuffle_ids(ids)

    def slicer(ids):
        for id in ids:
            x = dataset.load_x(id)
            y = dataset.load_y(id)
            yield from iterate_slices(x, y, empty=False)

    return Pipeline(
        Source(slicer(ids), buffer_size=10),
        Chunker(chunk_size=batch_size, buffer_size=2),
        LambdaTransformer(combine_batch, n_workers=1, buffer_size=3)
    )
