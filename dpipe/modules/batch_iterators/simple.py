import numpy as np

from ..datasets import Dataset
from .utils import combine_batch
import dpipe.external.pdp.pdp as pdp


def shuffle_ids(ids):
    return np.random.permutation(ids)


def load_by_ids(dataset, ids):
    for patient_id in ids:
        yield dataset.load_x(patient_id), dataset.load_y(patient_id)


def make_simple_iter(
        ids, dataset: Dataset, batch_size, *, shuffle=False):
    if shuffle:
        ids = shuffle_ids(ids)

    return pdp.Pipeline(
        pdp.Source(load_by_ids(dataset, ids), buffer_size=30),
        pdp.Chunker(chunk_size=batch_size, buffer_size=2),
        pdp.LambdaTransformer(combine_batch, buffer_size=3)
    )
