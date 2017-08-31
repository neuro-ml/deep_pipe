import numpy as np

import dpipe.externals.pdp.pdp as pdp


def shuffle_ids(ids):
    return np.random.permutation(ids)


def load_by_ids(load_x, load_y, ids):
    for patient_id in ids:
        yield load_x(patient_id), load_y(patient_id)


def make_simple_iter(
        ids, load_x, load_y, batch_size, *, shuffle=False):
    if shuffle:
        ids = shuffle_ids(ids)

    return pdp.Pipeline(
        pdp.Source(load_by_ids(load_x, load_y, ids), buffer_size=30),
        pdp.Chunker(chunk_size=batch_size, buffer_size=2),
        pdp.LambdaTransformer(pdp.combine_batches, buffer_size=3)
    )
