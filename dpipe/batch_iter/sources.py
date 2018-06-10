import numpy as np

from dpipe.medim.utils import squeeze_first


def sample_random_id(ids):
    while True:
        yield np.random.choice(ids)


def load_by_random_id(*loaders, ids):
    for id_ in sample_random_id(ids):
        yield tuple(loader(id_) for loader in loaders)
