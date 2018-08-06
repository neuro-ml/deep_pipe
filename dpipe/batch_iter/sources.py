import numpy as np

from dpipe.medim.utils import pam


def sample_ids(ids, weights=None):
    if weights is not None:
        weights = np.asarray(weights)
        assert (weights >= 0).all() and (weights > 0).any(), weights
        weights = weights / weights.sum()

    while True:
        yield np.random.choice(ids, p=weights)


def load_by_random_id(*loaders, ids, weights=None):
    for id_ in sample_ids(ids, weights):
        yield tuple(pam(loaders, id_))
