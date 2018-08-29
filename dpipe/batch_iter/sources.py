from typing import Sequence, Callable

import numpy as np

from dpipe.medim.utils import pam


def sample_ids(ids: Sequence, weights: Sequence[float] = None):
    """
    Infinitely yield identifiers randomly sampled from `ids` according to `weights`.

    Parameters
    ----------
    ids: Sequence
    weights: Sequence[float]
        The weights associated with each id. If None, the weights are assumed to be equal.
    """
    if weights is not None:
        weights = np.asarray(weights)
        assert (weights >= 0).all() and (weights > 0).any(), weights
        weights = weights / weights.sum()

    while True:
        yield np.random.choice(ids, p=weights)


def load_by_random_id(*loaders: Callable, ids: Sequence, weights: Sequence[float] = None):
    """
    Infinitely yield objects loaded by identifiers randomly sampled from `ids` according to `weights`.

    Parameters
    ----------
    loaders: Callable(identifier)
    ids: Sequence
    weights: Sequence[float]
        The weights associated with each id. If None, the weights are assumed to be equal.
    """
    for id_ in sample_ids(ids, weights):
        yield tuple(pam(loaders, id_))
