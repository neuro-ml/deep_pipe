from typing import Sequence, Callable

import numpy as np

from dpipe.medim.utils import pam


def sample_ids(sequence: Sequence, weights: Sequence[float] = None):
    """
    Infinitely yield samples from ``sequence`` according to ``weights``.

    Parameters
    ----------
    sequence
    weights
        The weights associated with each element. If None, the weights are assumed to be equal.
    """
    if weights is not None:
        weights = np.asarray(weights)
        assert (weights >= 0).all() and (weights > 0).any(), weights
        weights = weights / weights.sum()

    while True:
        yield np.random.choice(sequence, p=weights)


def load_by_random_id(*loaders: Callable, ids: Sequence, weights: Sequence[float] = None):
    """
    Infinitely yield objects loaded by identifiers randomly sampled from ``ids`` according to ``weights``.

    Parameters
    ----------
    loaders
    ids
    weights
        The weights associated with each id. If None, the weights are assumed to be equal.
    """
    for id_ in sample_ids(ids, weights):
        yield tuple(pam(loaders, id_))
