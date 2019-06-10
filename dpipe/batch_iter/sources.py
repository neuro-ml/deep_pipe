from typing import Sequence, Callable

import numpy as np

from dpipe.medim.utils import pam, squeeze_first

__all__ = 'sample', 'load_by_random_id'


def sample(sequence: Sequence, weights: Sequence[float] = None):
    """
    Infinitely yield samples from ``sequence`` according to ``weights``.

    Parameters
    ----------
    sequence: Sequence
        the sequence of elements to sample from.
    weights: Sequence[float], None, optional
        the weights associated with each element. If ``None``, the weights are assumed to be equal.
        Should be the same size as ``sequence``.
    """
    if weights is not None:
        weights = np.asarray(weights)
        assert (weights >= 0).all() and (weights > 0).any(), weights
        weights = weights / weights.sum()

    while True:
        yield np.random.choice(sequence, p=weights)


def load_by_random_id(*loaders: Callable, ids: Sequence, weights: Sequence[float] = None):
    """
    Infinitely yield objects loaded by ``loaders`` according to the identifier from ``ids``.
    The identifiers are randomly sampled from ``ids`` according to the ``weights``.

    Parameters
    ----------
    loaders: Callable
        function, which loads object by its id.
    ids: Sequence
        the sequence of identifiers to sample from.
    weights: Sequence[float], None, optional
        The weights associated with each id. If ``None``, the weights are assumed to be equal.
        Should be the same size as ``ids``.
    """
    for id_ in sample(ids, weights):
        yield squeeze_first(tuple(pam(loaders, id_)))
