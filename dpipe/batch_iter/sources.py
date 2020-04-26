from typing import Sequence, Callable, Union

import numpy as np

from dpipe.itertools import pam, squeeze_first

__all__ = 'sample', 'load_by_random_id'


def sample(sequence: Sequence, weights: Sequence[float] = None, random_state: Union[np.random.RandomState, int] = None):
    """
    Infinitely yield samples from ``sequence`` according to ``weights``.

    Parameters
    ----------
    sequence: Sequence
        the sequence of elements to sample from.
    weights: Sequence[float], None, optional
        the weights associated with each element. If ``None``, the weights are assumed to be equal.
        Should be the same size as ``sequence``.
    random_state
        if not None - used to set the random seed for reproducibility reasons.
    """
    if weights is not None:
        weights = np.asarray(weights)
        assert (weights >= 0).all() and (weights > 0).any(), weights
        weights = weights / weights.sum()

    if not isinstance(random_state, np.random.RandomState):
        random_state = np.random.RandomState(random_state)

    while True:
        index = random_state.choice(len(sequence), p=weights)
        yield sequence[index]


def load_by_random_id(*loaders: Callable, ids: Sequence, weights: Sequence[float] = None,
                      random_state: Union[np.random.RandomState, int] = None):
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
    random_state
        if not None - used to set the random seed for reproducibility reasons.
    """
    for id_ in sample(ids, weights, random_state):
        yield squeeze_first(tuple(pam(loaders, id_)))
