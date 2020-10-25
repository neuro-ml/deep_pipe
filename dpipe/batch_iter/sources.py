from typing import Sequence, Callable, Union

import numpy as np

from bisect import bisect

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
    assert(len(sequence) == len(weights)), 'len(sequence) is not equal to len(weights)'
    
    if not isinstance(random_state, np.random.RandomState):
        random_state = np.random.RandomState(random_state)
        
    if weights is None:
        get_index = lambda: int(random_state.rand()*(len(sequence)))
    else:
        weights = np.asarray(weights)
        assert (weights >= 0).all() and (weights > 0).any(), weights
        weights /= weights.sum()
        weights_sort_args  = np.argsort(weights)
        weights_accum_sort = np.add.accumulate(weights[weights_sort_args])
        get_index = lambda: weights_sort_args[bisect(weights_accum_sort,random_state.rand())]        
        
    while True:
        yield sequence[get_index()]


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
