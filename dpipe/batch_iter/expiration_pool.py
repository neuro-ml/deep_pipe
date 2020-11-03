from collections import Iterable
from functools import partial

import numpy as np

from .pipeline import Iterator


class ExpirationPool(Iterator):
    """
    A simple expiration pool for time consuming operations that don't fit into RAM.
    See `expiration_pool` for details.

    Examples
    --------
    >>> batch_iter = Infinite(
        # ... some expensive operations, e.g. loading from disk, or preprocessing
        ExpirationPool(pool_size, repetitions),
        # ... here are the values from pool
        # ... other lightweight operations
        # ...
    )
    """

    def __init__(self, pool_size: int, repetitions: int):
        super().__init__(partial(expiration_pool, pool_size=pool_size, repetitions=repetitions), )


def expiration_pool(iterable: Iterable, pool_size: int, repetitions: int):
    """
    Caches ``pool_size`` items from ``iterable``.
    The item is removed from cache after it was generated ``repetitions`` times.
    After an item is removed, a new one is extracted from the ``iterable``.
    """

    assert pool_size > 0
    assert repetitions > 0

    def sample_value():
        # TODO: use randomdict?
        idx = np.random.randint(0, len(freq))
        val = list(freq)[idx]
        freq[val] += 1
        if freq[val] == repetitions:
            del freq[val]
        return val

    freq = {}
    for value in iterable:
        freq[value] = 0
        yield sample_value()

        while len(freq) >= pool_size:
            yield sample_value()

    while len(freq):
        yield sample_value()
