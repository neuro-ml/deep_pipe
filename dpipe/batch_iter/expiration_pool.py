from typing import Iterable
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

    def __init__(self, pool_size: int, repetitions: int, iterations: int = 1):
        super().__init__(partial(expiration_pool, pool_size=pool_size, repetitions=repetitions, iterations=iterations))


def expiration_pool(iterable: Iterable, pool_size: int, repetitions: int, iterations: int = 1):
    """
    Caches ``pool_size`` items from ``iterable``.
    The item is removed from cache after it was generated ``repetitions`` times.
    After an item is removed, a new one is extracted from the ``iterable``.
    Finally, ``iterations`` controls how many values are generated after a new value is added,
    thus speeding up the pipeline at early stages.
    """

    assert pool_size > 0
    assert repetitions > 0
    iterable = enumerate(iterable)

    def sample_value():
        # TODO: use randomdict?
        idx = np.random.choice(list(value_frequency))
        value, frequency = value_frequency[idx]
        frequency += 1
        if frequency == repetitions:
            del value_frequency[idx]
        else:
            value_frequency[idx] = [value, frequency]
        return value

    value_frequency = {}  # i -> [value, frequency]
    for idx, value in iterable:
        value_frequency[idx] = [value, 0]

        for _ in range(iterations):
            if value_frequency:
                yield sample_value()

        while len(value_frequency) >= pool_size:
            yield sample_value()

    while len(value_frequency):
        yield sample_value()
