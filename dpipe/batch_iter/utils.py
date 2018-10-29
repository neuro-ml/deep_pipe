from typing import Callable, Iterable

import numpy as np

from dpipe.medim.itertools import lmap
from dpipe.medim.preprocessing import pad_to_shape


def pad_batch_equal(batch, padding_values=0):
    """Pad each element of ``batch`` to obtain a correctly shaped array."""
    max_shapes = np.max(lmap(np.shape, batch), axis=0)
    return np.array([pad_to_shape(x, max_shapes, padding_values=padding_values) for x in batch])


def multiply(func: Callable):
    """
    Returns a functions that takes an iterable and maps ``func`` over it.
    Useful when multiple batches require the same function.
    """

    def wrapped(x: Iterable):
        return tuple(map(func, x))

    name = getattr(func, '__name__', '`func`')
    wrapped.__doc__ = f"Maps `{name}` over ``x``."
    return wrapped
