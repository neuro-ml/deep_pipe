from typing import Callable, Iterable

import numpy as np

from dpipe.medim.axes import AxesLike, AxesParams
from dpipe.medim.itertools import lmap
from dpipe.medim.preprocessing import pad_to_shape


def pad_batch_equal(batch, padding_values: AxesParams = 0, ratio: AxesParams = 0.5):
    """
    Pad each element of ``batch`` to obtain a correctly shaped array.

    See Also
    --------
    `pad_to_shape`
    """
    max_shapes = np.max(lmap(np.shape, batch), axis=0)
    return np.array([pad_to_shape(x, max_shapes, padding_values=padding_values, ratio=ratio) for x in batch])


def multiply(func: Callable, *args, **kwargs):
    """
    Returns a function that takes an iterable and maps ``func`` over it.
    Useful when multiple batches require the same function.

    ``args`` and ``kwargs`` are passed to ``func`` as additional arguments.
    """

    def wrapped(xs: Iterable) -> tuple:
        return tuple(func(x, *args, **kwargs) for x in xs)

    name = getattr(func, '__name__', '`func`')
    wrapped.__doc__ = f"Maps `{name}` over ``xs``."
    return wrapped


def apply_at(index: AxesLike, func: Callable, *args, **kwargs):
    """
    Returns a function that takes an iterable and applies ``func`` to the values at the corresponding ``index``.

    ``args`` and ``kwargs`` are passed to ``func`` as additional arguments.

    Examples
    --------
    >>> first_sqr = apply_at(0, np.square)
    >>> first_sqr([3, 2, 1])
    >>> (9, 2, 1)
    """
    index = set(np.atleast_1d(index).tolist())

    def wrapped(xs: Iterable) -> tuple:
        return tuple(func(x, *args, **kwargs) if i in index else x for i, x in enumerate(xs))

    return wrapped


def random_apply(p: float, func: Callable, *args, **kwargs):
    """
    Returns a function that applies ``func`` with a given probability ``p``.

    ``args`` and ``kwargs`` are passed to ``func`` as additional arguments.
    """

    def wrapped(x):
        if np.random.binomial(1, p):
            x = func(x, *args, **kwargs)
        return x

    return wrapped
