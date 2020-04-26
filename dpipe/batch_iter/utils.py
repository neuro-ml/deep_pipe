from typing import Callable, Iterable, Sequence

import numpy as np

from dpipe.im.axes import AxesLike, AxesParams
from dpipe.itertools import lmap, squeeze_first
from dpipe.im import pad_to_shape


def pad_batch_equal(batch, padding_values: AxesParams = 0, ratio: AxesParams = 0.5):
    """
    Pad each element of ``batch`` to obtain a correctly shaped array.

    References
    ----------
    `pad_to_shape`
    """
    max_shapes = np.max(lmap(np.shape, batch), axis=0)
    # if not scalars
    if max_shapes.size != 0:
        batch = [pad_to_shape(x, max_shapes, padding_values=padding_values, ratio=ratio) for x in batch]
    return np.array(batch)


def unpack_args(func: Callable, *args, **kwargs):
    """
    Returns a function that takes an iterable and unpacks it while calling ``func``.

    ``args`` and ``kwargs`` are passed to ``func`` as additional arguments.

    Examples
    --------
    >>> def add(x, y):
    >>>     return x + y
    >>>
    >>> add_ = unpack_args(add)
    >>> add(1, 2) == add_([1, 2])
    >>> True
    """

    def wrapper(xs, *args_, **kwargs_):
        return func(*xs, *args_, *args, **kwargs_, **kwargs)

    return wrapper


def multiply(func: Callable, *args, **kwargs):
    """
    Returns a function that takes an iterable and maps ``func`` over it.
    Useful when multiple batches require the same function.

    ``args`` and ``kwargs`` are passed to ``func`` as additional arguments.
    """

    def wrapped(xs: Iterable, *args_, **kwargs_) -> tuple:
        return tuple(func(x, *args_, *args, **kwargs_, **kwargs) for x in xs)

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

    def wrapped(xs: Sequence, *args_, **kwargs_) -> tuple:
        index_ = {i + len(xs) if i < 0 else i for i in index}
        for idx in index_:
            if idx < 0 or idx >= len(xs):
                raise IndexError(f'Index {idx} out of bounds.')

        return tuple(func(x, *args_, *args, **kwargs_, **kwargs) if i in index_ else x for i, x in enumerate(xs))

    return wrapped


def zip_apply(*functions: Callable, **kwargs):
    """
    Returns a function that takes an iterable and zips ``functions`` over it.

    ``kwargs`` are passed to each function as additional arguments.

    Examples
    --------
    >>> zipper = zip_apply(np.square, np.sqrt)
    >>> zipper([4, 9])
    >>> (16, 3)
    """

    def wrapped(xs: Sequence, *args, **kwargs_) -> tuple:
        return tuple(func(x, *args, **kwargs_, **kwargs) for func, x in zip(functions, xs))

    return wrapped


def random_apply(p: float, func: Callable, *args, **kwargs):
    """
    Returns a function that applies ``func`` with a given probability ``p``.

    ``args`` and ``kwargs`` are passed to ``func`` as additional arguments.
    """

    def wrapped(*args_, **kwargs_):
        if np.random.binomial(1, p):
            return func(*args_, *args, **kwargs_, **kwargs)
        return squeeze_first(args_)

    return wrapped


def sample_args(func: Callable, *args: Callable, **kwargs: Callable):
    """
    Returns a function that samples arguments for ``func`` from ``args`` and ``kwargs``.

    Each argument in ``args`` and ``kwargs`` must be a callable that samples a random value.

    Examples
    --------
    >>> from scipy.ndimage import  rotate
    >>>
    >>> random_rotate = sample_args(rotate, angle=np.random.normal)
    >>> random_rotate(x)
    >>> # same as
    >>> rotate(x, angle=np.random.normal())
    """

    def wrapped(*args_, **kwargs_):
        return func(*args_, *([arg() for arg in args]), **kwargs_, **{name: arg() for name, arg in kwargs.items()})

    return wrapped
