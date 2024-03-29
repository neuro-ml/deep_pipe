from contextlib import suppress
from functools import wraps
from itertools import chain
from operator import itemgetter
from threading import Thread
from typing import Iterable, Sized, Union, Callable, Sequence, Any, Tuple
from queue import Queue

import numpy as np


def pam(functions: Iterable[Callable], *args, **kwargs):
    """
    Inverse of `map`. Apply a sequence of callables to fixed arguments.

    Examples
    --------
    >>> list(pam([np.sqrt, np.square, np.cbrt], 64))
    [8, 4096, 4]
    """
    for f in functions:
        yield f(*args, **kwargs)


def zip_equal(*args: Union[Sized, Iterable]) -> Iterable[Tuple]:
    """
    zip over the given iterables, but enforce that all of them exhaust simultaneously.

    Examples
    --------
    >>> zip_equal([1, 2, 3], [4, 5, 6]) # ok
    >>> zip_equal([1, 2, 3], [4, 5, 6, 7]) # raises ValueError
    # ValueError is raised even if the lengths are not known
    >>> zip_equal([1, 2, 3], map(np.sqrt, [4, 5, 6])) # ok
    >>> zip_equal([1, 2, 3], map(np.sqrt, [4, 5, 6, 7])) # raises ValueError
    """
    if not args:
        return

    lengths = []
    all_lengths = []
    for arg in args:
        try:
            lengths.append(len(arg))
            all_lengths.append(len(arg))
        except TypeError:
            all_lengths.append('?')

    if lengths and not all(x == lengths[0] for x in lengths):
        from .checks import join
        raise ValueError(f'The arguments have different lengths: {join(all_lengths)}.')

    iterables = [iter(arg) for arg in args]
    while True:
        result = []
        for it in iterables:
            with suppress(StopIteration):
                result.append(next(it))

        if len(result) != len(args):
            break
        yield tuple(result)

    if len(result) != 0:
        raise ValueError(f'The iterables did not exhaust simultaneously.')


def head_tail(iterable: Iterable) -> Tuple[Any, Iterable]:
    """
    Split the ``iterable`` into the first and the rest of the elements.

    Examples
    --------
    >>> head, tail = head_tail(map(np.square, [1, 2, 3]))
    >>> head, list(tail)
    1, [4, 9]
    """
    iterable = iter(iterable)
    return next(iterable), iterable


def peek(iterable: Iterable) -> Tuple[Any, Iterable]:
    """
    Return the first element from ``iterable`` and the whole iterable.

    Notes
    -----
    The incoming ``iterable`` might be mutated, use the returned iterable instead.

    Examples
    --------
    >>> original_iterable = map(np.square, [1, 2, 3])
    >>> head, iterable = peek(original_iterable)
    >>> head, list(iterable)
    1, [1, 4, 9]
    # list(original_iterable) would return [4, 9]
    """
    head, tail = head_tail(iterable)
    return head, chain([head], tail)


def lmap(func: Callable, *iterables: Iterable) -> list:
    """Composition of list and map."""
    return list(map(func, *iterables))


def pmap(func: Callable, iterable: Iterable, *args, **kwargs) -> Iterable:
    """
    Partial map.
    Maps ``func`` over ``iterable`` using ``args`` and ``kwargs`` as additional arguments.
    """
    for value in iterable:
        yield func(value, *args, **kwargs)


class AsyncPmap:
    def __init__(self, func: Callable, iterable: Iterable, *args, **kwargs) -> None:
        self.__func = func
        self.__iterable = iterable
        self.__args = args
        self.__kwargs = kwargs

        self.__result_queue = Queue(1)
        
        self.__working_thread = Thread(
            target = self._prediction_func
        )
        self.__exhausted = False

    def start(self) -> None:
        self.__working_thread.start()

    def stop(self) -> None:
        self.__working_thread.join()
        assert not self.__working_thread.is_alive()
        self.__exhausted = True

    def _prediction_func(self) -> None:
        try:
            for value in self.__iterable:
                self.__result_queue.put((self.__func(value, *self.__args, **self.__kwargs), True))
            raise StopIteration
        except BaseException as e:
            self.__result_queue.put((e, False))


    def __iter__(self):
        return self

    def __next__(self) -> Any:
        if self.__exhausted:
            raise StopIteration

        obj, success = self.__result_queue.get()

        if not success:
            self.stop()
            raise obj

        return obj


def dmap(func: Callable, dictionary: dict, *args, **kwargs):
    """
    Transform the ``dictionary`` by mapping ``func`` over its values.
    ``args`` and ``kwargs`` are passed as additional arguments.

    Examples
    --------
    >>> dmap(np.square, {'a': 1, 'b': 2})
    {'a': 1, 'b': 4}
    """
    return {k: func(v, *args, **kwargs) for k, v in dictionary.items()}


def zdict(keys: Iterable, values: Iterable) -> dict:
    """Create a dictionary from ``keys`` and ``values``."""
    return dict(zip_equal(keys, values))


def squeeze_first(inputs):
    """Remove the first dimension in case it is singleton."""
    if len(inputs) == 1:
        inputs = inputs[0]
    return inputs


def flatten(iterable: Iterable, iterable_types: Union[tuple, type] = None) -> list:
    """
    Recursively flattens an ``iterable`` as long as it is an instance of ``iterable_types``.

    Examples
    --------
    >>> flatten([1, [2, 3], [[4]]])
    [1, 2, 3, 4]
    >>> flatten([1, (2, 3), [[4]]])
    [1, (2, 3), 4]
    >>> flatten([1, (2, 3), [[4]]], iterable_types=(list, tuple))
    [1, 2, 3, 4]
    """
    if iterable_types is None:
        iterable_types = type(iterable)
    if not isinstance(iterable, iterable_types):
        return [iterable]

    return sum((flatten(value, iterable_types) for value in iterable), [])


def filter_mask(iterable: Iterable, mask: Iterable[bool]) -> Iterable:
    """Filter values from ``iterable`` according to ``mask``."""
    return map(itemgetter(1), filter(itemgetter(0), zip_equal(mask, iterable)))


def extract(sequence: Sequence, indices: Iterable):
    """Extract ``indices`` from ``sequence``."""
    return [sequence[i] for i in indices]


def negate_indices(indices: Iterable, length: int):
    """Return valid indices for a sequence of len ``length`` that are not present in ``indices``."""
    other_indices = np.ones(length, bool)
    other_indices[list(indices)] = False
    return np.where(other_indices)[0]


def make_chunks(iterable: Iterable, chunk_size: int, incomplete: bool = True):
    """
    Group ``iterable`` into chunks of size ``chunk_size``.

    Parameters
    ----------
    iterable
    chunk_size
    incomplete
        whether to yield the last chunk in case it has a smaller size.
    """
    chunk = []
    for value in iterable:
        chunk.append(value)
        if len(chunk) == chunk_size:
            yield tuple(chunk)
            chunk = []

    if incomplete and chunk:
        yield chunk


def collect(func: Callable):
    """
    Make a function that returns a list from a function that returns an iterator.

    Examples
    --------
    >>> @collect
    >>> def squares(n):
    >>>     for i in range(n):
    >>>         yield i ** 2
    >>>
    >>> squares(3)
    [1, 4, 9]
    """

    @wraps(func)
    def wrapper(*args, **kwargs):
        return list(func(*args, **kwargs))

    wrapper.__annotations__['return'] = list
    return wrapper


def stack(axis: int = 0, dtype: np.dtype = None):
    """
    Stack the values yielded by a generator function along a given ``axis``.
    ``dtype`` (if any) determines the data type of the resulting array.

    Examples
    --------
    >>> @stack(1)
    >>> def consecutive(n):
    >>>     for i in range(n):
    >>>         yield i, i+1
    >>>
    >>> consecutive(3)
    array([[0, 1, 2],
           [1, 2, 3]])
    """

    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            value = np.stack(list(func(*args, **kwargs)), axis=axis)
            if dtype is not None:
                value = value.astype(dtype)
            return value

        wrapper.__annotations__['return'] = np.ndarray
        return wrapper

    return decorator


def recursive_conditional_map(xr, f, condition):
    """Walks recursively through iterable data structure ``xr``. Applies ``f`` on objects that satisfy ``condition``."""
    return tuple(f(x) if condition(x) else recursive_conditional_map(x, f, condition) for x in xr)
