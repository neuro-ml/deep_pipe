from contextlib import suppress
from functools import wraps
from operator import itemgetter
from typing import Iterable, Sized, Union, Callable


def pam(functions: Iterable[Callable], *args, **kwargs):
    """Inverse of `map`. Apply a sequence of callables to fixed arguments."""
    for f in functions:
        yield f(*args, **kwargs)


def zip_equal(*args: Union[Sized, Iterable]):
    """zip over the given iterables, but enforce that all of them exhaust simultaneously."""
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
        raise ValueError(f'The arguments have different lengths: {", ".join(map(str, all_lengths))}.')

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


def lmap(func: Callable, *iterables: Iterable) -> list:
    """Composition of list and map."""
    return list(map(func, *iterables))


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
