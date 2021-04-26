from functools import wraps
from typing import Callable
import numpy as np

from .itertools import extract


def join(values):
    return ", ".join(map(str, values))


def check_shape_along_axis(*arrays, axis):
    sizes = []
    for array in arrays:
        sizes.append(tuple(extract(array.shape, np.core.numeric.normalize_axis_tuple(axis, array.ndim))))

    if any(x != sizes[0] for x in sizes):
        raise ValueError(f'Arrays of equal size along axis {axis} are required: {join(sizes)}')


def check_len(*args):
    lengths = list(map(len, args))
    if any(length != lengths[0] for length in lengths):
        raise ValueError(f'Arguments of equal length are required: {join(lengths)}')


def check_bool(*arrays):
    for i, a in enumerate(arrays):
        assert a.dtype == bool, f'{i}: {a.dtype}'


def check_shapes(*arrays):
    shapes = [array.shape for array in arrays]
    if any(shape != shapes[0] for shape in shapes):
        raise ValueError(f'Arrays of equal shape are required: {join(shapes)}')


def add_check_function(check_function: Callable):
    """Decorator that checks the function's arguments via ``check_function`` before calling it."""

    def decorator(func: Callable):
        @wraps(func)
        def wrapper(*args, **kwargs):
            check_function(*args, *kwargs.values())
            return func(*args, **kwargs)

        return wrapper

    name = getattr(check_function, '__name__', '`func`')
    decorator.__doc__ = f"Check the function's arguments via `{name}` before calling it."
    return decorator


add_check_bool, add_check_shapes, add_check_len = map(add_check_function, [
    check_bool, check_shapes, check_len
])
