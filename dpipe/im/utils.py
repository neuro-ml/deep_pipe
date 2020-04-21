import os
import inspect

from .axes import check_axes
from dpipe.itertools import *
from .shape_utils import *


def identity(x):
    return x


# TODO: weaken the restriction to same ndim
def apply_along_axes(func: Callable, x: np.ndarray, axes: AxesLike, *args, **kwargs):
    """
    Apply ``func`` to slices from ``x`` taken along ``axes``.
    ``args`` and ``kwargs`` are passed as additional arguments.

    Notes
    -----
    ``func`` must return an array of the same shape as it received.
    """
    axes = check_axes(axes)
    if len(axes) == x.ndim:
        return func(x)

    other_axes = negate_indices(axes, x.ndim)
    begin = np.arange(len(other_axes))

    y = np.moveaxis(x, other_axes, begin)
    result = np.stack([func(patch, *args, **kwargs) for patch in y.reshape(-1, *extract(x.shape, axes))])
    return np.moveaxis(result.reshape(*y.shape), begin, other_axes)


def build_slices(start: Sequence[int], stop: Sequence[int] = None) -> Tuple[slice, ...]:
    """
    Returns a tuple of slices built from ``start`` and ``stop``.

    Examples
    --------
    >>> build_slices([1, 2, 3], [4, 5, 6])
    (slice(1, 4), slice(2, 5), slice(3, 6))
    >>> build_slices([10, 11])
    (slice(10), slice(11))
    """
    if stop is not None:
        check_len(start, stop)
        return tuple(map(slice, start, stop))

    return tuple(map(slice, start))


def cache_to_disk(func: Callable, path: str, load: Callable, save: Callable) -> Callable:
    """
    Cache a function to disk.

    Parameters
    ----------
    func: Callable
    path: str
        the root folder where the function will be cached.
    load: Callable(path, *args, **kwargs)
        load the value for `func(*args, **kwargs)`.
    save: Callable(value, path, *args, **kwargs)
        save the value for `func(*args, **kwargs)`.
    """
    signature = inspect.signature(func)
    os.makedirs(path, exist_ok=True)

    def get_all_args(args, kwargs):
        bindings = signature.bind(*args, **kwargs)
        bindings.apply_defaults()
        return bindings.args, bindings.kwargs

    @wraps(func)
    def wrapper(*args, **kwargs):
        args, kwargs = get_all_args(args, kwargs)

        try:
            return load(path, *args, **kwargs)
        except FileNotFoundError:
            pass

        value = func(*args, **kwargs)
        save(value, path, *args, **kwargs)
        return value

    return wrapper


def get_random_tuple(low, high, size):
    return tuple(np.random.randint(low, high, size=size, dtype=int))


def composition(func: Callable, *args, **kwargs):
    """
    Applies ``func`` to the output of the decorated function.
    ``args`` and ``kwargs`` are passed as additional positional and keyword arguments respectively.
    """

    def decorator(decorated: Callable):
        @wraps(decorated)
        def wrapper(*args_, **kwargs_):
            return func(decorated(*args_, **kwargs_), *args, **kwargs)

        return wrapper

    return decorator


def name_changed(func: Callable, old_name: str, date: str):
    return np.deprecate(func, old_name=old_name, new_name=func.__name__)
