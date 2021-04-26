import os
import inspect

from .axes import check_axes, AxesParams
from ..itertools import *
from .shape_utils import *


def identity(x):
    return x


# TODO: weaken the restriction to same ndim
def apply_along_axes(func: Callable, x: np.ndarray, axis: AxesLike, *args, **kwargs):
    """
    Apply ``func`` to slices from ``x`` taken along ``axes``.
    ``args`` and ``kwargs`` are passed as additional arguments.

    Notes
    -----
    ``func`` must return an array of the same shape as it received.
    """
    axis = check_axes(axis)
    if len(axis) == x.ndim:
        return func(x)

    other_axes = negate_indices(axis, x.ndim)
    begin = np.arange(len(other_axes))

    y = np.moveaxis(x, other_axes, begin)
    result = np.stack([func(patch, *args, **kwargs) for patch in y.reshape(-1, *extract(x.shape, axis))])
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


def get_mask_volume(mask: np.ndarray, *spacing: AxesParams, location: bool = False) -> float:
    """
    Calculates the ``mask`` volume given its spatial ``spacing``.

    Parameters
    ----------
    mask
    spacing
        each value represents the spacing for the corresponding axis.
        If float - the values are uniformly spaced along this axis.
        If Sequence[float] - the values are non-uniformly spaced.
    location
        whether to interpret the Sequence[float] in ``spacing`` as values' locations or spacings.
        If ``True`` - the deltas are used as spacings.
    """
    from . import pad

    assert len(spacing) == mask.ndim
    assert mask.dtype == bool

    def _to_deltas(locations):
        ratio = 0.5  # to we need to customize this?
        deltas = np.abs(np.diff(locations))
        deltas = pad(deltas, 1, padding_values=deltas.min())
        deltas = ratio * deltas[:-1] + (1 - ratio) * deltas[1:]
        return deltas

    even = []
    weight = 1
    mesh = np.array(1)
    for axis, value in enumerate(spacing):
        value = np.asarray(value)
        if value.size == 1:
            weight *= value
            even.append(axis)
        else:
            assert mask.shape[axis] == len(value)
            if location:
                value = _to_deltas(value)

            mesh = mesh[..., None] * value

    mask = mask.sum(tuple(even))
    assert mask.shape == mesh.shape
    return mask.flatten() @ mesh.flatten() * weight


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


# this function is too general, so nobody uses it
@np.deprecate
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
