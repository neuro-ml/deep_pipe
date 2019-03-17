import os
import inspect

from .axes import check_axes, AxesLike
from .checks import check_len
from .itertools import *


def decode_segmentation(x, segm_decoding_matrix) -> np.array:
    assert np.issubdtype(x.dtype, np.integer), f'Segmentation dtype must be int, but {x.dtype} provided'
    return np.rollaxis(segm_decoding_matrix[x], -1)


# TODO: weaken the restriction to same ndim
def apply_along_axes(func: Callable, x: np.ndarray, axes: AxesLike):
    """
    Apply ``func`` to slices from ``x`` taken along ``axes``.

    Parameters
    ----------
    func
         function to apply. Note that both the input and output must have the same shape.
    x
    axes
    """
    axes = check_axes(axes)
    if len(axes) == x.ndim:
        return func(x)

    other_axes = negate_indices(axes, x.ndim)
    begin = np.arange(len(other_axes))

    y = np.moveaxis(x, other_axes, begin)
    result = np.stack(map(func, y.reshape(-1, *extract(x.shape, axes))))
    return np.moveaxis(result.reshape(*y.shape), begin, other_axes)


def extract_dims(array, ndim=1):
    """Decrease the dimensionality of ``array`` by extracting ``ndim`` leading singleton dimensions."""
    for _ in range(ndim):
        assert len(array) == 1
        array = array[0]
    return array


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


def scale(x):
    x_min, x_max = x.min(), x.max()
    return (x - x_min) / (x_max - x_min)


# TODO: doc
def bytescale(x):
    return np.uint8(np.round(255 * scale(x)))


# TODO: doc
def makedirs_top(path, mode=0o777, exist_ok=False):
    """Creates a parent folder if any. See `os.makedirs` for details."""
    folder = os.path.dirname(path)
    if folder:
        os.makedirs(folder, mode, exist_ok)


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


# 07.02.2019
@np.deprecate
def load_by_ids(*loaders: Callable, ids: Sequence, shuffle: bool = False):
    """
    Yields tuples of objects given their ``loaders`` and ``ids``.

    Parameters
    ----------
    loaders: Callable(id)
    ids
    shuffle
        whether to shuffle the ids before yielding.
    """
    if shuffle:
        ids = np.random.permutation(ids)
    for identifier in ids:
        yield squeeze_first(tuple(pam(loaders, identifier)))


def pad(x, padding, padding_values):
    # TODO it might be dangerous
    padding = np.broadcast_to(padding, [x.ndim, 2])

    new_shape = np.array(x.shape) + np.sum(padding, axis=1)
    new_x = np.zeros(new_shape, dtype=x.dtype)
    new_x[:] = padding_values

    start = padding[:, 0]
    end = np.where(padding[:, 1] != 0, -padding[:, 1], None)
    new_x[build_slices(start, end)] = x
    return new_x


def get_random_tuple(low, high, size):
    return tuple(np.random.randint(low, high, size=size, dtype=int))


def unpack_args(func: Callable):
    @wraps(func)
    def wrapper(argument):
        return func(*argument)

    return wrapper


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
