import os
from functools import wraps
from typing import Sequence
import inspect

from .types import AxesLike
from .shape_utils import check_axes
from .checks import add_check_len
from .itertools import *


def decode_segmentation(x, segm_decoding_matrix) -> np.array:
    assert np.issubdtype(x.dtype, np.integer), f'Segmentation dtype must be int, but {x.dtype} provided'
    return np.rollaxis(segm_decoding_matrix[x], -1)


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

    y = np.moveaxis(x, other_axes, begin).reshape(-1, *extract(x.shape, axes))
    result = np.stack(map(func, y)).reshape(*x.shape)
    return np.moveaxis(result, begin, other_axes)


def extract_dims(array, ndims=1):
    """Decrease the dimensionality of `array` by extracting `ndim` leading singleton dimensions."""
    for _ in range(ndims):
        assert len(array) == 1
        array = array[0]
    return array


@add_check_len
def build_slices(start, stop):
    return tuple(map(slice, start, stop))


def scale(x):
    x_min, x_max = x.min(), x.max()
    return (x - x_min) / (x_max - x_min)


def bytescale(x):
    return np.uint8(np.round(255 * scale(x)))


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


def load_image(path: str):
    """
    Load an image located at `path`.
    The following extensions are supported:
        npy, tif, hdr, img, nii, nii.gz
    """
    if path.endswith('.npy'):
        return np.load(path)
    if path.endswith(('.nii', '.nii.gz', '.hdr', '.img')):
        import nibabel as nib
        return nib.load(path).get_data()
    if path.endswith('.tif'):
        from PIL import Image
        with Image.open(path) as image:
            return np.asarray(image)
    if path.endswith(('.png', '.jpg')):
        from imageio import imread
        return imread(path)
    raise ValueError(f"Couldn't read image from path: {path}.\nUnknown file extension.")


def load_by_ids(*loaders: Callable, ids: Sequence, shuffle: bool = False):
    """
    Yields tuples of objects given their loaders and ids.

    Parameters
    ----------
    loaders: Callable(id)
    ids: Sequence
        a sequence of ids to load
    shuffle: bool, optional
        whether to shuffle the ids before yielding
    """
    if shuffle:
        ids = np.random.permutation(ids)
    for identifier in ids:
        yield squeeze_first(tuple(loader(identifier) for loader in loaders))


def zdict(keys: Iterable, values: Iterable) -> dict:
    """Create a `dict` from ``keys`` and ``values``."""
    return dict(zip_equal(keys, values))


def pad(x, padding, padding_values):
    padding = np.broadcast_to(padding, [x.ndim, 2])

    new_shape = np.array(x.shape) + np.sum(padding, axis=1)
    new_x = np.zeros(new_shape, dtype=x.dtype)
    new_x[:] = padding_values

    start = padding[:, 0]
    end = np.where(padding[:, 1] != 0, -padding[:, 1], None)
    new_x[build_slices(start, end)] = x
    return new_x
