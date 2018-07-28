from contextlib import suppress
from typing import Sized, Sequence, Iterable, Callable, Union

import numpy as np


def decode_segmentation(x, segm_decoding_matrix) -> np.array:
    assert np.issubdtype(x.dtype, np.integer), f'Segmentation dtype must be int, but {x.dtype} provided'
    return np.rollaxis(segm_decoding_matrix[x], -1)


def build_slices(start, stop):
    assert len(start) == len(stop)
    return tuple(map(slice, start, stop))


def scale(x):
    x_min, x_max = x.min(), x.max()
    return (x - x_min) / (x_max - x_min)


def bytescale(x):
    return np.uint8(np.round(255 * scale(x)))


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
        Loaders for x and y
    ids: Sequence
        a sequence of ids to load
    shuffle: bool, optional
        whether to shuffle the ids before yielding
    """
    if shuffle:
        ids = np.random.permutation(ids)
    for identifier in ids:
        yield squeeze_first(tuple(loader(identifier) for loader in loaders))


def pam(functions: Iterable[Callable], *args, **kwargs):
    """Inverse of `map`. Apply a sequence of callables to fixed arguments."""
    for f in functions:
        yield f(*args, **kwargs)


def zip_equal(*args: Union[Sized, Iterable]):
    if not args:
        return

    lengths = []
    for arg in args:
        with suppress(TypeError):
            lengths.append(len(arg))

    if lengths and not all(x == lengths[0] for x in lengths):
        raise ValueError('The arguments have different lengths.')

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


def squeeze_first(inputs):
    """Remove the first dimension in case it is singleton."""
    if len(inputs) == 1:
        inputs = inputs[0]
    return inputs


def flatten(iterable: Iterable, iterable_types: tuple = None) -> list:
    if iterable_types is None:
        iterable_types = type(iterable)
    result = []
    for value in iterable:
        if isinstance(value, iterable_types):
            result.extend(flatten(value, iterable_types))
        else:
            result.append(value)
    return result


def pad(x, padding, padding_values):
    padding = np.broadcast_to(padding, [x.ndim, 2])

    new_shape = np.array(x.shape) + np.sum(padding, axis=1)
    new_x = np.zeros(new_shape, dtype=x.dtype)
    new_x[:] = padding_values

    start = padding[:, 0]
    end = np.where(padding[:, 1] != 0, -padding[:, 1], None)
    new_x[build_slices(start, end)] = x
    return new_x


def add_first_dim(x):
    return x[None]


# Legacy
add_batch_dim = np.deprecate(add_first_dim, old_name='add_batch_dim', new_name='add_first_dim')
