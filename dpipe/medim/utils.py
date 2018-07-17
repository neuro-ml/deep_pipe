from collections import Sized
from typing import Sequence, Iterable, Callable

import numpy as np


def decode_segmentation(x, segm_decoding_matrix) -> np.array:
    assert np.issubdtype(x.dtype, np.integer), f'Segmentation dtype must be int, but {x.dtype} provided'
    return np.rollaxis(segm_decoding_matrix[x], -1)


def build_slices(start, stop):
    assert len(start) == len(stop)
    return tuple(map(slice, start, stop))


def get_axes(axes, ndim):
    if axes is None:
        axes = range(-ndim, 0)
    return list(axes)


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

    Parameters
    ----------
    path: str
        Path to the image.

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
    raise ValueError(f"Couldn't read image from path: {path}.\n"
                     "Unknown file extension.")


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


def zip_equal(*args: Sized):
    """Check that all arguments have the same length then apply `zip` to them."""
    if not all(len(x) == len(args[0]) for x in args):
        raise ValueError('All the iterables must have the same size')

    return zip(*args)


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


def add_first_dim(x):
    return x[None]


# Legacy
add_batch_dim = add_first_dim
