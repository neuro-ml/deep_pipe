from typing import Sequence

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
    return list(sorted(axes))


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


def load_by_ids(*loaders, ids: Sequence, shuffle: bool = False):
    """
    Yields pairs of objects given their loaders and ids

    Parameters
    ----------
    loaders: List[callable(id)]
        Loaders for x and y
    ids: Sequence
        a sequence of ids to load
    shuffle: bool, optional
        whether to shuffle the ids before yielding
    """
    if shuffle:
        ids = np.random.permutation(ids)
    for identifier in ids:
        result = tuple(loader(identifier) for loader in loaders)
        if len(result) == 1:
            result = result[0]
        yield result
