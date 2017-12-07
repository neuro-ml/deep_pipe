from typing import Sequence

import numpy as np


def extract(l, idx):
    return [l[i] for i in idx]


def build_slices(start, end):
    assert len(start) == len(end)
    return list(map(slice, start, end))


def pad(x, padding, padding_values):
    padding = np.array(padding)

    new_shape = np.array(x.shape) + np.sum(padding, axis=1)
    new_x = np.zeros(new_shape, dtype=x.dtype)
    new_x[:] = padding_values

    start = padding[:, 0]
    end = np.where(padding[:, 1] != 0, -padding[:, 1], None)
    new_x[build_slices(start, end)] = x

    return new_x


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

    raise ValueError(f"Couldn't read image from path: {path}.\n"
                     "Unknown file extension.")


def load_by_ids(load_x: callable, load_y: callable, ids: Sequence, shuffle: bool = False):
    """
    Yields pairs of objects given their loaders and ids

    Parameters
    ----------
    load_x,load_y: callable(id)
        Loaders for x and y
    ids: Sequence
        a sequence of ids to load
    shuffle: bool, optional
        whether to shuffle the ids before yielding
    """
    if shuffle:
        ids = np.random.permutation(ids)
    for patient_id in ids:
        yield load_x(patient_id), load_y(patient_id)
