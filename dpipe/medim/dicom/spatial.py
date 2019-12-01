from typing import Union

import numpy as np

from ..utils import extract_dims, lmap
from ..itertools import zip_equal
from .utils import *

__all__ = [
    'get_orientation_matrix', 'get_orientation_axis', 'restore_orientation_matrix',
    'should_flip', 'normalize_orientation',
]


def get_orientation_matrix(metadata: Union[pd.Series, pd.DataFrame]):
    """Required columns: ImageOrientationPatient[0-5]"""
    orientation = metadata[ORIENTATION].astype(float).values.reshape(-1, 2, 3)
    cross = np.cross(orientation[:, 0], orientation[:, 1], axis=1)
    result = np.concatenate([orientation, cross[:, None]], axis=1)

    if metadata.ndim == 1:
        result = extract_dims(result)
    return result


def get_orientation_axis(metadata: Union[pd.Series, pd.DataFrame]):
    """Required columns: ImageOrientationPatient[0-5]"""
    m = get_orientation_matrix(metadata)
    matrix = np.atleast_3d(m)
    result = np.array([np.nan if np.isnan(row).any() else np.abs(row).argmax(axis=0)[2] for row in matrix])

    if m.ndim == 2:
        result = extract_dims(result)
    return result


def restore_orientation_matrix(metadata: Union[pd.Series, pd.DataFrame]):
    """
    Fills nan values (if possible) in ``metadata``'s ImageOrientationPatient* rows.

    Required columns: ImageOrientationPatient[0-5]

    Notes
    -----
    The input dataframe will be mutated.
    """

    def restore(vector):
        null = pd.isnull(vector)
        if null.any() and not null.all():
            length = 1 - (vector[~null] ** 2).sum()
            vector = vector.copy()
            vector[null] = length / np.sqrt(null.sum())

        return vector

    x, y = np.moveaxis(metadata[ORIENTATION].astype(float).values.reshape(-1, 2, 3), 1, 0)

    result = np.concatenate([lmap(restore, x), lmap(restore, y)], axis=1)

    if metadata.ndim == 1:
        result = extract_dims(result)

    metadata[ORIENTATION] = result
    return metadata


def order_slice_locations(dicom_metadata):
    return np.array(sorted(zip_equal(
        split_floats(dicom_metadata.InstanceNumbers),
        split_floats(dicom_metadata.SliceLocations),
    ))).T


def should_flip(dicom_metadata):
    """
    Returns True if the whole series' should be flipped
    in order to account for 'HFS' patient position.
    """
    _, locations = order_slice_locations(dicom_metadata)
    direction = dicom_metadata.PatientPosition == 'HFS'
    flip = locations[0] > locations[-1]
    return flip != direction


def normalize_orientation(image, row):
    """
    Transposes and flips the ``image`` to standard (Coronal, Sagittal, Axial) orientation.
    """
    if not contains_info(row, *ORIENTATION):
        raise ValueError('There is no enough metadata to standardize the image orientation.')

    m = get_orientation_matrix(row)
    if np.isnan(get_orientation_matrix(row)).any():
        raise ValueError('There is no enough metadata to standardize the image orientation.')

    if contains_info(row, 'InstanceNumbers', 'SliceLocations') and should_flip(row):
        image = image[..., ::-1]

    for i, j in enumerate(np.abs(m).argmax(axis=1)):
        if m[i, j] < 0:
            image = np.flip(image, axis=i)
    return image.transpose(*np.abs(m).argmax(axis=0))
