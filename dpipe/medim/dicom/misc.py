import os
from operator import itemgetter
from os.path import join as jp
from typing import Sequence, Union

import numpy as np
import pandas as pd
from pydicom import read_file

from ..io import PathLike
from ..utils import extract_dims, lmap, composition
from ..itertools import zip_equal, collect

ORIENTATION = [f'ImageOrientationPatient{i}' for i in range(6)]


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


def _contains_info(row, *cols):
    return all(col in row and pd.notnull(row[col]) for col in cols)


def load_series(row: pd.Series, base_path: PathLike = None) -> np.ndarray:
    """
    Loads an image based on its ``row`` in the metadata dataframe.

    If ``base_path`` is not None, PathToFolder is assumed to be relative to it.

    Required columns: PathToFolder, FileNames.
    """
    folder, files = row.PathToFolder, row.FileNames.split('/')
    if base_path is not None:
        folder = os.path.join(base_path, folder)
    if _contains_info(row, 'InstanceNumbers'):
        files = map(itemgetter(1), sorted(zip_equal(map(int, row.InstanceNumbers.split(',')), files)))

    x = np.stack((read_file(jp(folder, file)).pixel_array for file in files), axis=-1)
    if _contains_info(row, 'RescaleSlope'):
        x = x * row.RescaleSlope
    if _contains_info(row, 'RescaleIntercept'):
        x = x + row.RescaleIntercept

    if not _contains_info(row, *ORIENTATION):
        return x

    m = get_orientation_matrix(row)
    assert not np.isnan(m).any()

    for i, j in enumerate(np.abs(m).argmax(axis=1)):
        if m[i, j] < 0:
            x = np.flip(x, axis=i)
    return x.transpose(*np.abs(m).argmax(axis=0))


@composition('\n'.join)
def get_structure(images: pd.DataFrame, patient_cols: Sequence[str], study_cols: Sequence[str],
                  series_cols: Sequence[str]):
    """
    Get a tree-like structure containing information from the given columns.

    Required columns: PatientID, StudyInstanceUID, SeriesInstanceUID, SeriesDescription.
    """

    @composition('\n'.join)
    def move_right(header, strings):
        yield header

        for i, value in enumerate(strings):
            if i == len(strings) - 1:
                start = '└── '
                middle = '    '
            else:
                start = '├── '
                middle = '│   '

            for j, line in enumerate(value.splitlines()):
                if j == 0:
                    line = start + line
                else:
                    line = middle + line

                yield line

    @collect
    def get_cols(row, cols):
        for col in cols:
            if col in row and pd.notnull(row[col]):
                yield f'{col}: {row[col]}'

    def describe_series(row):
        series_id = row.SeriesInstanceUID.split('.')[-1]
        series_description = row.SeriesDescription or '<no description>'
        return move_right(f'Series: {series_description} (id={series_id})', get_cols(row, series_cols))

    def describe_study(all_series):
        header = 'Study: ' + all_series.iloc[0].StudyInstanceUID
        attrs = get_cols(all_series.iloc[0], study_cols)
        if attrs:
            attrs[-1] += '\n '
        series = [describe_series(row) for _, row in all_series.iterrows()]
        return move_right(header, attrs + series)

    for patient, studies_groups in images.groupby('PatientID'):
        studies = [describe_study(all_series) for _, all_series in studies_groups.groupby('StudyInstanceUID')]
        yield move_right('Patient: ' + patient, get_cols(studies_groups.iloc[0], patient_cols) + studies)
        yield ''


def print_structure(images: pd.DataFrame, patient_cols: Sequence[str], study_cols: Sequence[str],
                    series_cols: Sequence[str]):
    """
    Print a tree-like structure containing information from the given columns.

    Required columns: PatientID, StudyInstanceUID, SeriesInstanceUID, SeriesDescription.
    """
    print(get_structure(images, patient_cols, study_cols, series_cols))
