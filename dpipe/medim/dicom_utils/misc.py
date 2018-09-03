from operator import itemgetter
from os.path import join as jp
from typing import Sequence, Union

import numpy as np
import pandas as pd
from pydicom import read_file

from ..utils import extract_dims
from ..itertools import zip_equal

ORIENTATION = [f'ImageOrientationPatient{i}' for i in range(6)]


def get_orientation_matrix(metadata: Union[pd.Series, pd.DataFrame]):
    """Required columns: ImageOrientationPatient[0-5]"""
    orientation = metadata[ORIENTATION].astype(float).values.reshape(-1, 2, 3)
    cross = np.cross(orientation[:, 0], orientation[:, 1], axis=1)
    result = np.concatenate([orientation, cross[:, None]], axis=1)

    if metadata.ndim == 1:
        result = extract_dims(result)
    return result


def _contains_info(row, *cols):
    return all(col in row and pd.notnull(row[col]) for col in cols)


def load_by_meta(row: pd.Series) -> np.ndarray:
    """
    Loads an image based on its ``row`` in the metadata dataframe.

    Required columns: PathToFolder, FileNames.
    """
    folder, files = row.PathToFolder, row.FileNames.split('/')
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

    transpose = np.abs(m).argmax(axis=0)
    for i, j in enumerate(transpose):
        if m[i, j] < 0:
            x = np.flip(x, axis=i)
    return x.transpose(*transpose)


def print_structure(images: pd.DataFrame, patient_cols: Sequence[str], study_cols: Sequence[str],
                    series_cols: Sequence[str]):
    def print_indent(*messages, level=0):
        print('|' + '--' * level, *messages)

    def print_cols(row, cols, level):
        for col in cols:
            value = row[col]
            if pd.notnull(value):
                print_indent(col, ':', value, level=level)

    for patient, studies in images.groupby('PatientID'):
        print_indent('Patient:', patient)
        print_cols(studies.iloc[0], patient_cols, 1)

        for study, all_series in studies.groupby('StudyInstanceUID'):
            print_indent()
            print_indent('Study:', study, level=1)
            print_cols(studies.iloc[0], study_cols, 2)

            for series, data in all_series.groupby('SeriesInstanceUID'):
                for _, row in data.iterrows():
                    print_indent()
                    print_indent('Series:', series, level=2)
                    print_cols(row, series_cols, 3)
        print()
