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


# TODO: this surely can be simplified
def print_structure(images: pd.DataFrame, patient_cols: Sequence[str], study_cols: Sequence[str],
                    series_cols: Sequence[str]):
    """
    Print a tree-like structure containing information from the given columns.

    Required columns: PatientID, StudyInstanceUID, SeriesInstanceUID, SeriesDescription.
    """

    def print_indent(*messages, levels):
        prefix = '  '
        for j, last in enumerate(levels):
            if is_last(j, levels):
                if last:
                    prefix += '└──'
                else:
                    prefix += '├──'
            else:
                if last:
                    prefix += '    '
                else:
                    prefix += '│   '

        print(prefix, *messages)

    def is_last(index, array):
        return index == len(array) - 1

    def print_cols(row, cols, levels, last):
        cols = [c for c in cols if pd.notnull(row[c])]
        for i, col in enumerate(cols):
            value = row[col]
            print_indent(col + ':', value, levels=levels + [last and is_last(i, cols)])

    patients = list(images.groupby('PatientID'))
    for i, (patient, studies) in enumerate(patients):
        print('Patient:', patient)
        print_cols(studies.iloc[0], patient_cols, [], False)
        studies = list(studies.groupby('StudyInstanceUID'))

        for ii, (study, all_series) in enumerate(studies):
            study_levels = [is_last(ii, studies)]
            print_indent('Study:', study, levels=study_levels)
            print_cols(all_series.iloc[0], study_cols, study_levels, False)

            all_series = list(all_series.iterrows())

            for iii, (_, row) in enumerate(all_series):
                series_levels = study_levels + [is_last(iii, all_series)]
                print('      │')
                series_id = row.SeriesInstanceUID.split('.')[-1]
                series_description = row.SeriesDescription or '<no description>'
                print_indent(series_description, f'(id={series_id})', levels=series_levels)
                print_cols(row, series_cols, series_levels, True)
        print()
