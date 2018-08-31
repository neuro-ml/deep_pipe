from operator import itemgetter
from os.path import join as jp
from typing import Sequence

import numpy as np
import pandas as pd
from pydicom import read_file

from ..utils import zip_equal


def get_orientation_matrix(metadata):
    try:
        orientation = metadata[[f'ImageOrientationPatient{i}' for i in range(6)]].astype(float)
    except KeyError:
        return np.full((3, 3), np.nan)
    cross = np.cross(orientation[:3], orientation[3:])
    return np.reshape(list(orientation) + list(cross), (3, 3))


def load_by_meta(metadata: pd.Series) -> np.ndarray:
    """
    Loads an image based on its row in the metadata dataframe.

    Parameters
    ----------
    metadata
        a row from the dataframe.
    """
    folder, files = metadata.PathToFolder, metadata.FileNames.split('/')
    if isinstance(metadata.InstanceNumbers, str):
        files = map(itemgetter(1), sorted(zip_equal(map(int, metadata.InstanceNumbers.split(',')), files)))

    x = np.stack((read_file(jp(folder, file)).pixel_array for file in files), axis=-1)
    if 'RescaleSlope' in metadata and not np.isnan(metadata.RescaleSlope):
        x = x * metadata.RescaleSlope
    if 'RescaleIntercept' in metadata and not np.isnan(metadata.RescaleIntercept):
        x = x + metadata.RescaleIntercept

    m = get_orientation_matrix(metadata)
    if np.isnan(m).any():
        return x

    xs, ys = np.where(np.abs(m.round()) == 1)
    permutation = xs.argsort()
    xs, ys = xs[permutation], ys[permutation]
    for axis, flip in zip(ys, m[(xs, ys)] < 0):
        if flip:
            x = np.flip(x, axis=axis)
    return x.transpose(*ys)


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
