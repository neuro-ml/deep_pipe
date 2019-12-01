import os
from operator import itemgetter
from os.path import join as jp
from typing import Sequence

import numpy as np
from pydicom import read_file

from ..io import PathLike
from ..utils import composition
from ..itertools import zip_equal, collect
from .spatial import *
from .utils import *

__all__ = 'load_series', 'get_structure', 'print_structure'


def load_series(row: pd.Series, base_path: PathLike = None, orientation: bool = None) -> np.ndarray:
    """
    Loads an image based on its ``row`` in the metadata dataframe.

    If ``base_path`` is not None, PathToFolder is assumed to be relative to it.

    If ``orientation`` is True, the loaded image will be transposed and flipped
    to standard (Coronal, Sagittal, Axial) orientation.

    Required columns: PathToFolder, FileNames.
    """
    folder, files = row.PathToFolder, row.FileNames.split('/')
    if base_path is not None:
        folder = os.path.join(base_path, folder)
    if contains_info(row, 'InstanceNumbers'):
        files = map(itemgetter(1), sorted(zip_equal(map(int, row.InstanceNumbers.split(',')), files)))

    x = np.stack([read_file(jp(folder, file)).pixel_array for file in files], axis=-1)
    if contains_info(row, 'RescaleSlope'):
        x = x * row.RescaleSlope
    if contains_info(row, 'RescaleIntercept'):
        x = x + row.RescaleIntercept

    if orientation is None:
        orientation = contains_info(row, *ORIENTATION)
    if not orientation:
        return x

    return normalize_orientation(x, row)


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
