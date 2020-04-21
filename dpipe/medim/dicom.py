from warnings import warn

import pandas as pd
from dpipe.im.utils import composition, Sequence, collect, name_changed
from dicom_csv import *

warn('`dpipe.medim.dicom` has been moved to a separate library `dicom-csv`.', DeprecationWarning)
warn('`dpipe.medim.dicom` has been moved to a separate library `dicom-csv`.', UserWarning, stacklevel=2)

join_dicom_tree = name_changed(join_tree, 'join_dicom_tree', '05.02.2020')


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
