import numpy as np
import pandas as pd


def _remove_dots(x):
    try:
        return str(int(float(x)))
    except ValueError:
        return x


def aggregate_images(dataframe: pd.DataFrame) -> pd.DataFrame:
    """Groups DICOM metadata into images."""

    def get_unique_cols(df):
        return [col for col in df.columns if len(df[col].dropna().unique()) == 1]

    def careful_drop(df, cols):
        for col in cols:
            if col in df:
                df.drop(col, 1, inplace=True)
        return df

    def process_group(entry):
        # TODO: should check that some typical cols are unique, e.g. ImageOrientationPatient*
        res = entry.iloc[[0]][get_unique_cols(entry)]
        res['FileNames'] = '/'.join(entry.FileName)
        res['SlicesCount'] = len(entry)
        # entries sometimes have no `InstanceNumber`
        # TODO: probably partially sorted slices will also do
        # TODO: detect duplicates
        try:
            res['InstanceNumbers'] = ','.join(map(_remove_dots, entry.InstanceNumber))
        except ValueError:
            res['InstanceNumbers'] = None

        return careful_drop(res, ['InstanceNumber', 'FileName'])

    group_by = ['PatientID', 'SeriesInstanceUID', 'StudyInstanceUID', 'PathToFolder']
    if 'SequenceName' in dataframe:
        group_by.append('SequenceName')

    is_string = [dataframe[col].apply(lambda x: isinstance(x, str)).all() for col in group_by]
    if not all(is_string):
        not_strings = ', '.join(np.array(group_by)[np.logical_not(is_string)])
        raise ValueError(f'The following columns do not contain only strings: {not_strings}')

    return dataframe.groupby(group_by).apply(process_group).reset_index(drop=True)


def normalize_identifiers(dataframe: pd.DataFrame) -> pd.DataFrame:
    """
    Converts PatientID to str and fills nan values in SequenceName.

    Notes
    -----
    The input `dataframe` will be mutated.
    """

    dataframe['PatientID'] = dataframe.PatientID.apply(_remove_dots)
    if 'SequenceName' in dataframe:
        dataframe.SequenceName.fillna('', inplace=True)
    return dataframe


def select(dataframe: pd.DataFrame, query: str, **where: str) -> pd.DataFrame:
    query = ' '.join(query.format(**where).splitlines())
    return dataframe.query(query).dropna(axis=1, how='all').dropna(axis=0, how='all')
