import os
from operator import itemgetter
from os.path import join as jp
from typing import Sequence

import numpy as np
import pandas as pd
from tqdm import tqdm
from pydicom import valuerep, errors, read_file

from .utils import zip_equal

serial = {'ImagePositionPatient', 'ImageOrientationPatient', 'PixelSpacing'}
person_class = (valuerep.PersonName3, valuerep.PersonNameBase)


def throw(e):
    raise e


def _remove_dots(x):
    try:
        return str(int(float(x)))
    except ValueError:
        return x


def files_to_df(folder, files):
    result = []
    for file in files:
        if file == 'DICOMDIR':
            continue

        entry = {}
        result.append(entry)

        entry['PathToFolder'] = folder
        entry['FileName'] = file
        try:
            dc = read_file(jp(folder, file))
        except (errors.InvalidDicomError, OSError, NotImplementedError):
            entry['NoError'] = False
            continue

        try:
            has_px = hasattr(dc, 'pixel_array')
        except (TypeError, NotImplementedError):
            # for some formats the following packages might be required
            # conda install -c clinicalgraphics gdcm
            has_px = False

        entry['HasPixelArray'] = has_px
        entry['NoError'] = True

        for attr in dc.dir():
            value = dc.get(attr)

            if isinstance(value, person_class):
                entry[attr] = str(value)

            elif attr in serial:
                for pos, num in enumerate(value):
                    entry[f'{attr}{pos}'] = num

            elif isinstance(value, (int, float, str)):
                entry[attr] = value

    return pd.DataFrame(result)


def folder_to_df(path):
    for root, _, files in os.walk(path):
        return files_to_df(root, files)


def walk_dicom_tree(top: str, ignore_extensions: Sequence[str] = (), verbose: bool = True):
    for extension in ignore_extensions:
        if not extension.startswith('.'):
            raise ValueError(f'Each extension must start with a dot: "{extension}".')

    def walker():
        for root_, _, files_ in os.walk(top, onerror=throw):
            files_ = [file for file in files_ if not any(file.endswith(ext) for ext in ignore_extensions)]
            if files_:
                yield root_, files_

    iterator = walker()
    if verbose:
        iterator = tqdm(iterator)

    for root, files in iterator:
        relative = os.path.relpath(root, top)
        if verbose:
            iterator.set_description(relative)
        yield relative, files_to_df(root, files)


def join_dicom_tree(top: str, ignore_extensions: Sequence[str] = (), verbose: bool = True) -> pd.DataFrame:
    """
    Returns a dataframe containing metadata for each file in all the subfolders of `top`.

    Parameters
    ----------
    top: str
    ignore_extensions: Sequence[str], optional
    verbose: bool, optional
        whether to display a `tqdm` progressbar.
    """
    return pd.concat(map(itemgetter(1), walk_dicom_tree(top, ignore_extensions, verbose)))


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
    metadata: pd.Series
        a row from the dataframe

    Returns
    -------
    image: np.ndarray
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
