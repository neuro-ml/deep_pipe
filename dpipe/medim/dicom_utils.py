import os
from operator import itemgetter
from os.path import join as jp

import numpy as np
import pydicom
import pandas as pd
from tqdm import tqdm

serial = {'ImagePositionPatient', 'ImageOrientationPatient', 'PixelSpacing'}
person_class = (pydicom.valuerep.PersonName, pydicom.valuerep.PersonName3,
                pydicom.valuerep.PersonNameBase, pydicom.valuerep.PersonNameUnicode)


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
            dc = pydicom.read_file(jp(folder, file))
        except (pydicom.errors.InvalidDicomError, OSError):
            entry['NoError'] = False
            continue

        try:
            has_px = hasattr(dc, 'pixel_array')
        except (TypeError, NotImplementedError):
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

    return pd.DataFrame.from_dict(result)


def folder_to_df(path):
    for root, _, files in os.walk(path):
        return files_to_df(root, files)


def walk_dicom_tree(top, verbose=True):
    iterator = filter(lambda batch: batch[2], os.walk(top))
    if verbose:
        iterator = tqdm(iterator)

    for root, _, files in iterator:
        relative = os.path.relpath(root, top)
        if verbose:
            iterator.set_description(relative)
        yield relative, files_to_df(root, files)


def join_dicom_tree(top, verbose=True):
    return pd.concat(map(itemgetter(1), walk_dicom_tree(top, verbose)))


def aggregate_images(dataframe: pd.DataFrame):
    def get_unique_cols(df):
        return [col for col in df.columns if len(df[col].dropna().unique()) <= 1]

    def process_group(entry):
        res = entry.iloc[[0]][get_unique_cols(entry)]
        res['FileNames'] = '/'.join(entry.FileName)
        res['SlicesCount'] = len(entry)
        res['InstanceNumbers'] = ','.join(map(lambda x: str(int(x)), entry.InstanceNumber))
        return res

    group_by = ['PatientID', 'SeriesInstanceUID', 'StudyInstanceUID', 'PathToFolder', 'SequenceName']
    is_string = [dataframe[col].apply(lambda x: isinstance(x, str)).all() for col in group_by]
    if not all(is_string):
        not_strings = ', '.join(np.array(np.logical_not(is_string)))
        raise ValueError(f'The following columns do not contain only strings: {not_strings}')

    return dataframe.groupby(group_by).apply(process_group).reset_index(drop=True)


def select(dataframe, query: str, **where):
    query = ' '.join(query.format(**where).splitlines())
    return dataframe.query(query).dropna(axis=1, how='all').dropna(axis=0, how='all')


def load_by_meta(metadata):
    folder, files = metadata.PathToFolder, metadata.FileNames.split('/')
    x = np.stack((pydicom.read_file(jp(folder, file)).pixel_array
                  for _, file in sorted(zip(map(int, metadata.InstanceNumbers.split(',')), files))), axis=-1)

    orientation = metadata[[f'ImageOrientationPatient{i}' for i in range(6)]].astype(float)
    cross = np.cross(orientation[:3], orientation[3:])
    m = np.reshape(list(orientation) + list(cross), (3, 3))
    if np.isnan(m).any():
        # TODO: warn
        return x

    xs, ys = np.where(np.abs(m.round()) == 1)
    permutation = xs.argsort()
    xs, ys = xs[permutation], ys[permutation]
    for axis, flip in zip(ys, m[(xs, ys)] < 0):
        if flip:
            x = np.flip(x, axis=axis)
    return x.transpose(*ys)
