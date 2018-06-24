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


def folder_to_df(path, verbose=True):
    iterator = sorted(os.listdir(path))
    if verbose:
        iterator = tqdm(iterator)

    result = []
    for file in iterator:
        file_path = jp(path, file)
        if file == 'DICOMDIR' or not os.path.isfile(file_path):
            continue

        if verbose:
            iterator.set_description(file)

        entry = {}
        result.append(entry)

        entry['PathToFolder'] = path
        entry['FileName'] = file
        try:
            dc = pydicom.read_file(file_path)
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


def walk_dicom_tree(top, verbose=True):
    iterator = os.walk(top)
    if verbose:
        iterator = tqdm(iterator)

    for root, folders, files in iterator:
        relative = os.path.relpath(root, top)
        if verbose:
            iterator.set_description(relative)

        if files:
            yield relative, folder_to_df(root, verbose=False)


def load_by_meta(row):
    folder, files = row.PathToFolder, row.FileNames.split('/')
    x = np.stack((pydicom.read_file(jp(folder, file)).pixel_array for file in files), axis=-1)

    orientation = row[[f'ImageOrientationPatient{i}' for i in range(6)]].astype(float)
    cross = np.cross(orientation[:3], orientation[3:])
    m = np.reshape(list(orientation) + list(cross), (3, 3))
    return x, m


def join_dicom_tree(top, verbose=True):
    return pd.concat(map(itemgetter(1), walk_dicom_tree(top, verbose)))


def aggregate_images(dataframe):
    def get_unique_cols(df):
        return [col for col in df.columns if len(df[col].dropna().unique()) <= 1]

    def process_group(entry):
        res = entry.iloc[[0]][get_unique_cols(entry)]
        res['FileNames'] = '/'.join(entry.FileName)
        res['SlicesCount'] = len(entry)
        return res

    return dataframe.groupby(
        ('PatientID', 'SeriesInstanceUID', 'StudyID', 'PathToFolder', 'SequenceName')
    ).apply(process_group).reset_index(drop=True)


def select(dataframe, query: str, **where):
    query = ' '.join(query.format(**where).splitlines())
    return dataframe.query(query).dropna(axis=1, how='all').dropna(axis=0, how='all')
