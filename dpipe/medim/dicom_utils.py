import os
from os.path import join as jp

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
        if verbose:
            iterator.set_description(file)
        if file == 'DICOMDIR':
            continue
        # TODO: process only dicoms

        entry = {}
        result.append(entry)

        file_path = jp(path, file)
        entry['PathToFile'] = file_path

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
        if any(x.endswith('.dcm') for x in files):
            continue

        relative = os.path.relpath(root, top)
        if verbose:
            iterator.set_description(relative)

        yield relative, folder_to_df(root, verbose=False)
