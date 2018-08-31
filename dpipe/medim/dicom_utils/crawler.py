import os
from operator import itemgetter
from os.path import join as jp
from typing import Sequence

import pandas as pd
from tqdm import tqdm
from pydicom import valuerep, errors, read_file

serial = {'ImagePositionPatient', 'ImageOrientationPatient', 'PixelSpacing'}
person_class = (valuerep.PersonName3, valuerep.PersonNameBase)


def throw(e):
    raise e


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
    top
    ignore_extensions
    verbose
        whether to display a `tqdm` progressbar.
    """
    return pd.concat(map(itemgetter(1), walk_dicom_tree(top, ignore_extensions, verbose)))
