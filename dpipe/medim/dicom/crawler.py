"""Contains functions for gathering metadata from individual DICOM files or entire directories."""
import os
from operator import itemgetter
from os.path import join as jp
from typing import Sequence, Iterable

import pandas as pd
from tqdm import tqdm
from pydicom import valuerep, errors, read_file

from ..io import PathLike

__all__ = 'get_file_meta', 'join_dicom_tree'

serial = {'ImagePositionPatient', 'ImageOrientationPatient', 'PixelSpacing'}
person_class = (valuerep.PersonName3, valuerep.PersonNameBase)


def _throw(e):
    raise e


def get_file_meta(path: PathLike) -> dict:
    """
    Get a dict containing the metadata from the DICOM file located at ``path``.

    Notes
    -----
    The following keys are added:
        | NoError: whether an exception was raised during reading the file.
        | HasPixelArray: (if NoError is True) whether the file contains a pixel array.
        | PixelArrayShape: (if HasPixelArray is True) the shape of the pixel array.

    For some formats the following packages might be required:
        >>> conda install -c clinicalgraphics gdcm
    """
    result = {}

    try:
        dc = read_file(path)
        result['NoError'] = True
    except (errors.InvalidDicomError, OSError, NotImplementedError):
        result['NoError'] = False
        return result

    try:
        has_px = hasattr(dc, 'pixel_array')
    except (TypeError, NotImplementedError):
        has_px = False
    else:
        if has_px:
            result['PixelArrayShape'] = ','.join(map(str, dc.pixel_array.shape))
    result['HasPixelArray'] = has_px

    for attr in dc.dir():
        try:
            value = dc.get(attr)
        except NotImplementedError:
            continue

        if isinstance(value, person_class):
            result[attr] = str(value)

        elif attr in serial:
            for pos, num in enumerate(value):
                result[f'{attr}{pos}'] = num

        elif isinstance(value, (int, float, str)):
            result[attr] = value

    return result


def files_to_df(folder: PathLike, files: Iterable[PathLike]) -> pd.DataFrame:
    result = []
    for file in files:
        entry = get_file_meta(jp(folder, file))
        entry['PathToFolder'] = folder
        entry['FileName'] = file
        result.append(entry)

    return pd.DataFrame(result)


def folder_to_df(path):
    for root, _, files in os.walk(path):
        return files_to_df(root, files)


def walk_dicom_tree(top: PathLike, ignore_extensions: Sequence[str] = (), verbose: bool = True):
    for extension in ignore_extensions:
        if not extension.startswith('.'):
            raise ValueError(f'Each extension must start with a dot: "{extension}".')

    def walker():
        for root_, _, files_ in os.walk(top, onerror=_throw):
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


def join_dicom_tree(top: PathLike, ignore_extensions: Sequence[str] = (), verbose: bool = True) -> pd.DataFrame:
    """
    Returns a dataframe containing metadata for each file in all the subfolders of ``top``.

    Parameters
    ----------
    top
    ignore_extensions
    verbose
        whether to display a `tqdm` progressbar.

    Notes
    -----
    The following columns are added:
        | NoError: whether an exception was raised during reading the file.
        | HasPixelArray: (if NoError is True) whether the file contains a pixel array.
        | PixelArrayShape: (if HasPixelArray is True) the shape of the pixel array.
        | PathToFolder
        | FileName

    For some formats the following packages might be required:
        >>> conda install -c clinicalgraphics gdcm
    """
    return pd.concat(map(itemgetter(1), walk_dicom_tree(top, ignore_extensions, verbose)))
