import os
from warnings import warn

import numpy as np

from dpipe.dataset import CSV
from dpipe.dataset.base import ImageDataset
from dpipe.medim.dicom import load_series
from dpipe.medim.io import PathLike


class DICOMDataset(CSV, ImageDataset):
    """
    A loader for DICOM series.
    All the metadata is stored at ``filename`` and the DICOM files are located relative to ``path``.

    Parameters
    ----------
    path: PathLike
        the path to the data.
    filename: str
        the relative path to the csv dataframe. Default is ``meta.csv``.
    index_col: str, None, optional
        the column that will be used as index. Must contain unique values. Default is ``id``.

    References
    ----------
    `aggregate_images`, `CSV`
    """
    n_chans_image = 1

    def __init__(self, path: PathLike, filename: str = None, index_col: str = 'PatientID'):
        if filename is None:
            warn('The new interface requires that the filename is passed as a separate argument.', DeprecationWarning)
            path, filename = os.path.split(path)
        super().__init__(path, filename, index_col)

    def load_image(self, identifier) -> np.ndarray:
        return load_series(self.df.loc[identifier], self.path)

    def load_modality(self, identifier) -> str:
        return self.get(identifier, 'Modality')
