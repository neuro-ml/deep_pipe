import os
from warnings import warn

import numpy as np

from dpipe.dataset import CSV
from dpipe.dataset.base import ImageDataset
from dpipe.medim.dicom import load_series
from dpipe.medim.io import PathLike


class DICOMDataset(CSV, ImageDataset):
    n_chans_image = 1

    def __init__(self, path: PathLike, filename: str = None, index_col='PatientID'):
        if filename is None:
            warn('The new interface requires that the filename is passed as a separate argument.', DeprecationWarning)
            path, filename = os.path.split(path)
        super().__init__(path, filename, index_col)

    def load_image(self, identifier) -> np.ndarray:
        return load_series(self.df.loc[identifier], self.path)

    def load_modality(self, identifier):
        return self.get(identifier, 'Modality')
