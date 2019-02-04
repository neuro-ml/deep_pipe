import os

import numpy as np

from dpipe.dataset import CSV
from dpipe.dataset.base import ImageDataset
from dpipe.medim.dicom import load_series


class DICOMDataset(CSV, ImageDataset):
    def __init__(self, path: str, index_col='PatientID'):
        super().__init__(*os.path.split(path), index_col)
        self.n_chans_image = 1

    def load_image(self, identifier) -> np.ndarray:
        return load_series(self.df.loc[identifier])

    def load_modality(self, identifier):
        return self.get(identifier, 'Modality')

    def load_shape(self, identifier):
        return (*map(int, self.get(identifier, 'PixelArrayShape').split(',')), int(self.get(identifier, 'SlicesCount')))
