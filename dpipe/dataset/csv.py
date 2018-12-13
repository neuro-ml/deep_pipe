import os

import numpy as np
import pandas as pd

from dpipe.medim import load_image
from dpipe.medim.dicom import load_series
from .base import Dataset, ImageDataset


def multiple_columns(method, index, columns):
    return np.array([method(index, col) for col in columns])


class CSV(Dataset):
    """A small wrapper for csv files."""

    def __init__(self, path: str, filename: str = 'meta.csv', index_col: str = 'id'):
        self.path = path
        self.filename = filename

        df = pd.read_csv(os.path.join(path, filename))
        if index_col is not None:
            df[index_col] = df[index_col].astype(str)
            df = df.set_index(index_col).sort_index()
            if len(df.index.unique()) != len(df):
                raise ValueError(f'The column "{index_col}" doesn\'t contain unique values.')

        self.df: pd.DataFrame = df
        self.ids = tuple(self.df.index)

    def get(self, index, col):
        return self.df.loc[index, col]

    def get_global_path(self, index: str, col: str) -> str:
        """
        Join the slice's result with the data frame's ``path``.
        Often data frames contain path to data, this is a convenient way to obtain
        the global path.
        """
        return os.path.join(self.path, self.get(index, col))

    def load(self, index: str, col: str, loader=load_image):
        return loader(self.get_global_path(index, col))


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

    def load_pixel_spacing(self, identifier):
        return self.get(identifier, ['PixelSpacing0', 'PixelSpacing1']).values
