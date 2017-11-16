import os

import numpy as np
import pandas as pd

from dpipe.config import register
from dpipe.medim.utils import load_image
from .segmentation import Segmentation, DatasetInt


@register()
class CSV:
    def __init__(self, path, filename='meta.csv', index_col='id'):
        self.path = path
        self.filename = filename

        df = pd.read_csv(os.path.join(path, filename))
        if index_col is not None:
            df[index_col] = df[index_col].astype(str)
            df = df.set_index(index_col).sort_index()
        self.df = df

        self._ids = list(self.df.index)

    @property
    def ids(self):
        return self._ids

    def get(self, index, col):
        result = self.df.loc[index, col]
        try:
            result = result.as_matrix()
        except AttributeError:
            pass
        return result


class FromCSV:
    """
    A mixin for the Dataset class. Adds support for csv files.
    """

    def __init__(self, data_path, modalities, metadata_rpath):
        self.data_path = data_path
        self.modality_cols = modalities

        df = pd.read_csv(os.path.join(data_path, metadata_rpath))
        df['id'] = df.id.astype(str)
        df = df.set_index('id').sort_index()
        self.df = df

        self._ids = list(self.df.index)

    @property
    def ids(self):
        return self._ids

    @property
    def n_chans_mscan(self):
        return len(self.modality_cols)

    def _load_by_paths(self, paths):
        return np.asarray([load_image(os.path.join(self.data_path, path))
                           for path in paths])

    def load_mscan(self, patient_id):
        paths = self.df[self.modality_cols].loc[patient_id]
        return np.array(self._load_by_paths(paths), dtype='float32')


@register('csv_multi')
class FromCSVMultiple(FromCSV, Segmentation):
    def __init__(self, data_path, modalities, targets, metadata_rpath):
        super().__init__(data_path, modalities, metadata_rpath)

        self.target_cols = targets

    def load_segm(self, patient_id) -> np.array:
        image = self.load_msegm(patient_id)
        weights = np.arange(1, len(self.target_cols) + 1)
        return np.einsum('ijkl,i', image, weights)

    def load_msegm(self, patient_id) -> np.array:
        paths = self.df[self.target_cols].loc[patient_id]
        image = self._load_by_paths(paths)
        if not (set(np.unique(image).astype(float)) - {0., 1.}):
            # in this case it's ok to convert to bool
            image = image.astype(np.bool)
        assert image.dtype == np.bool

        return image

    @property
    def n_chans_segm(self):
        return self.n_chans_msegm

    @property
    def n_chans_msegm(self):
        return len(self.target_cols)


@register('csv_int')
class FromCSVInt(FromCSV, DatasetInt):
    def __init__(self, data_path, modalities, target, metadata_rpath,
                 segm2msegm_matrix):
        super().__init__(data_path, modalities, metadata_rpath)
        assert type(target) is str
        self.target_col = target

        assert np.issubdtype(segm2msegm_matrix.dtype, np.bool)
        self._segm2msegm_matrix = np.array(segm2msegm_matrix, dtype=bool)

    @property
    def segm2msegm_matrix(self) -> np.array:
        return self._segm2msegm_matrix

    def load_segm(self, patient_id):
        path = self.df[self.target_col].loc[patient_id]
        return load_image(os.path.join(self.data_path, path))
