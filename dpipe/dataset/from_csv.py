import os

import numpy as np
import pandas as pd

from dpipe.config import register
from dpipe.medim.utils import load_image
from .base import Dataset


class FromCSV(Dataset):
    def __init__(self, data_path, modalities, metadata_rpath='meta.csv'):
        self.data_path = data_path
        self.modality_cols = modalities

        df = pd.read_csv(os.path.join(data_path, metadata_rpath))
        df['id'] = df.id.astype(str)
        df = df.set_index('id').sort_index()
        self.df = df

        self._patient_ids = list(self.df.index)

    @property
    def patient_ids(self):
        return self._patient_ids

    @property
    def segm2msegm_matrix(self) -> np.array:
        return self._segm2msegm_matrix

    @property
    def n_chans_mscan(self):
        return len(self.modality_cols)

    def _load_by_paths(self, paths):
        return np.asarray([load_image(os.path.join(self.data_path, path))
                           for path in paths])

    def load_mscan(self, patient_id):
        paths = self.df[self.modality_cols].loc[patient_id]
        return np.array(self._load_by_paths(paths))


@register('csv_multi')
class FromCSVMultiple(FromCSV):
    def __init__(self, data_path, modalities, targets, metadata_rpath):
        super().__init__(data_path, modalities, metadata_rpath=metadata_rpath)

        self.target_cols = targets
        self._segm2msegm_matrix = np.eye(len(self.target_cols) + 1,
                                         len(self.target_cols), k=-1)

    def load_segm(self, patient_id) -> np.array:
        image = self.load_msegm(patient_id)
        weights = np.arange(1, len(self.target_cols))
        return np.einsum('ijkl,i', image, weights)

    def load_msegm(self, patient_id) -> np.array:
        paths = self.dataFrame[self.target_cols].loc[patient_id]
        image = self._load_by_paths(paths)
        assert image.dtype == np.bool

        return image


@register('csv_int')
class FromCSVInt(FromCSV):
    def __init__(self, data_path, modalities, target, metadata_rpath,
                 segm2msegm_matrix):
        super().__init__(data_path, modalities, metadata_rpath)
        assert type(target) is str
        self.target_col = target

        assert np.issubdtype(segm2msegm_matrix.dtype, np.bool)
        self._segm2msegm_matrix = np.array(segm2msegm_matrix, dtype=bool)

    def load_segm(self, patient_id):
        path = self.dataFrame[self.target_col].loc[patient_id]
        return load_image(os.path.join(self.data_path, path))
