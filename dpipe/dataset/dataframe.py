import os

import numpy as np
import pandas as pd

from dpipe.medim.utils import load_image
from .base import Dataset


class FromMetadata(Dataset):
    def __init__(self, data_path, modalities, target,
                 metadata_rpath, segm2msegm_matrix):
        self.data_path = data_path
        self.modality_cols = modalities
        assert type(target) is str
        self.target_col = target
        self._segm2msegm_matrix = np.array(segm2msegm_matrix, dtype=bool)

        df = pd.read_csv(os.path.join(data_path, metadata_rpath))
        df['id'] = df.id.astype(str)
        self._patient_ids = df.id.as_matrix()
        self.dataFrame = df.set_index('id')

    @property
    def segm2msegm_matrix(self):
        return self._segm2msegm_matrix

    def _load_by_paths(self, paths):
        return [load_image(os.path.join(self.data_path, path)) for path in paths]

    def load_mscan(self, patient_id):
        paths = self.dataFrame[self.modality_cols].loc[patient_id]
        return np.array(self._load_by_paths(paths))

    def load_segm(self, patient_id):
        path = self.dataFrame[self.target_col].loc[patient_id]
        return load_image(os.path.join(self.data_path, path))

    @property
    def patient_ids(self):
        return self._patient_ids

    @property
    def n_chans_mscan(self):
        return len(self.modality_cols)
