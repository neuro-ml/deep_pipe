import os
import functools

import numpy as np
import pandas as pd

from dpipe.medim.utils import load_image
from .base import Dataset


class FromDataFrame(Dataset):
    def __init__(self, data_path, modalities, targets, filename='data.csv'):
        self.data_path = data_path
        self.modality_cols = modalities
        self.target_cols = targets

        df = pd.read_csv(os.path.join(data_path, filename))
        df['id'] = df.id.astype(str)
        self._patient_ids = df.id.as_matrix()
        self.dataFrame = df.set_index('id')

    def load_by_paths(self, paths):
        res = []
        for image in paths:
            image = os.path.join(self.data_path, image)
            x = load_image(image)
            res.append(x)

        return np.asarray(res)

    def load_msegm(self, patient):
        # super method is needed here to allow chaining in its children
        super().load_msegm(patient)

        paths = self.dataFrame[self.target_cols].loc[patient]
        image = self.load_by_paths(paths)

        return image >= .5

    def load_mscan(self, patient_id):
        super().load_mscan(patient_id)

        paths = self.dataFrame[self.modality_cols].loc[patient_id]
        image = self.load_by_paths(paths)

        return image.astype('float32')

    def load_segm(self, patient_id):
        super().load_segm(patient_id)

    @property
    def segm2msegm_matrix(self):
        return super().segm2msegm_matrix

    @property
    def patient_ids(self):
        return self._patient_ids

    @property
    def n_chans_mscan(self):
        return len(self.modality_cols)

    @property
    def n_chans_msegm(self):
        return len(self.target_cols)

    @property
    def n_classes(self):
        return len(self.target_cols)

def partial(data_path=None, modalities=None, targets=None, filename=None):
    kwargs = {}
    if data_path is not None:
        kwargs['data_path'] = data_path
    if modalities is not None:
        kwargs['modalities'] = modalities
    if targets is not None:
        kwargs['targets'] = targets
    if filename is not None:
        kwargs['filename'] = filename

    return functools.partial(FromDataFrame, **kwargs)