from .base import Dataset

from os.path import join

import numpy as np
import pandas as pd


class Brats(Dataset):
    def __init__(self, processed_path):
        super().__init__(processed_path)
        self.processed_path = processed_path
        self.metadata = pd.read_csv(join(processed_path, 'metadata.csv'),
                                    index_col='id')
        self._patient_ids = self.metadata.index.values

    def build_data_name(self, patient_id):
        return join(self.processed_path, 'data', patient_id)

    def load_mscan(self, patient_id):
        dataname = self.build_data_name(patient_id)
        return np.load(dataname + '_mscan.npy')

    def load_segm(self, patient_id):
        dataname = self.build_data_name(patient_id)
        return np.load(dataname + '_segm.npy')

    def load_msegm(self, patient_id):
        segm = self.load_segm(patient_id)
        return self.segm2msegm(segm)

    @property
    def patient_ids(self):
        return self._patient_ids

    @property
    def n_chans_mscan(self):
        return 4

    @property
    def n_chans_msegm(self):
        return 3


class Brats2015(Brats):
    def segm2msegm(self, segm):
        r = np.zeros((3, *segm.shape), dtype=bool)
        r[0] = segm > 0
        r[1] = (segm == 1) | (segm >= 3)
        r[2] = (segm == 4)
        return r

    @property
    def n_classes(self):
        return 5

    @property
    def spatial_size(self):
        return [146, 181, 160]


class Brats2017(Brats):
    """Data loader for brats 2017. We have replaced label 4 with 3 during data
    preparation."""
    def segm2msegm(self, segm):
        r = np.zeros((3, *segm.shape), dtype=bool)
        r[0] = segm > 0
        r[1] = (segm == 1) | (segm == 3)
        r[2] = (segm == 3)
        return r

    @property
    def n_classes(self):
        return 4

    @property
    def spatial_size(self):
        return [157, 189, 149]