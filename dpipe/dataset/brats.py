from abc import ABCMeta
from os.path import join
from functools import lru_cache

import numpy as np
import pandas as pd

from .base import Dataset


def cached_property(f):
    return property(lru_cache(1)(f))


class Brats(Dataset, metaclass=ABCMeta):
    def __init__(self, data_path):
        self.data_path = data_path
        self.metadata = pd.read_csv(join(data_path, 'metadata.csv'),
                                    index_col='id')

    def _build_data_name(self, patient_id):
        return join(self.data_path, 'data', patient_id)

    @cached_property
    def patient_ids(self):
        ids = sorted(self.metadata.index.values)
        # FIXME Wrong sample
        ids = [i for i in ids if i != 'Brats17_CBICA_AYW_1']
        return ids

    @property
    def n_chans_mscan(self):
        return 4

    def load_mscan(self, patient_id):
        dataname = self._build_data_name(patient_id)
        return np.load(dataname + '_mscan.npy')

    def load_segm(self, patient_id):
        dataname = self._build_data_name(patient_id)
        segm = np.load(dataname + '_segm.npy')
        return np.array(segm, dtype=np.uint8)


class Brats2015(Brats):
    @cached_property
    def segm2msegm(self):
        return np.array([
            [0, 0, 0],
            [1, 1, 0],
            [1, 0, 0],
            [1, 1, 0],
            [1, 1, 1]
        ], dtype=bool)


class Brats2017(Brats):
    """Data loader for brats 2017. We have replaced label 4 with 3 during data
    preparation."""

    @cached_property
    def segm2msegm(self):
        return np.array([
            [0, 0, 0],
            [1, 1, 0],
            [1, 0, 0],
            [1, 1, 1]
        ], dtype=bool)
