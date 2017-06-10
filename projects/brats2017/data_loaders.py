from abc import ABC, abstractmethod

from os.path import join

import numpy as np
import pandas as pd


class Brats(ABC):
    def __init__(self, processed_path):
        self.processed_path = processed_path
        self.metadata = pd.read_csv(join(processed_path, 'metadata.csv'),
                                    index_col='id')
        self.patients = self.metadata.index.values

    def build_data_name(self, patient):
        return join(self.processed_path, 'data', patient)

    def load_mscan(self, patient):
        dataname = self.build_data_name(patient)
        return np.load(dataname + '_mscan.npy')

    def load_segm(self, patient):
        dataname = self.build_data_name(patient)
        return np.load(dataname + '_segm.npy')

    def load_msegm(self, patient):
        segm = self.load_segm(patient)
        return self.segm2msegm(segm)

    @abstractmethod
    def segm2msegm(self, segm):
        pass

    @property
    def n_modalities(self):
        return 4

    @property
    def n_chans_msegm(self):
       return 3

    @property
    @abstractmethod
    def n_classes(self):
        pass

    @property
    @abstractmethod
    def spatial_size(self):
        pass


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
    """Data loader for brats 2017. We have replaced label 4 with 3."""
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