from abc import abstractmethod

import numpy as np

from dpipe.dataset.base import DataSet


class Segmentation(DataSet):
    @abstractmethod
    def load_segm(self, patient_id) -> np.array:
        """"Method returns segmentation of shape [x, y, z], filled with int
         values"""
        pass

    @abstractmethod
    def load_msegm(self, patient_id) -> np.array:
        """"Method returns multimodal segmentation of shape
         [n_chans_msegm, x, y, z]. We use this result to compute dice scores"""
        pass

    @property
    @abstractmethod
    def n_chans_segm(self):
        pass

    @property
    @abstractmethod
    def n_chans_msegm(self):
        pass

    @property
    @abstractmethod
    def n_chans_x(self) -> int:
        pass
