from abc import abstractmethod
from typing import Sequence, Union

import numpy as np

from dpipe.dataset.base import DataSet


class Segmentation(DataSet):
    def load_x(self, identifier: Union[str, int]):
        return self.load_mscan(identifier)

    @property
    @abstractmethod
    def n_chans_mscan(self) -> int:
        pass

    @abstractmethod
    def load_mscan(self, patient_id) -> np.array:
        """"Method returns multimodal scan of shape [n_chans_mscan, x, y, z]"""
        pass


class DatasetInt(Segmentation):
    @abstractmethod
    def load_segm(self, patient_id) -> np.array:
        """"Method returns segmentation of shape [x, y, z], filled with int
         values"""
        pass

    @property
    @abstractmethod
    def segm2msegm_matrix(self) -> np.array:
        """2d matrix, filled with mapping segmentation to msegmentation.
        Rows for int value from segmentation and column for channel values in
        multimodal segmentation, corresponding for each row."""
        pass

    def segm2msegm(self, x) -> np.array:
        assert np.issubdtype(x.dtype, np.integer), \
            f'Segmentation dtype must be int, but {x.dtype} provided'
        return np.rollaxis(self.segm2msegm_matrix[x], 3, 0)

    def load_msegm(self, patient_id) -> np.array:
        """"Method returns multimodal segmentation of shape
         [n_chans_msegm, x, y, z]. We use this result to compute dice scores"""
        return self.segm2msegm(self.load_segm(patient_id))

    @property
    def n_chans_segm(self):
        return self.segm2msegm_matrix.shape[0]

    @property
    def n_chans_msegm(self):
        return self.segm2msegm_matrix.shape[1]
