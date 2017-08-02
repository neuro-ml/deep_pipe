from abc import ABC, abstractmethod
from functools import lru_cache

import numpy as np


class Dataset(ABC):
    @property
    @abstractmethod
    def patient_ids(self) -> [str]:
        pass

    @property
    @abstractmethod
    def n_chans_mscan(self) -> int:
        pass

    @abstractmethod
    def load_mscan(self, patient_id) -> np.array:
        """"Method returns multimodal scan of shape [n_chans_mscan, x, y, z]"""
        pass

    @abstractmethod
    def load_segm(self, patient_id) -> np.array:
        """"Method returns segmentation of shape [x, y, z], filled with int
         values"""
        pass

    @property
    @abstractmethod
    def segm2msegm(self) -> np.array:
        """2d matrix, filled with mapping segmentation to msegmentation.
        Rows for int value from segmentation and column for channel values in
        multimodal segmentation, corresponding for each row."""
        pass

    def load_msegm(self, patient_id) -> np.array:
        """"Method returns multimodal segmentation of shape
         [n_chans_msegm, x, y, z]. We use this result to compute dice scores"""
        return self.segm2msegm[self.load_segm(patient_id)]


class Proxy:
    def __init__(self, shadowed):
        self._shadowed = shadowed

    def __getattr__(self, name):
        return getattr(self._shadowed, name)


def make_cached(dataset) -> Dataset:
    n = len(dataset.patient_ids)

    class CachedDataset(Proxy):
        @lru_cache(n)
        def load_mscan(self, patient_id):
            return self._shadowed.load_mscan(patient_id)

        @lru_cache(n)
        def load_segm(self, patient_id):
            return self._shadowed.load_segm(patient_id)

        @lru_cache(n)
        def load_msegm(self, patient_id):
            return self._shadowed.load_msegm(patient_id)

    return CachedDataset(dataset)