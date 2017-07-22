from abc import ABC, abstractmethod
from functools import lru_cache

import numpy as np
from typing import List


class Dataset(ABC):
    @abstractmethod
    def __init__(self, data_path):
        self.data_path = data_path

    @abstractmethod
    def load_mscan(self, patient_id) -> np.array:
        """"Method returns multimodal scan of shape [n_chans_mscan, x, y, z]"""
        pass

    @abstractmethod
    def load_segm(self, patient_id) -> np.array:
        """"Method returns segmentation of shape [x, y, z], filled with int
         values"""
        pass

    @abstractmethod
    def load_msegm(self, patient) -> np.array:
        """"Method returns multimodel segmentation of shape
         [n_chans_msegm, x, y, z]. We use this result to compute dice scores"""
        pass

    @abstractmethod
    def segm2msegm(self, segm) -> np.array:
        pass

    @property
    @abstractmethod
    def patient_ids(self) -> List[str]:
        pass

    @property
    @abstractmethod
    def n_chans_mscan(self) -> int:
        pass

    @property
    @abstractmethod
    def n_chans_msegm(self) -> int:
        pass

    @property
    @abstractmethod
    def n_classes(self) -> int:
        """Number of classes for this problem. Supposed to be consistent with
        maximum int value in load_segm"""

    def load_x(self, patient_id):
        return self.load_mscan(patient_id)

    def load_y(self, patient_id):
        return self.load_msegm(patient_id)


def make_cached(dataset) -> Dataset:
    n = len(dataset.patient_ids)

    class CachedDataset:
        def __init__(self, dataset):
            self.dataset = dataset

        @lru_cache(n)
        def load_x(self, patient_id):
            return self.dataset.load_x(patient_id)

        @lru_cache(n)
        def load_y(self, patient_id):
            return self.dataset.load_y(patient_id)

        def __getattr__(self, name):
            return getattr(self.dataset, name)

    return CachedDataset(dataset)
