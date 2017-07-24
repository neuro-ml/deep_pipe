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
        pass


class Proxy:
    def __init__(self, shadowed):
        self._shadowed = shadowed

    def __getattr__(self, name):
        return getattr(self._shadowed, name)


    def load_x(self, patient_id):
        return self.load_mscan(patient_id)


def make_tasked_dataset(dataset, dataset_task):
    if dataset_task == 'segm':
        return make_segm_y(dataset)
    elif dataset_task == 'msegm':
        return make_msegm_y(dataset)
    else:
        raise ValueError('Unknown dataset type\n' + \
                         'Received: {}'.format(dataset_task) + \
                         'Possible values segm and msegm')


def make_msegm_y(dataset) -> Dataset:
    class TaskedDataset(Proxy):
        def load_x(self, patient_id):
            return self._shadowed.load_mscan(patient_id)

        def load_y(self, patient_id):
            return self._shadowed.load_msegm(patient_id)

        @property
        def n_chans_out(self):
            return self.n_chans_msegm

    return TaskedDataset(dataset)


def make_segm_y(dataset) -> Dataset:
    class TaskedDataset(Proxy):
        def load_x(self, patient_id):
            return self._shadowed.load_mscan(patient_id)

        def load_y(self, patient_id):
            return self._shadowed.load_segm(patient_id)

        @property
        def n_chans_out(self):
            return self.n_classes

    return TaskedDataset(dataset)


def make_cached(dataset) -> Dataset:
    n = len(dataset.patient_ids)

    class CachedDataset(Proxy):
        @lru_cache(n)
        def load_x(self, patient_id):
            return self._shadowed.load_x(patient_id)

        @lru_cache(n)
        def load_y(self, patient_id):
            return self._shadowed.load_y(patient_id)

    return CachedDataset(dataset)
