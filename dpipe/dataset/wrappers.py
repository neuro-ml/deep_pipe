import functools

import numpy as np

from .base import Dataset
import dpipe.medim as medim


class Proxy:
    def __init__(self, shadowed):
        self._shadowed = shadowed

    def __getattr__(self, name):
        return getattr(self._shadowed, name)


def make_cached(dataset: Dataset) -> Dataset:
    n = len(dataset.patient_ids)

    class CachedDataset(Proxy):
        @functools.lru_cache(n)
        def load_mscan(self, patient_id):
            return self._shadowed.load_mscan(patient_id)

        @functools.lru_cache(n)
        def load_segm(self, patient_id):
            return self._shadowed.load_segm(patient_id)

        @functools.lru_cache(n)
        def load_msegm(self, patient_id):
            return self._shadowed.load_msegm(patient_id)

    return CachedDataset(dataset)


def make_bbox_extraction(dataset: Dataset) -> Dataset:
    # Use this small cache to speed up data loading. Usually users load
    # all scans for the same person at the same time
    load_mscan = functools.lru_cache(3)(dataset.load_mscan)

    class BBoxedDataset(Proxy):
        def load_mscan(self, patient_id):
            img = load_mscan(patient_id)
            mask = np.any(img > 0, axis=0)
            return medim.bb.extract([img], mask)[0]

        def load_segm(self, patient_id):
            img = self._shadowed.load_segm(patient_id)
            mask = np.any(load_mscan(patient_id) > 0, axis=0)
            return medim.bb.extract([img], mask=mask)[0]

        def load_msegm(self, patient_id):
            img = self._shadowed.load_msegm(patient_id)
            mask = np.any(load_mscan(patient_id) > 0, axis=0)
            return medim.bb.extract([img], mask=mask)[0]

    return BBoxedDataset(dataset)


def make_normalized(dataset: Dataset, mean=True, std=True,
                    drop_percentile: int = None) -> Dataset:
    class NormalizedDataset(Proxy):
        def load_mscan(self, patient_id):
            img = self._shadowed.load_mscan(patient_id)
            return medim.prep.normalize_mscan(img, mean=mean, std=std,
                                              drop_percentile=drop_percentile)

    return NormalizedDataset(dataset)


def make_normalized_sub(dataset: Dataset) -> Dataset:
    class NormalizedDataset(Proxy):
        def load_mscan(self, patient_id):
            mscan = self._shadowed.load_mscan(patient_id)
            mask = np.any(mscan > 0, axis=0)
            mscan_inner = medim.bb.extract([mscan], mask)[0]

            mscan = mscan / mscan_inner.std(axis=(1, 2, 3), keepdims=True)

            return mscan

    return NormalizedDataset(dataset)


def add_groups(dataset: Dataset, group_col) -> Dataset:
    class GroupedFromMetadata(Proxy):
        @property
        def groups(self):
            return self.dataFrame[group_col].as_matrix()

    return GroupedFromMetadata(dataset)
