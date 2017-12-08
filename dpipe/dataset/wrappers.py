"""Wrappers aim to change the dataset's behaviour"""

import functools
from typing import List, Sequence
from collections import ChainMap

import numpy as np
import dpipe.medim as medim
from .base import Dataset, SegmentationDataset


class Proxy:
    def __init__(self, shadowed):
        self._shadowed = shadowed

    def __getattr__(self, name):
        return getattr(self._shadowed, name)


def cache_methods(dataset: Dataset, methods: Sequence[str]) -> Dataset:
    """
    Wrapper that caches the dataset's methods.

    Parameters
    ----------
    dataset: Dataset
    methods: Sequence
        a sequence of methods names to be cached

    Returns
    -------
    cached_dataset: Dataset
        a wrapped dataset
    """
    cache = functools.lru_cache(len(dataset.ids))

    new_methods = {method: staticmethod(cache(getattr(dataset, method))) for method in methods}
    proxy = type('Cached', (Proxy,), new_methods)
    return proxy(dataset)


def cache_segmentation_dataset(dataset: Dataset) -> Dataset:
    return cache_methods(dataset, ("load_image", "load_segm", "load_msegm"))


def apply(instance, methods, function):
    def decorator(method):
        def wrapper(self, *args, **kwargs):
            return function(method(*args, **kwargs))

        return wrapper

    new_methods = {method: decorator(getattr(instance, method)) for method in methods}
    proxy = type('Apply', (Proxy,), new_methods)
    return proxy(instance)


def apply_mask(dataset: SegmentationDataset, mask_modality_id: int = None,
               mask_value: int = None) -> SegmentationDataset:
    class MaskedDataset(Proxy):
        def load_image(self, patient_id):
            images = self._shadowed.load_image(patient_id)
            mask = images[mask_modality_id]
            mask_bin = (mask > 0 if mask_value is None else mask == mask_value)
            assert np.sum(mask_bin) > 0, 'The obtained mask is empty'
            images = [image * mask for image in images[:-1]]
            return np.array(images)

        @property
        def n_chans_image(self):
            return self._shadowed.n_chans_image - 1

    return dataset if mask_modality_id is None else MaskedDataset(dataset)


def bbox_extraction(dataset: SegmentationDataset) -> SegmentationDataset:
    # Use this small cache to speed up data loading. Usually users load
    # all scans for the same person at the same time
    load_image = functools.lru_cache(3)(dataset.load_image)

    class BBoxedDataset(Proxy):
        def load_image(self, patient_id):
            img = load_image(patient_id)
            mask = np.any(img > 0, axis=0)
            return medim.bb.extract([img], mask)[0]

        def load_segm(self, patient_id):
            img = self._shadowed.load_segm(patient_id)
            mask = np.any(load_image(patient_id) > 0, axis=0)
            return medim.bb.extract([img], mask=mask)[0]

        def load_msegm(self, patient_id):
            img = self._shadowed.load_msegm(patient_id)
            mask = np.any(load_image(patient_id) > 0, axis=0)
            return medim.bb.extract([img], mask=mask)[0]

    return BBoxedDataset(dataset)


def normalized(dataset: SegmentationDataset, mean, std, drop_percentile: int = None) -> SegmentationDataset:
    class NormalizedDataset(Proxy):
        def load_image(self, idx):
            img = self._shadowed.load_image(idx)
            return medim.prep.normalize_mscan(img, mean=mean, std=std,
                                              drop_percentile=drop_percentile)

    return NormalizedDataset(dataset)


def add_groups_from_df(dataset: Dataset, group_col: str) -> Dataset:
    class GroupedFromMetadata(Proxy):
        @property
        def groups(self):
            return self._shadowed.df[group_col].as_matrix()

    return GroupedFromMetadata(dataset)


def add_groups_from_ids(dataset: Dataset, separator: str) -> Dataset:
    roots = [pi.split(separator)[0] for pi in dataset.ids]
    root2group = dict(map(lambda x: (x[1], x[0]), enumerate(set(roots))))
    groups = tuple(root2group[pi.split(separator)[0]] for pi in dataset.ids)

    class GroupsFromIDs(Proxy):
        @property
        def groups(self):
            return groups

    return GroupsFromIDs(dataset)


def merge_datasets(datasets: List[SegmentationDataset]) -> SegmentationDataset:
    [np.testing.assert_array_equal(a.segm2msegm_matrix, b.segm2msegm_matrix)
     for a, b, in zip(datasets, datasets[1:])]

    assert all(dataset.n_chans_image == datasets[0].n_chans_image for dataset in datasets)

    patient_id2dataset = ChainMap(*({pi: dataset for pi in dataset.ids}
                                    for dataset in datasets))

    ids = sorted(list(patient_id2dataset.keys()))

    class MergedDataset(Proxy):
        @property
        def ids(self):
            return ids

        def load_image(self, patient_id):
            return patient_id2dataset[patient_id].load_image(patient_id)

        def load_segm(self, patient_id):
            return patient_id2dataset[patient_id].load_segm(patient_id)

        def load_msegm(self, patient_id):
            return patient_id2dataset[patient_id].load_msegm(patient_id)

    return MergedDataset(datasets[0])


def weighted(dataset: Dataset, thickness: str) -> Dataset:
    class WeightedBoundariesDataset(Proxy):
        def load_weighted_mask(self, patient_id) -> np.array:
            paths = [self.df[thickness].loc[patient_id]]
            image = self._load_by_paths(paths)

            return image

    return WeightedBoundariesDataset(dataset)


class Padded(Proxy):
    def __init__(self, dataset: SegmentationDataset, shape: list, axes: list):
        super().__init__(dataset)
        self.shape = shape
        self.axes = axes

    def load_image(self, patient_id):
        img = self._shadowed.load_image(patient_id)
        return medim.preprocessing.pad(img, self.shape, self.axes)

    def load_segm(self, patient_id):
        img = self._shadowed.load_segm(patient_id)
        return medim.preprocessing.pad(img, self.shape, self.axes)

    def load_msegm(self, patient_id):
        img = self._shadowed.load_msegm(patient_id)
        return medim.preprocessing.pad(img, self.shape, self.axes)
