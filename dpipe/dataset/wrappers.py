"""Wrappers aim to change the dataset's behaviour"""

import functools
from itertools import chain
from typing import List, Sequence, Callable, Iterable
from collections import ChainMap, namedtuple

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
    """
    cache = functools.lru_cache(len(dataset.ids))

    new_methods = {method: staticmethod(cache(getattr(dataset, method))) for method in methods}
    proxy = type('Cached', (Proxy,), new_methods)
    return proxy(dataset)


cache_segmentation_dataset = functools.partial(cache_methods, methods=['load_image', 'load_segm'])


def apply(instance, **methods: Callable):
    """
    Applies a given function to the output of a given method.

    Parameters
    ----------
    instance
    methods: Callable
        each keyword argument has the form `method_name=func_to_apply`.
        `func_to_apply` is applied to the `method_name` method.
    """

    def decorator(method, func):
        @functools.wraps(method)
        def wrapper(*args, **kwargs):
            return func(method(*args, **kwargs))

        return staticmethod(wrapper)

    new_methods = {method: decorator(getattr(instance, method), func) for method, func in methods.items()}
    proxy = type('Apply', (Proxy,), new_methods)
    return proxy(instance)


def set_attributes(instance, **attributes):
    """
    Sets or overwrites attributes with those provided as keyword arguments.

    Parameters
    ----------
    instance
    attributes
        each keyword argument has the form `attr_name=attr_value`.
    """
    proxy = type('SetAttr', (Proxy,), attributes)
    return proxy(instance)


def change_ids(dataset: Dataset, change_id: Callable, methods: Iterable[str] = None) -> Dataset:
    """
    Change the `dataset`'s ids according to the `change_id` function and adapt the provided `methods`
    to work with the new ids.

    Parameters
    ----------
    dataset: Dataset
    change_id: Callable(str) -> str
    methods: str, optional
        the list of methods to be adapted. Each method takes a single argument - the identifier.
    """
    assert 'ids' not in methods
    ids = tuple(map(change_id, dataset.ids))
    assert len(set(ids)) == len(ids), 'The resulting ids are not unique'
    new_to_old = dict(zip(ids, dataset.ids))
    methods = set(methods or []) | {'load_image'}

    def decorator(method):
        @functools.wraps(method)
        def wrapper(identifier):
            return method(new_to_old[identifier])

        return staticmethod(wrapper)

    attributes = {method: decorator(getattr(dataset, method)) for method in methods}
    attributes['ids'] = ids
    proxy = type('ChangedID', (Proxy,), attributes)
    return proxy(dataset)


def rebind(instance, methods):
    """Binds the `methods` to the last proxy."""

    new_methods = {method: getattr(instance, method).__func__ for method in methods}
    proxy = type('Rebound', (Proxy,), new_methods)
    return proxy(instance)


def bbox_extraction(dataset: SegmentationDataset) -> SegmentationDataset:
    # Use this small cache to speed up data loading when calculating the mask
    load_image = functools.lru_cache(1)(dataset.load_image)

    class BBoxedDataset(Proxy):
        def __init__(self, shadowed):
            super().__init__(shadowed)
            self._start_stop = {}

        def load_image(self, identifier):
            return self._extract(identifier, load_image(identifier))

        def load_segm(self, identifier):
            return self._extract(identifier, dataset.load_segm(identifier))

        def get_start_stop(self, identifier):
            if identifier not in self._start_stop:
                img = load_image(identifier)
                mask = np.any(img > img.min(axis=tuple(range(1, img.ndim)), keepdims=True), axis=0)
                self._start_stop[identifier] = tuple(medim.bb.get_start_stop(mask))
            return self._start_stop[identifier]

        def _extract(self, identifier, tensor):
            start, stop = self.get_start_stop(identifier)
            return tensor[(..., *medim.bb.build_slices(start, stop))]

    return BBoxedDataset(dataset)


def normalized(dataset: Dataset, mean: bool = True, std: bool = True, drop_percentile: int = None) -> Dataset:
    return apply(dataset, load_image=functools.partial(medim.prep.normalize_multichannel_image, mean=mean, std=std,
                                                       drop_percentile=drop_percentile))


def merge(*datasets: Dataset, methods: Sequence[str] = None) -> Dataset:
    """
    Merge several datasets into one by preserving the provided methods.

    Parameters
    ----------
    datasets: Dataset
    methods: Sequence[str], optional
        the list of methods to be preserved. Each method must take a single argument - the identifier.
    """

    ids = tuple(id_ for dataset in datasets for id_ in dataset.ids)
    assert len(set(ids)) == len(ids), 'The ids are not unique'
    n_chans_images = {dataset.n_chans_image for dataset in datasets}
    assert len(n_chans_images) == 1, 'Each dataset must have the same number of channels'

    id_to_dataset = ChainMap(*({id_: dataset for id_ in dataset.ids} for dataset in datasets))
    n_chans_image = list(n_chans_images)[0]
    methods = list(set(methods or []) | {'load_image'})

    def decorator(method_name):
        def wrapper(identifier):
            return getattr(id_to_dataset[identifier], method_name)(identifier)

        return wrapper

    Merged = namedtuple('Merged', methods + ['ids', 'n_chans_image'])
    return Merged(*chain(map(decorator, methods), [ids, n_chans_image]))


# TODO: deprecated
# Deprecated
# ----------

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
    assert all(dataset.n_chans_image == datasets[0].n_chans_image for dataset in datasets)

    patient_id2dataset = ChainMap(*({pi: dataset for pi in dataset.ids} for dataset in datasets))

    ids = sorted(list(patient_id2dataset.keys()))

    class MergedDataset(Proxy):
        @property
        def ids(self):
            return ids

        def load_image(self, patient_id):
            return patient_id2dataset[patient_id].load_image(patient_id)

        def load_segm(self, patient_id):
            return patient_id2dataset[patient_id].load_segm(patient_id)

    return MergedDataset(datasets[0])


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
