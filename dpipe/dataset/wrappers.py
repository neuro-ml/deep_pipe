"""Wrappers change the dataset's behaviour."""
import functools
from os.path import join as jp
from itertools import chain
from typing import Sequence, Callable, Iterable
from collections import ChainMap, namedtuple
from warnings import warn

import numpy as np

import dpipe.medim as medim
from dpipe.medim.checks import join
from dpipe.medim.itertools import zdict
from dpipe.medim.utils import cache_to_disk
from .base import Dataset, SegmentationDataset


class Proxy:
    def __init__(self, shadowed):
        self._shadowed = shadowed

    def __getattr__(self, name):
        return getattr(self._shadowed, name)


def cache_methods(instance=None, methods: Iterable[str] = None, dataset=None) -> Dataset:
    """Cache the ``instamce``'s ``methods``."""
    cache = functools.lru_cache(None)
    if dataset is not None:
        warn('the argument "dataset" is deprecated. unse "instance" instead')
        assert instance is None
        instance = dataset

    new_methods = {method: staticmethod(cache(getattr(instance, method))) for method in methods}
    proxy = type('Cached', (Proxy,), new_methods)
    return proxy(instance)


def cache_methods_to_disk(instance, base_path: str, **methods: str) -> Dataset:
    """
    Cache the ``instamce``'s ``methods`` to disk.

    Parameters
    ----------
    instance
    base_path
    methods
         each keyword argument has the form `method_name=path_to_cache`, the path is relative to ``base_path``.
         The methods are assumed to take a single argument of type `str`.

    Notes
    -----
    The values are cached and loaded using `np.save` and `np.load` respectively.
    """

    def path_by_id(path, identifier):
        return jp(path, f'{identifier}.npy')

    def load(path, identifier):
        return np.load(path_by_id(path, identifier))

    def save(value, path, identifier):
        np.save(path_by_id(path, identifier), value)

    new_methods = {method: staticmethod(cache_to_disk(getattr(instance, method), jp(base_path, path), load, save))
                   for method, path in methods.items()}
    proxy = type('CachedToDisk', (Proxy,), new_methods)
    return proxy(instance)


cache_segmentation_dataset = functools.partial(cache_methods, methods=['load_image', 'load_segm'])


def apply(instance, **methods: Callable):
    """
    Applies a given function to the output of a given method.

    Parameters
    ----------
    instance
    methods
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


def change_ids(dataset: Dataset, change_id: Callable, methods: Iterable[str] = ()) -> Dataset:
    """
    Change the ``dataset``'s ids according to the ``change_id`` function and adapt the provided ``methods``
    to work with the new ids.

    Parameters
    ----------
    dataset
    change_id: Callable(str) -> str
    methods
        the list of methods to be adapted. Each method takes a single argument - the identifier.
    """
    assert 'ids' not in methods
    ids = tuple(map(change_id, dataset.ids))
    if len(set(ids)) != len(ids):
        raise ValueError('The resulting ids are not unique.')
    new_to_old = zdict(ids, dataset.ids)

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
    """Binds the ``methods`` to the last proxy."""
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
            self._shapes = {}

        def load_image(self, identifier):
            return self._extract(identifier, load_image(identifier))

        def load_segm(self, identifier):
            return self._extract(identifier, dataset.load_segm(identifier))

        def get_start_stop(self, identifier):
            if identifier not in self._start_stop:
                img = load_image(identifier)
                mask = np.any(img > img.min(axis=tuple(range(1, img.ndim)), keepdims=True), axis=0)
                self._start_stop[identifier] = tuple(medim.box.mask2bounding_box(mask))
                self._shapes[identifier] = img.shape
            return self._start_stop[identifier]

        def get_original_padding(self, identifier):
            start, stop = self.get_start_stop(identifier)
            return tuple(zip(start, self._shapes[identifier] - stop))

        def _extract(self, identifier, tensor):
            start, stop = self.get_start_stop(identifier)
            return tensor[(..., *medim.utils.build_slices(start, stop))]

    return BBoxedDataset(dataset)


def normalized(dataset: Dataset, mean: bool = True, std: bool = True, drop_percentile: int = None) -> Dataset:
    return apply(dataset, load_image=functools.partial(medim.prep.normalize_multichannel_image, mean=mean, std=std,
                                                       drop_percentile=drop_percentile))


def merge(*datasets: Dataset, methods: Sequence[str] = (), attributes: Sequence[str] = ()) -> Dataset:
    """
    Merge several ``datasets`` into one by preserving the provided ``methods`` and ``attributes``.

    Parameters
    ----------
    datasets
    methods
        the list of methods to be preserved. Each method must take a single argument - the identifier.
    attributes
        the list of attributes to be preserved. For each dataset their values must coincide.
    """
    clash = set(methods) & set(attributes)
    if clash:
        raise ValueError(f'Method names clash with attribute names: {join(clash)}.')
    ids = tuple(id_ for dataset in datasets for id_ in dataset.ids)
    if len(set(ids)) != len(ids):
        raise ValueError('The ids are not unique.')

    preserved_attributes = []
    for attribute in attributes:
        values = {getattr(dataset, attribute) for dataset in datasets}
        preserved_attributes.append(list(values)[0])
        if len(values) != 1:
            raise ValueError(f'Datasets have different values of attribute "{attribute}".')

    def decorator(method_name):
        def wrapper(identifier):
            return getattr(id_to_dataset[identifier], method_name)(identifier)

        return wrapper

    id_to_dataset = ChainMap(*({id_: dataset for id_ in dataset.ids} for dataset in datasets))
    Merged = namedtuple('Merged', chain(['ids'], methods, attributes))
    return Merged(*chain([ids], map(decorator, methods), preserved_attributes))


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
