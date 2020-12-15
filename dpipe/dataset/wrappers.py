"""
Wrappers change the dataset's behaviour.
See the :doc:`tutorials/wrappers` tutorial for more details.
"""
import functools
from itertools import chain
from types import MethodType, FunctionType
from typing import Sequence, Callable, Iterable
from collections import ChainMap, namedtuple
from pathlib import Path

import numpy as np

from dpipe.checks import join
from dpipe.io import save_numpy, PathLike, load_or_create, load_numpy
from dpipe.itertools import zdict, collect
from dpipe.im.preprocessing import normalize
from .base import Dataset


class Proxy:
    """Base class for all wrappers."""

    def __init__(self, shadowed):
        self._shadowed = shadowed

    def __getattr__(self, name):
        return getattr(self._shadowed, name)

    def __dir__(self):
        return list(set(super().__dir__()) | set(dir(self._shadowed)))


@collect
def _get_public_methods(instance):
    for attr in dir(instance):
        if not attr.startswith('_') and isinstance(getattr(instance, attr), (MethodType, FunctionType)):
            yield attr


def cache_methods(instance, methods: Iterable[str] = None, maxsize: int = None):
    """Cache the ``instance``'s ``methods``. If ``methods`` is None, all public methods will be cached."""
    if methods is None:
        methods = _get_public_methods(instance)

    cache = functools.lru_cache(maxsize)
    new_methods = {method: staticmethod(cache(getattr(instance, method))) for method in methods}
    proxy = type('Cached', (Proxy,), new_methods)
    return proxy(instance)


def cache_methods_to_disk(instance, base_path: PathLike, loader: Callable = load_numpy, saver: Callable = save_numpy,
                          **methods: str):
    """
    Cache the ``instance``'s ``methods`` to disk.

    Parameters
    ----------
    instance
        arbitrary object
    base_path: str
        the path, all other paths of ``methods`` relative to.
    methods: str
        each keyword argument has the form ``method_name=path_to_cache``.
        The methods are assumed to take a single argument of type ``str``.
    loader
        loads a single object given its path.
    saver: Callable(value, path)
        saves a single object to the given path.
    """
    base_path = Path(base_path)

    def decorator(method, folder):
        method = getattr(instance, method)
        path = base_path / folder
        path.mkdir(parents=True, exist_ok=True)

        @functools.wraps(method)
        def wrapper(identifier, *args, **kwargs):
            return load_or_create(
                path / f'{identifier}.npy', method, identifier, *args, **kwargs, save=saver, load=loader)

        return staticmethod(wrapper)

    new_methods = {method: decorator(method, folder) for method, folder in methods.items()}
    proxy = type('CachedToDisk', (Proxy,), new_methods)
    return proxy(instance)


def apply(instance, **methods: Callable):
    """
    Applies a given function to the output of a given method.

    Parameters
    ----------
    instance
        arbitrary object
    methods: Callable
        each keyword argument has the form ``method_name=func_to_apply``.
        ``func_to_apply`` is applied to the ``method_name`` method.

    Examples
    --------
    >>> # normalize will be applied to the output of load_image
    >>> dataset = apply(base_dataset, load_image=normalize)
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
        arbitrary object
    attributes
        each keyword argument has the form ``attr_name=attr_value``.
    """
    proxy = type('SetAttr', (Proxy,), attributes)
    return proxy(instance)


def change_ids(dataset: Dataset, change_id: Callable, methods: Iterable[str] = None) -> Dataset:
    """
    Change the ``dataset``'s ids according to the ``change_id`` function and adapt the provided ``methods``
    to work with the new ids.

    Parameters
    ----------
    dataset: Dataset
        the dataset to perform ids changing on.
    change_id: Callable(str) -> str
        the method which allows change ids. Output ids should be unique as well as old ids.
    methods: Iterable[str]
        the list of methods to be adapted. Each method takes a single argument - the identifier.
    """
    if methods is None:
        methods = _get_public_methods(dataset)

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


def merge(*datasets: Dataset, methods: Sequence[str] = None, attributes: Sequence[str] = ()) -> Dataset:
    """
    Merge several ``datasets`` into one by preserving the provided ``methods`` and ``attributes``.

    Parameters
    ----------
    datasets: Dataset
        sequence of datasets.
    methods: Sequence[str], None, optional
        the list of methods to be preserved. Each method should take an identifier as its first argument.
        If ``None``, all the common methods will be preserved.
    attributes: Sequence[str]
        the list of attributes to be preserved. For each dataset their values should be the same.
        Default is the empty sequence ``()``.
    """
    if methods is None:
        methods = set(_get_public_methods(datasets[0]))
        for dataset in datasets:
            methods = methods & set(_get_public_methods(dataset))

    clash = set(methods) & set(attributes)
    if clash:
        raise ValueError(f'Method names clash with attribute names: {join(clash)}.')
    ids = tuple(id_ for dataset in datasets for id_ in dataset.ids)
    if len(set(ids)) != len(ids):
        raise ValueError('The ids are not unique.')

    preserved_attributes = []
    for attribute in attributes:
        # can't use a set here, because not all attributes can be hashed
        values = []
        for dataset in datasets:
            value = getattr(dataset, attribute)
            if value not in values:
                values.append(value)

        if len(values) != 1:
            raise ValueError(f'Datasets have different values of attribute "{attribute}".')

        preserved_attributes.append(values[0])

    def decorator(method_name):
        def wrapper(identifier, *args, **kwargs):
            if identifier not in id_to_dataset:
                raise KeyError(f"This dataset doesn't contain the id {identifier}")

            return getattr(id_to_dataset[identifier], method_name)(identifier, *args, **kwargs)

        return wrapper

    id_to_dataset = ChainMap(*({id_: dataset for id_ in dataset.ids} for dataset in datasets))
    Merged = namedtuple('Merged', chain(['ids'], methods, attributes))
    return Merged(*chain([ids], map(decorator, methods), preserved_attributes))


def apply_mask(dataset: Dataset, mask_modality_id: int = -1, mask_value: int = None) -> Dataset:
    """
    Applies the ``mask_modality_id`` modality as the binary mask to the other modalities
    and remove the mask from sequence of modalities.

    Parameters
    ----------
    dataset: Dataset
        dataset which is used in the current task.
    mask_modality_id: int
        the index of mask in the sequence of modalities.
        Default is ``-1``, which means the last modality will be used as the mask.
    mask_value: int, None, optional
        the value in the mask to filter other modalities with.
        If ``None``, greater than zero filtering will be applied. Default is ``None``.

    Examples
    --------
    >>> modalities = ['flair', 't1', 'brain_mask']  # we are to apply brain mask to other modalities
    >>> target = 'target'
    >>>
    >>> dataset = apply_mask(
    >>>     dataset=Wmh2017(
    >>>         data_path=data_path,
    >>>         modalities=modalities,
    >>>         target=target
    >>>     ),
    >>>     mask_modality_id=-1,
    >>>     mask_value=1
    >>> )
    """

    class MaskedDataset(Proxy):
        def load_image(self, patient_id):
            images = self._shadowed.load_image(patient_id)
            mask = images[mask_modality_id]

            mask_bin = mask > 0 if mask_value is None else mask == mask_value
            if not np.sum(mask_bin) > 0:
                raise ValueError('The obtained mask is empty')

            images = [image * mask for image in images[:-1]]
            return np.array(images)

        @property
        def n_chans_image(self):
            return self._shadowed.n_chans_image - 1

    return MaskedDataset(dataset)
