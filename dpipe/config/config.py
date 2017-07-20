from functools import partial
from typing import Iterable

from dpipe.modules.batch_iterators import build_batch_iter
from dpipe.modules.batch_iterators.config import batch_iter_name2batch_iter
from dpipe.modules.datasets.base import make_cached, Dataset
from dpipe.modules.datasets.config import dataset_name2dataset
from dpipe.modules.dl import Optimizer, Model, FrozenModel
from dpipe.modules.dl.model_cores.config import model_core_name2model_core
from dpipe.modules.splits.config import get_split_name2get_split
from dpipe.modules.trains.config import train_name2train
from dpipe.experiments.config import splitter_name2build_experiment

__all__ = ['config_dataset', 'config_split', 'config_optimizer',
           'config_model', 'config_frozen_model', 'config_batch_iter',
           'config_train', 'config_build_experiment']

default_config = {
    "n_iters_per_epoch": None,

    "dataset_cached": True,
    "dataset__params": {},

    "split__params": {},

    "optimizer": "optimizer",
    "optimizer__params": {},

    "model_core__params": {},
    "restore_model_path": None,
    "batch_iter__params": {},

    "train": None,
    "train__params": {},
}

module_type2module_constructor_mapping = {
    'dataset': dataset_name2dataset,
    'split': get_split_name2get_split,
    'optimizer': {'optimizer': Optimizer},
    'model_core': model_core_name2model_core,
    'batch_iter': batch_iter_name2batch_iter,
    'train': train_name2train,
}


def config_module_builder(module_type, config, **kwargs):
    name = config[module_type]
    params = config[f'{module_type}__params']

    return partial(module_type2module_constructor_mapping[module_type][name],
                   **params, **kwargs)


def config_object(module_type, config, **kwargs):
    return config_module_builder(module_type, config, **kwargs)()


def config_dataset(config) -> Dataset:
    dataset = config_object('dataset', config)
    if config['dataset_cached']:
        dataset = make_cached(dataset)

    return dataset


def config_split(config, dataset: Dataset) -> Iterable:
    return config_object('split', config, dataset=dataset)


def config_build_experiment(config):
    build_experiment = splitter_name2build_experiment[config['split']]
    return build_experiment


def config_optimizer(config) -> Optimizer:
    return config_object('optimizer', config)


def config_train(config) -> callable:
    return config_module_builder('train', config)


def config_model(config, *, optimizer, n_chans_in, n_chans_out) -> Model:
    model_core = config_object('model_core', config, n_chans_in=n_chans_in,
                               n_chans_out=n_chans_out)
    return Model(model_core, optimizer=optimizer)


def config_frozen_model(config, *, n_chans_in, n_chans_out) -> FrozenModel:
    model_core = config_object('model_core', config, n_chans_in=n_chans_in,
                               n_chans_out=n_chans_out)
    return FrozenModel(model_core)


def config_batch_iter(config, *, ids, dataset) -> Iterable:
    get_batch_iter = config_module_builder('batch_iter', config, ids=ids,
                                           dataset=dataset)

    return build_batch_iter(get_batch_iter, config['n_iters_per_epoch'])
