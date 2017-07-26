from typing import Iterable

from dpipe.experiments.config import experiment_name2build_experiment
from dpipe.modules.batch_iterators import build_batch_iter_factory
from dpipe.modules.batch_iterators.config import batch_iter_name2batch_iter
from dpipe.modules.datasets.base import make_cached, make_tasked_dataset, \
    Dataset
from dpipe.modules.datasets.config import dataset_name2dataset
from dpipe.modules.splits.config import get_split_name2get_split
from .utils import config_object, config_partial

__all__ = ['config_dataset', 'config_split', 'config_batch_iter',
           'config_build_experiment']

module_builders = {
    'dataset': dataset_name2dataset,
    'split': get_split_name2get_split,
    'batch_iter': batch_iter_name2batch_iter,
    'experiment': experiment_name2build_experiment
}


def config_dataset(config) -> Dataset:
    dataset = config_object('dataset', config, module_builders)
    dataset = make_tasked_dataset(dataset, config['dataset_task'])
    return dataset if not config['dataset_cached'] else make_cached(dataset)


def config_split(config, dataset: Dataset) -> Iterable:
    return config_object('split', config, module_builders, dataset=dataset)


def config_build_experiment(config):
    return config_partial('experiment', config, module_builders)


def config_batch_iter(config, *, ids, dataset) -> Iterable:
    get_batch_iter = config_partial('batch_iter', config, module_builders,
                                     ids=ids, dataset=dataset)

    return build_batch_iter_factory(get_batch_iter, config['n_iters_per_epoch'])
