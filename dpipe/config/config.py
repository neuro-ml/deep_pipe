from functools import partial
from typing import Iterable

from dpipe.modules.batch_iterators import build_batch_iter_factory
from dpipe.modules.batch_iterators.config import batch_iter_name2batch_iter
from dpipe.modules.datasets.base import make_cached, make_tasked_dataset,\
    Dataset
from dpipe.modules.datasets.config import dataset_name2dataset
from dpipe.modules.dl import optimize, Model, FrozenModel
from dpipe.modules.dl.config import predictor_name2predictor, loss_name2loss
from dpipe.modules.dl.model_cores.config import model_core_name2model_core
from dpipe.modules.splits.config import get_split_name2get_split
from dpipe.modules.trains.config import train_name2train
from dpipe.experiments.config import experiment_name2build_experiment

__all__ = ['config_dataset', 'config_split', 'config_model',
           'config_frozen_model', 'config_batch_iter',
           'config_train', 'config_build_experiment']

default_config = {
    'n_iters_per_epoch': None,
    'dataset_cached': True
}

module_type2module_constructor_mapping = {
    'dataset': dataset_name2dataset,
    'split': get_split_name2get_split,
    'model_core': model_core_name2model_core,
    'batch_iter': batch_iter_name2batch_iter,
    'train': train_name2train,
    'predict': predictor_name2predictor,
    'loss': loss_name2loss,
    'experiment': experiment_name2build_experiment
}


def _config_module_builder(module_type, config, **kwargs):
    name = config[module_type]
    try:
        constructor = module_type2module_constructor_mapping[module_type][name]
    except KeyError:
        raise ValueError(
            'Wrong module name provided\n' + \
            'Module type: {}\n'.format(module_type) + \
            'Provided name: {}\n'.format(name) + \
            'Available names: {}\n'.format(
                [*module_type2module_constructor_mapping[module_type].keys()]))
    params = config.get('{}__params'.format(module_type), {})
    return partial(constructor, **params, **kwargs)


def _config_object(module_type, config, **kwargs):
    return _config_module_builder(module_type, config, **kwargs)()


def config_dataset(config) -> Dataset:
    dataset = _config_object('dataset', config)
    dataset = make_tasked_dataset(dataset, config['dataset_task'])
    return dataset if not config['dataset_cached'] else make_cached(dataset)


def config_split(config, dataset: Dataset) -> Iterable:
    return _config_object('split', config, dataset=dataset)


def config_build_experiment(config):
    return _config_module_builder('experiment', config)


def config_train(config) -> callable:
    return _config_module_builder('train', config)


def _config_optimizer(config) -> callable:
    return partial(optimize, tf_optimizer_name=config['optimizer'],
                   **config.get('optimizer__params', {}))


def config_model(config, dataset: Dataset) -> Model:
    predict = _config_module_builder('predict', config)
    loss = _config_module_builder('loss', config)
    optimizer = _config_optimizer(config)

    model_core = _config_object('model_core', config,
                                n_chans_in=dataset.n_chans_mscan,
                                n_chans_out=dataset.n_chans_out)
    return Model(model_core, predict=predict, loss=loss, optimize=optimizer)


def config_frozen_model(config, dataset: Dataset) -> FrozenModel:
    predict = _config_module_builder('predict', config)
    model_core = _config_object('model_core', config,
                                n_chans_in=dataset.n_chans_mscan,
                                n_chans_out=dataset.n_chans_out)
    return FrozenModel(model_core, predict=predict)


def config_batch_iter(config, *, ids, dataset) -> Iterable:
    get_batch_iter = _config_module_builder('batch_iter', config, ids=ids,
                                            dataset=dataset)

    return build_batch_iter_factory(get_batch_iter, config['n_iters_per_epoch'])
