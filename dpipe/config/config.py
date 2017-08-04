from dpipe.datasets.base import make_cached, Dataset, DataLoader
from dpipe.datasets.config import dataset_name2dataset
from dpipe.splits.config import split_name2get_split
from dpipe.experiments.config import experiment_name2experiment_builder
from dpipe.batch_iters.config import batch_iter_fin_name2batch_iter, \
    batch_iter_inf_name2batch_iter
from dpipe.batch_iters.batch_iter_factory import BatchIterFactory, \
    BatchIterFactoryFin, BatchIterFactoryInf
from .utils import config_object, config_partial

__all__ = ['config_dataset', 'config_split', 'config_batch_iter_factory',
           'config_experiment', 'config_data_loader']


def config_dataset(config) -> Dataset:
    dataset = config_object(config, 'dataset', dataset_name2dataset)
    return make_cached(dataset) if config['dataset_cached'] else dataset


def config_data_loader(config, dataset) -> DataLoader:
    return DataLoader(dataset=dataset, problem=config['problem'])


def config_split(config, *, dataset: Dataset):
    return config_object(config, 'split', split_name2get_split, dataset=dataset)


def config_experiment(config, *, experiment_path, config_path, split):
    return config_object(
        config, 'experiment', experiment_name2experiment_builder, split=split,
        experiment_path=experiment_path, config_path=config_path)


def config_batch_iter_factory(config, *, ids, dataset) -> BatchIterFactory:
    assert ('batch_iter_fin' in config) ^ ('batch_iter_inf' in config)

    n_iters_per_epoch = config.get('n_iters_per_epoch', None)

    data_loader = config_data_loader(config, dataset)
    if 'batch_iter_fin' in config:
        assert n_iters_per_epoch is None

        get_batch_iter = config_partial(
            config, 'batch_iter_fin', batch_iter_fin_name2batch_iter, ids=ids,
            data_loader=data_loader
        )
        batch_iter_factory = BatchIterFactoryFin(get_batch_iter)
    else:
        assert n_iters_per_epoch is not None

        get_batch_iter = config_partial(
            'batch_iter_inf', config, batch_iter_inf_name2batch_iter, ids=ids,
            data_loader=data_loader
        )
        batch_iter_factory = BatchIterFactoryInf(get_batch_iter,
                                                 n_iters_per_epoch)
    return batch_iter_factory
