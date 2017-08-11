from dpipe.batch_iter_factory.base import BatchIterFactory, \
    BatchIterFactoryFin, BatchIterFactoryInf
from dpipe.batch_iters.config import name2batch_iter_fin
from dpipe.datasets.base import make_cached, Dataset
from dpipe.datasets.config import name2dataset
from dpipe.experiments.config import name2experiment
from dpipe.splits.config import name2split
from .utils import config_object, config_partial

__all__ = ['config_dataset', 'config_split', 'config_batch_iter_factory',
           'config_experiment', 'config_data_loader']


def config_dataset(config) -> Dataset:
    dataset = config_object(config, 'dataset', name2dataset)
    return make_cached(dataset) if config['dataset_cached'] else dataset


def config_split(config, *, dataset: Dataset):
    return config_object(config, 'split', name2split, dataset=dataset)


def config_experiment(config, *, experiment_path, config_path, split):
    return config_object(
        config, 'experiment', name2experiment, config_path=config_path,
        experiment_path=experiment_path, split=split,
    )


def config_batch_iter(config):
    assert ('batch_iter_fin' in config) ^ ('batch_iter_inf' in config)

    n_iters_per_epoch = config.get('n_iters_per_epoch', None)

    if 'batch_iter_fin' in config:
        assert n_iters_per_epoch is None

        get_batch_iter = config_partial(config, 'batch_iter_fin',
                                        name2batch_iter_fin)
    else:
        assert n_iters_per_epoch is not None

        get_batch_iter = config_partial(
            'batch_iter_inf', config, batch_iter_inf_name2batch_iter, ids=ids,
            data_loader=data_loader
        )


def config_batch_iter_factory(config, *, ids, data_loader: DataLoader) \
        -> BatchIterFactory:
    assert ('batch_iter_fin' in config) ^ ('batch_iter_inf' in config)

    n_iters_per_epoch = config.get('n_iters_per_epoch', None)

    data_loader = config_data_loader(config, data_loader)
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
