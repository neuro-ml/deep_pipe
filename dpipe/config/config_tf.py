from functools import partial

from .utils import config_partial, config_object
from dpipe.modules.dl import optimize, Model, FrozenModel
from dpipe.modules.dl.config import predictor_name2predictor, loss_name2loss
from dpipe.modules.dl.model_cores.config import model_core_name2model_core
from dpipe.modules.dl.trains.config import train_name2train

__all__ = ['config_model', 'config_frozen_model', 'config_train']

module_builders = {
    'model_core': model_core_name2model_core,
    'train': train_name2train,
    'predict': predictor_name2predictor,
    'loss': loss_name2loss
}


def config_train(config) -> callable:
    return config_partial('train', config, module_builders)


def _config_optimizer(config) -> callable:
    return partial(optimize, tf_optimizer_name=config['optimizer'],
                   **config.get('optimizer__params', {}))


def config_model(config, dataset) -> Model:
    predict = config_partial('predict', config, module_builders)
    loss = config_partial('loss', config, module_builders)
    optimizer = _config_optimizer(config)

    model_core = config_object('model_core', config, module_builders,
                               n_chans_in=dataset.n_chans_mscan,
                               n_chans_out=dataset.n_chans_out)
    return Model(model_core, predict=predict, loss=loss, optimize=optimizer)


def config_frozen_model(config, dataset) -> FrozenModel:
    predict = config_partial('predict', config, module_builders)
    model_core = config_object('model_core', config, module_builders,
                               n_chans_in=dataset.n_chans_mscan,
                               n_chans_out=dataset.n_chans_out)
    return FrozenModel(model_core, predict=predict)
