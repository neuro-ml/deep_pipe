from functools import partial

from dpipe.dl import optimize, Model, FrozenModel
from dpipe.dl.config import predictor_name2predictor, loss_name2loss
from dpipe.dl.model_cores.config import model_core_name2model_core

from dpipe.dl.trains.config import train_name2train
from .utils import config_partial, config_object

__all__ = ['config_model', 'config_frozen_model', 'config_train']

module_builders = {
    'model_core': model_core_name2model_core,
    'predict': predictor_name2predictor,
    'loss': loss_name2loss,
}


def config_train(config) -> callable:
    return config_partial(config, 'train', train_name2train)


def _config_optimizer(config) -> callable:
    return partial(optimize, tf_optimizer_name=config['optimizer'],
                   **config.get('optimizer__params', {}))

#
# def _config_metrics(config) -> callable:
#     metric_names = co0nfig.get('metrics', [])
#     metrics = {name: metric_name2metric[name] for name in metric_names}
#     return metrics


def config_model(config, dataset) -> Model:
    predict = config_partial('predict', config, module_builders)
    loss = config_partial('loss', config, module_builders)
    optimizer = _config_optimizer(config)

    model_core = config_object('model_core', config, module_builders,
                               n_chans_in=dataset.n_chans_mscan,
                               n_chans_out=dataset.n_chans_out)
    return Model(model_core, predict=predict, loss=loss, optimize=optimizer)

#
# def config_model_controller(config, model, log_path, restore_model_path):
#     metrics = _config_metrics(config)
#     return ModelController(model, log_path=log_path, metrics=metrics,
#                            restore_model_path=restore_model_path)


def config_frozen_model(config, dataset) -> FrozenModel:
    predict = config_partial('predict', config, module_builders)
    model_core = config_object('model_core', config, module_builders,
                               n_chans_in=dataset.n_chans_mscan,
                               n_chans_out=dataset.n_chans_out)
    return FrozenModel(model_core, predict=predict)
