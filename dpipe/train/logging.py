import os
from collections import defaultdict
from functools import partial
from pathlib import Path
from typing import Sequence, Union

import numpy as np

from dpipe.commands import load_from_folder
from dpipe.io import PathLike
from dpipe.im.utils import zip_equal

__all__ = 'Logger', 'ConsoleLogger', 'TBLogger', 'NamedTBLogger', 'WANDBLogger'


def log_vector(logger, tag: str, vector, step: int):
    for i, value in enumerate(vector):
        logger.log_scalar(tag=tag + f'/{i}', value=value, step=step)


def log_scalar_or_vector(logger, tag, value: np.ndarray, step):
    value = np.asarray(value).flatten()
    if value.size > 1:
        log_vector(logger, tag, value, step)
    else:
        logger.log_scalar(tag, value, step)


def make_log_vector(logger, tag: str, first_step: int = 0) -> callable:
    def log(tag, value, step):
        log_vector(logger, tag, value, step)

    return logger._make_log(tag, first_step, log)


def group_dicts(dicts):
    groups = defaultdict(list)
    for entry in dicts:
        for name, value in entry.items():
            groups[name].append(value)

    return dict(groups)


class Logger:
    """Interface for logging during training."""

    def _dict(self, prefix, d, step):
        for name, value in d.items():
            self.value(f'{prefix}{name}', value, step)

    def train(self, train_losses: Sequence, step: int):
        """Log the ``train_losses`` at current ``step``."""
        raise NotImplementedError

    def value(self, name: str, value, step: int):
        """Log a single ``value``."""
        raise NotImplementedError

    def policies(self, policies: dict, step: int):
        """Log values coming from `ValuePolicy` objects."""
        self._dict('policies/', policies, step)

    def metrics(self, metrics: dict, step: int):
        """Log the metrics returned by the validation function during training."""
        self._dict('val/metrics/', metrics, step)


class ConsoleLogger(Logger):
    """A logger that writes to to stdout."""

    def value(self, name, value, step):
        print(f'{step:>05}: {name}: {value}', flush=True)

    def train(self, train_losses: Sequence[Union[dict, tuple, float]], step):
        text = ''
        if train_losses and isinstance(train_losses[0], dict):
            for name, values in group_dicts(train_losses).items():
                text += f'{name}: {np.mean(values)} '

        else:
            text += str(np.mean(train_losses, axis=0))

        self.value('Train loss', text, step)

    def policies(self, policies: dict, step: int):
        self._dict('Policies: ', policies, step)

    def metrics(self, metrics: dict, step: int):
        self._dict('Metrics: ', metrics, step)


class TBLogger(Logger):
    """A logger that writes to a tensorboard log file located at ``log_path``."""

    def __init__(self, log_path: PathLike):
        import tensorboard_easy
        self.logger = tensorboard_easy.Logger(log_path)

    def train(self, train_losses: Sequence[Union[dict, tuple, float]], step):
        if train_losses and isinstance(train_losses[0], dict):
            for name, values in group_dicts(train_losses).items():
                self.value(f'train/loss/{name}', np.mean(values), step)

        else:
            log_scalar_or_vector(self.logger, 'train/loss', np.mean(train_losses, axis=0), step)

    def value(self, name, value, step):
        dirname, base = os.path.split(name)

        count = base.count('__')
        if count > 1:
            raise ValueError(f'The tag name must contain at most one magic delimiter (__): {base}.')
        if count:
            base, kind = base.split('__')
        else:
            kind = 'scalar'

        name = os.path.join(dirname, base)

        if kind == 'vector':
            log = partial(log_vector, self.logger)
        else:
            log = getattr(self.logger, f'log_{kind}')
        log(name, np.asarray(value), step)

    def __getattr__(self, item):
        return getattr(self.logger, item)


class NamedTBLogger(TBLogger):
    """
    A logger that writes multiple train losses to a tensorboard log file located at ``log_path``.

    Each loss is assigned to a corresponding tag name from ``loss_names``.
    """

    def __init__(self, log_path: PathLike, loss_names: Sequence[str]):
        super().__init__(log_path)
        self.loss_names = loss_names

    def train(self, train_losses, step):
        values = np.mean(train_losses, axis=0)
        for name, value in zip_equal(self.loss_names, values):
            self.logger.log_scalar(f'train/loss/{name}', value, step)


class WANDBLogger(Logger):
    def __init__(self, project, run_name=None, *,
                 entity='neuro-ml', config=None, model=None, criterion=None, resume="auto"):
        """
        A logger that writes to a wandb run.

        Call wandb.login() before usage.
        """
        import wandb
        self._experiment = wandb.init(
            entity=entity,
            project=project,
            resume=resume
        )
        if run_name is not None:
            self._experiment.name = run_name  # can be changed manually

        if config is not None:
            self.config(config)

        if model is not None:
            self.watch(model, criterion)

    def value(self, name: str, value, step: int):
        self._experiment.log({name: value, 'step': step})

    def train(self, train_losses: Sequence[Union[dict, float]], step):
        if train_losses and isinstance(train_losses[0], dict):
            for name, values in group_dicts(train_losses).items():
                self.value(f'train/loss/{name}', np.mean(values), step)
        else:
            self.value('train/loss', np.mean(train_losses), step)

    def watch(self, model, criterion=None):
        self._experiment.watch(model, criterion=criterion)

    def config(self, config_args):
        self._experiment.config.update(config_args)

    def agg_metrics(self, agg_metrics: Union[dict, str, Path], section=''):
        """
        Log final metrics calculated in the end of experiment to summary table.
        Idea is to use these values for preparing leaderboard.

        agg_metrics: dictionary with name of metric as a key and with its value
        """
        if isinstance(agg_metrics, str) or isinstance(agg_metrics, Path):
            agg_metrics = {k if not section else f'{section}/{k}': v
                           for k, v in load_from_folder(agg_metrics, ext='.json')}
        elif section:
            agg_metrics = {f'{section}/{k}': v
                           for k, v in agg_metrics.items()}
        self._experiment.summary.update(agg_metrics)

    def ind_metrics(self, ind_metrics, step: int = 0, section: str = None):
        """
        Save individual metrics to a table to see bad cases

        ind_metrics: DataFrame
        step: int
        section: str, defines some metrics' grouping
        """
        from wandb import Table
        import pandas as pd
        if isinstance(ind_metrics, str) or isinstance(ind_metrics, Path):
            ind_metrics = pd.DataFrame.from_dict(
                {k: v for k, v in load_from_folder(ind_metrics, ext='.json')}).reset_index()
        table = Table(dataframe=ind_metrics)

        name = "Individual Metrics" if section is None else f"{section}/Individual Metrics"
        self._experiment.log({name: table, 'step': step})

    def image(self, name: str, *values, step: int, section: str = None,
              masks_keys: tuple = ('predictions', 'ground_truth')):
        """
        Method that logs images (set by values),
        each value is a dict with fields,preds,target and optinally caption defined
        Special policy that works as callback
        """
        from wandb import Image

        name = name if section is None else f"{section}/{name}"
        self._experiment.log(
            {
                name: [Image(
                    value['image'],
                    masks={k: {'mask_data': value[k]} for k in masks_keys},
                    caption=value.get('caption', None)
                ) for value in values],
                'step': step
            })

    def log_info(self, name: str, wandb_converter, *infos, section: str = None):
        name = name if section is None else f"{section}/{name}"
        self._experiment.log({name: [wandb_converter(info) for info in infos]})
