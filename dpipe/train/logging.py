from typing import Sequence

import tensorboard_easy
import numpy as np

from dpipe.medim.io import PathLike
from dpipe.medim.utils import zip_equal


def log_vector(logger: tensorboard_easy.Logger, tag: str, vector, step: int):
    for i, value in enumerate(vector):
        logger.log_scalar(tag=tag + f'/{i}', value=value, step=step)


def log_scalar_or_vector(logger, tag, value: np.ndarray, step):
    value = np.asarray(value).flatten()
    if value.size > 1:
        log_vector(logger, tag, value, step)
    else:
        logger.log_scalar(tag, value, step)


def make_log_vector(logger: tensorboard_easy.Logger, tag: str, first_step: int = 0) -> callable:
    def log(tag, value, step):
        log_vector(logger, tag, value, step)

    return logger._make_log(tag, first_step, log)


class Logger:
    def _dict(self, prefix, d, step):
        for name, value in d.items():
            self.value(f'{prefix}/{name}', value, step)

    def train(self, train_losses, step):
        raise NotImplementedError

    def value(self, name, value, step):
        raise NotImplementedError

    def policies(self, policies: dict, step: int):
        self._dict('policies', policies, step)

    def metrics(self, metrics, step):
        self._dict('val/metrics', metrics, step)


class ConsoleLogger(Logger):
    """Log ``train_losses`` and ``metrics`` to stdout."""

    def value(self, name, value, step):
        pass

    def train(self, train_losses, step):
        print(f'{step:>05}: train loss: {np.mean(train_losses)}', flush=True)

    def metrics(self, metrics, step):
        for name, value in metrics.items():
            print(f'{step:>05}: {name} = {value}')


class TBLogger(Logger):
    """A logger that writes to a tensorboard log file located at ``log_path``."""

    def __init__(self, log_path: PathLike):
        self.logger = tensorboard_easy.Logger(log_path)

    def train(self, train_losses, step):
        log_scalar_or_vector(self.logger, 'train/loss', np.mean(train_losses, axis=0), step)

    def value(self, name, value, step):
        log_scalar_or_vector(self.logger, name, value, step)

    def __getattr__(self, item):
        return getattr(self.logger, item)


class NamedTBLogger(TBLogger):
    """
    A logger that writes multiple train losses to a tensorboard log file located at ``log_path``.

    Each loss is assigned a corresponding tag name from ``loss_names``.
    """

    def __init__(self, log_path: PathLike, loss_names: Sequence[str]):
        super().__init__(log_path)
        self.loss_names = loss_names

    def train(self, train_losses, step):
        values = np.mean(train_losses, axis=0)
        for name, value in zip_equal(self.loss_names, values):
            self.logger.log_scalar(f'train/loss/{name}', value, step)
