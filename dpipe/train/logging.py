from typing import List

import tensorboard_easy
import numpy as np

from dpipe.medim.utils import zip_equal


def log_vector(logger: tensorboard_easy.Logger, tag: str, vector, step: int):
    """Adds a vector values to log."""
    for i, value in enumerate(vector):
        logger.log_scalar(tag=tag + f'/{i}', value=value, step=step)


def log_scalar_or_vector(logger, tag, value: np.ndarray, step):
    value = np.asarray(value).squeeze()
    if value.size > 1:
        log_vector(logger, tag, value, step)
    else:
        logger.log_scalar(tag, value, step)


def make_log_vector(logger: tensorboard_easy.Logger, tag: str, first_step: int = 0) -> callable:
    def log(tag, value, step):
        log_vector(logger, tag, value, step)

    return logger._make_log(tag, first_step, log)


class Logger:
    def train(self, train_losses, step):
        pass

    def validation(self, val_losses, step):
        pass

    def lr(self, lr, step):
        pass

    def metrics(self, metrics, step):
        pass


class TBLogger(Logger):
    def __init__(self, log_path):
        self.logger = tensorboard_easy.Logger(log_path)

    def train(self, train_losses, step):
        log_scalar_or_vector(self.logger, 'train/loss', np.mean(train_losses, axis=0), step)

    def lr(self, lr, step):
        log_scalar_or_vector(self.logger, 'train/lr', lr, step)

    def validation(self, val_losses, step):
        if val_losses:
            log_scalar_or_vector(self.logger, 'val/loss', np.mean(val_losses, axis=0), step)

    def metrics(self, metrics, step):
        for name, value in metrics.items():
            log_scalar_or_vector(self.logger, f'val/metrics/{name}', value, step)

    def __getattr__(self, item):
        return getattr(self.logger, item)


class NamedTBLogger(TBLogger):
    def __init__(self, log_path, loss_names: List[str]):
        super().__init__(log_path)
        self.task_names = loss_names

    def train(self, train_losses, step):
        values = np.mean(train_losses, axis=0)
        for name, value in zip_equal(self.task_names, values):
            self.logger.log_scalar(f'train/loss/{name}', value, step)

    def validation(self, val_losses, step):
        if val_losses:
            values = np.mean(val_losses, axis=0)
            for name, value in zip_equal(self.task_names, values):
                self.logger.log_scalar(f'val/loss/{name}', value, step)
