import tensorboard_easy
import numpy as np


def log_vector(logger: tensorboard_easy.Logger, tag: str, vector, step: int):
    """Adds a vector values to log."""
    for i, value in enumerate(vector):
        logger.log_scalar(tag=tag + f'/{i}', value=value, step=step)


def log_scalar_or_vector(logger, tag, value: np.ndarray, step):
    value = value.squeeze()
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
        log_scalar_or_vector(self.logger, 'val/loss', np.mean(val_losses, axis=0), step)

    def metrics(self, metrics, step):
        for name, value in metrics.items():
            log_scalar_or_vector(self.logger, f'val/metrics/{name}', value, step)
