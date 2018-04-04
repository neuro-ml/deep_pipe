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
    def train(self, train_losses):
        pass

    def validation(self, val_losses):
        pass

    def lr(self, lr):
        pass

    def metrics(self, metrics, epoch):
        pass


class TBLogger(Logger):
    def __init__(self, log_path):
        logger = tensorboard_easy.Logger(log_path)
        self.logger = logger
        self.train_log = logger.make_log_scalar('train/loss')
        self.lr_log = logger.make_log_scalar('train/lr')
        self.val_log = logger.make_log_scalar('val/loss')

    def train(self, train_losses):
        self.train_log(np.mean(train_losses, axis=0))

    def validation(self, val_losses):
        self.train_log(np.mean(val_losses, axis=0))

    def lr(self, lr):
        self.lr_log(lr)

    def metrics(self, metrics, epoch):
        for name, value in metrics.items():
            # check if not scalar
            try:
                log_vector(self.logger, f'metrics/{name}', value, epoch)
            except TypeError:
                self.logger.log_scalar(f'metrics/{name}', value, epoch)
