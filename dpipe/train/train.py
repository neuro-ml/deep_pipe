from typing import Callable

import numpy as np
from tensorboard_easy import Logger

from dpipe.model import Model
from dpipe.train.logging import log_scalar_or_vector
from dpipe.train.lr_base import LearningRatePolicy
from .batch_iter import BatchIter


def train_base(model: Model, batch_iter: BatchIter, n_epochs: int, lr_policy: LearningRatePolicy, log_path: str,
               validate: Callable = None, log_iteration=False):
    """
    Train a given model.

    Parameters
    ----------
    model: Model
        the model to train
    batch_iter: BatchIter
        batch iterator
    n_epochs: int
        number of epochs to train
    lr_policy: LearningRate
        the learning rate policy
    log_path: str
        the path where the logs will be stored
    validate: callable, optional
        a function that calculates the loss and metrics on the validation set
    """
    # TODO: stopping policy
    val_losses, metrics = None, None
    with Logger(log_path) as logger, batch_iter:
        train_log_write = logger.make_log_scalar('train/batch/loss')
        lr_log_write = logger.make_log_scalar('train/batch/lr')

        for epoch in range(n_epochs):
            # train the model
            train_losses = []
            for inputs in batch_iter:
                train_losses.append(model.do_train_step(*inputs, lr=lr_policy.lr))

                if log_iteration:
                    train_log_write(train_losses[-1])
                    lr_log_write(lr_policy.lr)

                lr_policy.step_finished(train_losses[-1])

            log_scalar_or_vector(logger, 'train/loss', np.mean(train_losses, axis=0), epoch)

            if validate is not None:
                val_losses, metrics = validate()
                for name, value in metrics.items():
                    log_scalar_or_vector(logger, f'val/metrics/{name}', value, epoch)
                log_scalar_or_vector(logger, 'val/loss', np.mean(val_losses, axis=0), epoch)

            log_scalar_or_vector(logger, 'train/lr', lr_policy.lr, epoch)
            lr_policy.epoch_finished(train_losses=train_losses, val_losses=val_losses, metrics=metrics)
