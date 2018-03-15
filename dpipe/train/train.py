from typing import Callable

import numpy as np
from tensorboard_easy import Logger

from dpipe.model import Model
from dpipe.train.logging import log_vector
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
        train_log_write = logger.make_log_scalar('train/loss')
        lr_log_write = logger.make_log_scalar('train/lr')

        for epoch in range(n_epochs):
            # train the model
            train_losses = []
            for inputs in batch_iter:
                train_losses.append(model.do_train_step(*inputs, lr=lr_policy.lr))

                if log_iteration:
                    train_log_write(train_losses[-1])
                    lr_log_write(lr_policy.lr)

                lr_policy.step_finished(train_losses[-1])

            logger.log_scalar('epoch/train_loss', np.mean(train_losses), epoch)

            if validate is not None:
                val_losses, metrics = validate()
                for name, value in metrics.items():
                    # check if not scalar
                    try:
                        log_vector(logger, f'metrics/{name}', value, epoch)
                    except TypeError:
                        logger.log_scalar(f'metrics/{name}', value, epoch)

                logger.log_scalar('epoch/val_loss', np.mean(val_losses), epoch)

            logger.log_scalar('epoch/lr', lr_policy.lr, epoch)
            lr_policy.epoch_finished(train_losses=train_losses, val_losses=val_losses, metrics=metrics)
