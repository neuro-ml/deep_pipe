from typing import Union

import numpy as np
from tensorboard_easy import Logger

from dpipe.batch_iter_factory import BatchIterFactory
from dpipe.model import Model
from dpipe.train.logging import log_vector


def train_base(model: Model, batch_iter_factory: BatchIterFactory, n_epochs: int, lr: Union[float, callable],
               log_path: str, validator: callable = None):
    """
    Train a model with a decreasing learning rate.

    Parameters
    ----------
    model: Model
        the model to train
    batch_iter_factory: BatchIterFactory
        a factory of train batch iterators
    n_epochs: int
        number of epochs to train
    lr: float, callable
        the learning rate. If callable it must have the following signature:
        (epoch, *, train_losses, val_losses, metrics) -> new_lr
    log_path: str
        the path where the logs will be stored
    validator: callable, optional
        a validator that calculates the loss and metrics on the validation set
    """
    # TODO: stopping policy
    train_losses = val_losses = metrics = None
    with batch_iter_factory, Logger(log_path) as logger:
        train_log_write = logger.make_log_scalar('train/loss')

        for epoch in range(n_epochs):
            # get the new learning rate:
            if callable(lr):
                new_lr = lr(epoch=epoch, train_losses=train_losses, val_losses=val_losses, metrics=metrics)
            else:
                new_lr = lr
            logger.log_scalar('train/lr', new_lr, epoch)

            # train the model
            with next(batch_iter_factory) as train_batch_iter:
                train_losses = []
                for inputs in train_batch_iter:
                    train_losses.append(model.do_train_step(*inputs, lr=new_lr))
                    train_log_write(train_losses[-1])
                logger.log_scalar('epoch/train_loss', np.mean(train_losses), epoch)

            if validator is not None:
                val_losses, metrics = validator()
                for name, value in metrics.items():
                    # check if not scalar
                    try:
                        log_vector(logger, f'metrics/{name}', value, epoch)
                    except TypeError:
                        logger.log_scalar(f'metrics/{name}', value, epoch)

                logger.log_scalar('epoch/val_loss', np.mean(val_losses), epoch)
