from typing import Callable

from .batch_iter import BatchIter
from .logging import Logger
from .lr_base import LearningRatePolicy


def train(do_train_step: Callable, batch_iter: BatchIter, n_epochs: int, lr_policy: LearningRatePolicy, logger: Logger,
          validate: Callable = None):
    """
    Train a given model.

    Parameters
    ----------
    do_train_step: Callable
    batch_iter: BatchIter
        batch iterator
    n_epochs: int
        number of epochs to train
    lr_policy: LearningRatePolicy
        the learning rate policy
    logger: Logger
    validate: callable, optional
        a function that calculates the loss and metrics on the validation set
    """
    # TODO: stopping policy
    val_losses, metrics = None, None
    with batch_iter:
        for epoch in range(n_epochs):
            # train the model
            train_losses = []
            for inputs in batch_iter:
                train_losses.append(do_train_step(*inputs, lr=lr_policy.lr))
                lr_policy.step_finished(train_losses[-1])
            lr_policy.epoch_finished(train_losses=train_losses, val_losses=val_losses, metrics=metrics)

            logger.train(train_losses, epoch)
            logger.lr(lr_policy.lr, epoch)

            if validate is not None:
                val_result = validate()
                if val_result is not None:
                    val_losses, metrics = val_result
                    logger.validation(val_losses, epoch)
                    logger.metrics(metrics, epoch)
