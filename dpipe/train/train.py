import numpy as np
from tensorboard_easy import Logger

from dpipe.model import Model
from dpipe.train.logging import log_vector
from dpipe.train.lr_base import LearningRate


def train_base(model: Model, batch_iterator: callable, n_epochs: int, lr_policy: LearningRate,
               log_path: str, validator: callable = None):
    """
    Train a given model.

    Parameters
    ----------
    model: Model
        the model to train
    batch_iterator: callable
        callable that returns a batch_iterator
    n_epochs: int
        number of epochs to train
    lr_policy: LearningRate
        the learning rate policy
    log_path: str
        the path where the logs will be stored
    validator: callable, optional
        a validator that calculates the loss and metrics on the validation set
    """
    # TODO: stopping policy
    val_losses, metrics = [], {}
    with Logger(log_path) as logger:
        train_log_write = logger.make_log_scalar('train/loss')
        lr_log_write = logger.make_log_scalar('train/lr')

        for epoch in range(n_epochs):
            lr_policy.next_epoch()

            # train the model
            train_losses = []
            for inputs in batch_iterator():
                lr_policy.next_step()
                # get the new learning rate
                lr = lr_policy.next_lr(train_losses=train_losses, val_losses=val_losses, metrics=metrics)
                lr_log_write(lr)

                train_losses.append(model.do_train_step(*inputs, lr=lr))
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
