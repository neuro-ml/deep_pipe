import functools
import math
from functools import partial

import numpy as np
from tensorboard_easy.logger import Logger

from dpipe.batch_iter_factory import BatchIterFactory
from dpipe.batch_predict import BatchPredict
from dpipe.config import register
from dpipe.model import Model
from dpipe.train.logging import log_vector
from .utils import make_find_next_lr, make_check_loss_decrease


def train(model: Model, train_batch_iter_factory: BatchIterFactory, validator, n_epochs: int,
          log_path: str, lr_init: float, lr_dec_mul: float, patience: int, rtol, atol,
          batch_predict: BatchPredict):
    """
    Train a model with a decreasing learning rate.

    Parameters
    ----------
    model: Model
        the model to train
    train_batch_iter_factory: BatchIterFactory
        a factory of train batch iterators
    validator: callable
        a validator that calculates the loss and metrics on the validation set
    n_epochs: int
        number of epochs to train
    log_path: str
    lr_init: float
        initial learning rate
    lr_dec_mul: float
        the learning rate decreasing multiplier
    """
    # TODO: lr policy, stopping policy
    logger = Logger(log_path)

    find_next_lr = make_find_next_lr(lr_init, lambda lr: lr * lr_dec_mul,
                                     partial(make_check_loss_decrease, patience=patience, rtol=rtol, atol=atol))

    train_log_write = logger.make_log_scalar('train/loss')
    lr_log_write = logger.make_log_scalar('train/lr')
    train_avg_log_write = logger.make_log_scalar('epoch/train_loss')
    val_avg_log_write = logger.make_log_scalar('epoch/val_loss')

    validate_fn = functools.partial(batch_predict.validate, validate_fn=model.do_val_step)
    lr = find_next_lr(math.inf)
    with train_batch_iter_factory, logger:
        for epoch in range(n_epochs):
            with next(train_batch_iter_factory) as train_batch_iter:
                train_losses = []
                for inputs in train_batch_iter:
                    train_losses.append(model.do_train_step(*inputs, lr=lr))
                    train_log_write(train_losses[-1])
                train_avg_log_write(np.mean(train_losses))

            val_losses, metrics_single, metrics_multiple = validator(validate=validate_fn)

            for name, values in metrics_single.items():
                logger.log_histogram(f'metrics/{name}', np.asarray(values), epoch)

            for name, value in metrics_multiple.items():
                try:
                    # check if not scalar
                    log_vector(logger, f'metrics/{name}', value, epoch)
                except TypeError:
                    logger.log_scalar(f'metrics/{name}', value, epoch)

            val_loss = np.mean(val_losses)
            val_avg_log_write(val_loss)
            lr = find_next_lr(val_loss)
            lr_log_write(lr)
