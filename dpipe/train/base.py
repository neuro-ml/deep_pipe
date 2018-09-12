from typing import Callable
from warnings import warn

from dpipe.batch_iter import BatchIter
from .policy import Policy
from .logging import Logger


def train(do_train_step: Callable, batch_iter: BatchIter, n_epochs: int, logger: Logger,
          validate: Callable = None, **policies: Policy):
    """
    Train a given model.

    Parameters
    ----------
    do_train_step
    batch_iter
        batch iterator
    n_epochs
        number of epochs to train
    policies:
        a collection of policies to be passed to ``do_train_step``
    logger
    validate
        a function that calculates the loss and metrics on the validation set
    """
    metrics = None
    with batch_iter:
        for epoch in range(n_epochs):
            # train the model
            train_losses = []
            for inputs in batch_iter:
                train_losses.append(do_train_step(
                    *inputs, **{name: policy.value for name, policy in policies.items()}))

                for policy in policies.values():
                    policy.step_finished(train_losses[-1])

            logger.train(train_losses, epoch)
            # TODO: generalize
            if 'lr' in policies:
                logger.lr(policies['lr'].value, epoch)

            if validate is not None:
                metrics = validate()
                if not isinstance(metrics, dict):
                    warn('Validation losses are deprecated. '
                         'If you need val losses just put them in the metrics dict.', DeprecationWarning)
                    metrics = metrics[1]

                logger.metrics(metrics, epoch)

            for policy in policies.values():
                policy.epoch_finished(train_losses=train_losses, metrics=metrics)
