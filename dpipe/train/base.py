from typing import Callable, Iterable

from dpipe.batch_iter import BatchIter
from dpipe.train.checkpoint import CheckpointManager
from .policy import Policy, ValuePolicy, EarlyStopping
from .logging import Logger


def one_epoch(epoch: int, do_train_step: Callable, batch_iter: Iterable, logger: Logger, validate: Callable = None,
              **policies: Policy):
    metrics = None
    train_losses = []
    for inputs in batch_iter:
        policy_values = {name: policy.value for name, policy in policies.items()
                         if issubclass(type(policy), ValuePolicy)}
        train_losses.append(do_train_step(*inputs, **policy_values))

        for policy in policies.values():
            policy.step_finished(train_losses[-1])

    logger.train(train_losses, epoch)
    # TODO: generalize
    if 'lr' in policies:
        logger.lr(policies['lr'].value, epoch)

    if validate is not None:
        metrics = validate()
        logger.metrics(metrics, epoch)

    for policy in policies.values():
        policy.epoch_finished(train_losses=train_losses, metrics=metrics)


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
        a collection of policies to run at each step.
        Policies, inherited from ValuePolicy will be passed to ``do_train_step``.
        The rest can be used to do early stopping with corresponding error.
    logger
    validate
        a function that calculates the loss and metrics on the validation set
    """

    with batch_iter:
        try:
            for epoch in range(n_epochs):
                one_epoch(epoch, do_train_step, batch_iter, logger, validate, **policies)
        except EarlyStopping:
            print('Early stopping!')


def train_with_checkpoints(do_train_step: Callable, batch_iter: BatchIter, n_epochs: int, logger: Logger,
                           checkpoint_manager: CheckpointManager, validate: Callable = None, **policies: Policy):
    checkpoint_manager.restore()
    current_epoch = checkpoint_manager.iteration

    with batch_iter:
        for epoch in range(current_epoch, n_epochs):
            one_epoch(epoch, do_train_step, batch_iter, logger, validate, **policies)
            checkpoint_manager.save()
