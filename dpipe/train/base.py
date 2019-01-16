from typing import Callable, Iterable

from dpipe.batch_iter import BatchIter
from .checkpoint import CheckpointManager, DummyCheckpointManager
from .policy import Policy, ValuePolicy, EarlyStopping
from .logging import Logger, DummyLogger


def one_epoch(epoch: int, do_train_step: Callable, batch_iter: Iterable, logger: Logger, validate: Callable = None,
              **policies: Policy):
    def get_values():
        return {name: policy_.value for name, policy_ in policies.items() if isinstance(policy_, ValuePolicy)}

    for policy in policies.values():
        policy.epoch_started(epoch)

    metrics = None
    train_losses = []
    for inputs in batch_iter:
        train_losses.append(do_train_step(*inputs, **get_values()))

    logger.train(train_losses, epoch)
    logger.policies(get_values(), epoch)

    if validate is not None:
        metrics = validate()
        logger.metrics(metrics, epoch)

    for policy in policies.values():
        policy.epoch_finished(epoch, train_losses=train_losses, metrics=metrics)


def train(do_train_step: Callable, batch_iter: BatchIter, logger: Logger = None,
          checkpoint_manager: CheckpointManager = None, validate: Callable = None, **policies: Policy):
    """
    Train a given model.

    Parameters
    ----------
    do_train_step
    batch_iter
        batch iterator
    logger
    checkpoint_manager
    validate
        a function that calculates metrics on the validation set
    policies:
        a collection of policies to run before and after epoch.
        Policies, inherited from `ValuePolicy` will be passed to ``do_train_step``.
        The rest can be used for early stopping.
    """
    if checkpoint_manager is None:
        checkpoint_manager = DummyCheckpointManager()
    if logger is None:
        logger = DummyLogger()

    epoch = checkpoint_manager.restore()

    with batch_iter:
        try:
            while True:
                one_epoch(epoch, do_train_step, batch_iter, logger, validate, **policies)
                checkpoint_manager.save(epoch)
                epoch += 1

        except EarlyStopping:
            pass
