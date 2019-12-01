import contextlib
from typing import Callable

import numpy as np

from .checkpoint import CheckpointManager
from .policy import Policy, ValuePolicy, EarlyStopping
from .logging import Logger


def one_epoch(epoch: int, train_step: Callable, batch_iter: Callable, logger: Logger, validate: Callable, **kwargs):
    def get_values():
        return {name: policy_.value for name, policy_ in policies.items() if isinstance(policy_, ValuePolicy)}

    policies = {name: value for name, value in kwargs.items() if isinstance(value, Policy)}
    values = {name: value for name, value in kwargs.items() if not isinstance(value, Policy)}

    for policy in policies.values():
        policy.epoch_started(epoch)

    metrics = None
    train_losses = []
    for inputs in batch_iter():
        train_losses.append(train_step(*inputs, **values, **get_values()))

    logger.train(train_losses, epoch)
    logger.policies(get_values(), epoch)

    if validate is not None:
        metrics = validate()
        logger.metrics(metrics, epoch)

    for policy in policies.values():
        policy.epoch_finished(epoch, train_losses=train_losses, metrics=metrics)


class _DummyCheckpointManager:
    def save(self, iteration: int):
        pass

    @staticmethod
    def restore():
        return 0


class _DummyLogger(Logger):
    def train(self, train_losses, step):
        pass

    def value(self, name, value, step):
        pass


@contextlib.contextmanager
def _build_context_manager(o):
    yield o


def train(train_step: Callable, batch_iter: Callable, n_epochs: int = np.inf, logger: Logger = None,
          checkpoint_manager: CheckpointManager = None, validate: Callable = None, **kwargs):
    """
    Performs a series of train and validation steps.

    Parameters
    ----------
    train_step: Callable
        a function to perform train step.
    batch_iter: Callable
        batch iterator.
    n_epochs: int
        maximal number of training epochs
    logger: Logger, None, optional
    checkpoint_manager: CheckpointManager, None, optional
    validate: Callable, None, optional
        a function to calculate metrics on the validation set.
    kwargs
        additional keyword arguments passed to ``train_step``.
        For instances of `ValuePolicy` their `value` attribute is passed.
        Other policies are used for early stopping.

    References
    ----------
    See the :doc:`tutorials/training` tutorial for more details.
    """
    if checkpoint_manager is None:
        checkpoint_manager = _DummyCheckpointManager()
    if logger is None:
        logger = _DummyLogger()
    if not hasattr(batch_iter, '__enter__'):
        batch_iter = _build_context_manager(batch_iter)

    epoch = checkpoint_manager.restore()

    with batch_iter as iterator:
        try:
            while epoch < n_epochs:
                one_epoch(epoch, train_step, iterator, logger, validate, **kwargs)
                checkpoint_manager.save(epoch)
                epoch += 1

        except EarlyStopping:
            pass
