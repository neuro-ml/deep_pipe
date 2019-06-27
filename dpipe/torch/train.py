from typing import Callable, Union

from .model import TorchModel
from ..batch_iter.base import BatchIter
from ..train.checkpoint import CheckpointManager
from ..train.policy import Policy, ValuePolicy, NEpochs, Constant
from ..train.logging import TBLogger, Logger
from ..train import train


def train_model(model: TorchModel, batch_iter: BatchIter, n_epochs: int, lr: Union[float, ValuePolicy],
                log_path: str = None, checkpoints_path: str = None, validate: Callable = None, **policies: Policy):
    """
    Train a given ``model``.

    Parameters
    ----------
    model
    batch_iter
    n_epochs
    lr
        the learning rate.
    log_path
        path where the tensorboard logs will be written. If None - no logs will be written.
    checkpoints_path
        path where checkpoints will be saved. If None - no checkpoints will be saved.
    validate: Callable() -> metrics
        a function that calculates metrics on the validation set.
    policies
        other policies.
    """
    logger = checkpoint_manager = None
    if log_path is not None:
        logger = log_path
        if not isinstance(logger, Logger):
            logger = TBLogger(log_path)

    if not isinstance(lr, ValuePolicy):
        lr = Constant(lr)

    policies['lr'] = lr
    if checkpoints_path is not None:
        checkpoint_manager = CheckpointManager(checkpoints_path, policies, {
            'model': model.model_core,
            'optimizer': model.optimizer
        })

    return train(model.do_train_step, batch_iter, logger, checkpoint_manager, validate,
                 n_epochs=NEpochs(n_epochs), **policies)
