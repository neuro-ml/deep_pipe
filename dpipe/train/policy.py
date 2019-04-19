from typing import Sequence, Callable, Dict

import numpy as np

from dpipe.dataset.base import AbstractAttribute, ABCAttributesMeta


class Policy:
    """Interface for various policies."""

    def epoch_started(self, epoch: int):
        """
        Update the policy before an epoch will start. The epochs numeration starts at zero.
        """

    def epoch_finished(self, epoch: int, *, train_losses: Sequence = None, metrics: dict = None):
        """
        Update the policy after an epoch is finished. The epochs numeration starts at zero.

        The history of ``train_losses`` and ``metrics`` from the entire ``epoch`` is provided as additional information.
        """


class ValuePolicy(Policy, metaclass=ABCAttributesMeta):
    """Interface for policies that have a ``value`` which changes over time."""
    value = AbstractAttribute('The current value')

    def __init__(self, initial):
        super().__init__()
        self.value = initial


# useful alias
Constant = ValuePolicy


class DecreasingOnPlateau(ValuePolicy):
    """
    Policy that traces average train loss and if it didn't decrease according to ``atol``
    or ``rtol`` for ``patience`` epochs, multiply `value` by ``multiplier``.
    """

    def __init__(self, *, initial: float, multiplier: float, patience: int, rtol, atol):
        super().__init__(initial)

        self.atol = atol
        self.rtol = rtol
        self.lr_dec_mul = multiplier
        self.patience = patience
        self.epochs_waited = 0
        self.margin_loss = np.inf

    def get_margin_loss(self, loss):
        return max([loss * (1 - self.rtol), loss - self.atol])

    def epoch_finished(self, epoch: int, *, train_losses: Sequence, **kwargs):
        loss = np.mean(train_losses)
        if loss < self.margin_loss:
            self.margin_loss = self.get_margin_loss(loss)
            self.epochs_waited = 0
        else:
            self.epochs_waited += 1

            if self.epochs_waited > self.patience:
                self.value *= self.lr_dec_mul
                self.epochs_waited = 0


class Exponential(ValuePolicy):
    """
    Exponentially change the `value` by a factor of ``multiplier`` each ``step_length`` epochs.
    If ``floordiv`` is False - the `value` will be changed continuously.
    """

    def __init__(self, initial: float, multiplier: float, step_length: int = 1, floordiv: bool = True):
        super().__init__(initial)
        self.multiplier = multiplier
        self.initial = initial
        self.step_length = step_length
        self.floordiv = floordiv

    def epoch_finished(self, epoch, **kwargs):
        power = epoch / self.step_length
        if self.floordiv:
            power = np.floor(power)
        self.value = self.initial * self.multiplier ** power


class Schedule(ValuePolicy):
    """Multiply `value` by multipliers given by ``epoch2value_multiplier`` at corresponding epochs."""

    def __init__(self, initial: float, epoch2value_multiplier: Dict[int, float]):
        super().__init__(initial)
        self.epoch2value_multiplier = epoch2value_multiplier

    def epoch_finished(self, epoch, **kwargs):
        if epoch in self.epoch2value_multiplier:
            self.value *= self.epoch2value_multiplier[epoch]

    # ----------------------------------
    # Factories to build Schedule object
    @staticmethod
    def constant_multiplier(initial: float, multiplier: float, epochs: Sequence[int]):
        return Schedule(initial=initial, epoch2value_multiplier=dict(zip(epochs, [multiplier] * len(epochs))))


class LambdaEpoch(ValuePolicy):
    """Use the passed function to calculate the `value` for the current epoch (starting with 0)."""

    def __init__(self, func: Callable):
        super().__init__(func(0))
        self.func = func

    def epoch_finished(self, epoch, **kwargs):
        self.value = self.func(epoch + 1)


class EarlyStopping(StopIteration):
    """Exception raised by policies in order to trigger early stopping."""


class LossStop(Policy):
    def __init__(self, max_ratio: float = 3):
        super().__init__()
        self.min_loss = np.inf
        self.max_ratio = max_ratio

    def epoch_finished(self, epoch, *, train_losses: Sequence[float] = None, metrics: dict = None):
        loss = np.mean(train_losses)
        self.min_loss = min(self.min_loss, loss)
        if loss > self.max_ratio * self.min_loss:
            raise EarlyStopping


class NEpochs(Policy):
    def __init__(self, n_epochs):
        super().__init__()
        self.n_epochs = n_epochs

    def epoch_started(self, epoch: int):
        if epoch >= self.n_epochs:
            raise EarlyStopping
