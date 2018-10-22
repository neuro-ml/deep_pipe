from typing import Sequence, Callable, Dict

import numpy as np

from dpipe.dataset.base import AbstractAttribute


class Policy:
    """Interface for various policies."""
    value = AbstractAttribute('The current value')

    def __init__(self, initial):
        self.epoch = 0
        self.step = 0
        self.total_steps = 0
        self.value = initial

    def on_epoch_finished(self, *, train_losses: Sequence[float] = None, metrics: dict = None):
        """
        Update the `value` after an epoch is finished.

        The history of ``train_losses`` and ``metrics`` from the entire epoch is provided as additional information.
        """

    def epoch_finished(self, *, train_losses: Sequence[float] = None, metrics: dict = None):
        """
        Update the `value` and the counters after an epoch is finished.

        Notes
        -----
        Normally you don't need to override this method. See `on_epoch_finished`.
        """
        self.on_epoch_finished(train_losses=train_losses, metrics=metrics)
        self.step = 0
        self.epoch += 1

    def on_step_finished(self, train_loss: float):
        """
        Update the `value` after a step is finished.

        The current ``train_loss`` is provided as additional information.
        """

    def step_finished(self, train_loss: float):
        """
        Update the `value` and the counters after a step is finished.

        Notes
        -----
        Normally you don't need to override this method. See `on_step_finished`.
        """
        self.on_step_finished(train_loss)
        self.step += 1
        self.total_steps += 1

    @property
    @np.deprecate
    def lr(self):
        return self.value


# useful alias
Constant = Policy


class DecreasingOnPlateau(Policy):
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

    def on_epoch_finished(self, *, train_losses: Sequence[float], **kwargs):
        loss = np.mean(train_losses)
        if loss < self.margin_loss:
            self.margin_loss = self.get_margin_loss(loss)
            self.epochs_waited = 0
        else:
            self.epochs_waited += 1

            if self.epochs_waited > self.patience:
                self.value *= self.lr_dec_mul
                self.epochs_waited = 0


class Exponential(Policy):
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

    def on_epoch_finished(self, **kwargs):
        if self.floordiv:
            power = self.epoch // self.step_length
        else:
            power = self.epoch / self.step_length
        self.value = self.initial * self.multiplier ** power


class Schedule(Policy):
    """Multiply `value` by multipliers given by ``epoch2value_multiplier`` at corresponding epochs."""

    def __init__(self, initial: float, epoch2value_multiplier: Dict[int, float]):
        super().__init__(initial)
        self.epoch2value_multiplier = epoch2value_multiplier

    def on_epoch_finished(self, **kwargs):
        if self.epoch in self.epoch2value_multiplier:
            self.value *= self.epoch2value_multiplier[self.epoch]

    # ----------------------------------
    # Factories to build Schedule object
    @staticmethod
    def constant_multiplier(initial: float, multiplier: float, epochs: Sequence[int]):
        return Schedule(initial=initial, epoch2value_multiplier=dict(zip(epochs, [multiplier] * len(epochs))))


class LambdaEpoch(Policy):
    """Use the passed function to calculate the `value` for the current epoch (starting with 0)."""

    def __init__(self, func: Callable):
        super().__init__(func(0))
        self.func = func

    def on_epoch_finished(self, **kwargs):
        self.value = self.func(self.epoch + 1)
