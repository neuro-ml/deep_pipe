from typing import Sequence

import numpy as np

from .lr_base import LearningRatePolicy


class Decreasing(LearningRatePolicy):
    """
    Learning rate policy that traces average train loss or val loss (trace_train or trace_val parameter) and if
    it didn't decrease according to atol or rtol for patience epochs, multiply lr by lr_dec_mul.
    """

    def __init__(self, *, lr_init: float, lr_dec_mul: float, patience: int, trace_train=False,
                 trace_val=False, rtol, atol):
        super().__init__(lr_init)

        assert (trace_train ^ trace_val), 'either trace_train or trace_val should be activated'

        self.lr_dec_mul = lr_dec_mul
        self.patience = patience
        self.epochs_waited = 0
        self.trace_train = trace_train
        self.get_margin_loss = lambda loss: max([loss * (1 - rtol), loss - atol])
        self.margin_loss = np.inf

    def on_epoch_finished(self, *, train_losses: Sequence[float] = None, val_losses: Sequence[float] = None, **kwargs):
        loss = np.mean(train_losses if self.trace_train else val_losses)
        if loss < self.margin_loss:
            self.margin_loss = self.get_margin_loss(loss)
            self.epochs_waited = 0
        else:
            self.epochs_waited += 1

            if self.epochs_waited > self.patience:
                self.lr *= self.lr_dec_mul
                self.epochs_waited = 0


class Exponential(LearningRatePolicy):
    def __init__(self, initial, multiplier, step_length=1, floordiv=True):
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
        self.lr = self.initial * self.multiplier ** power


class Schedule(LearningRatePolicy):
    def __init__(self, lr_init, epoch2lr_dec_mul):
        super().__init__(lr_init)
        self.epoch2lr_dec_mul = epoch2lr_dec_mul

    def on_epoch_finished(self, *, train_losses: Sequence[float] = None, val_losses: Sequence[float] = None,
                          metrics: dict = None):
        if self.epoch in self.epoch2lr_dec_mul:
            self.lr *= self.epoch2lr_dec_mul[self.epoch]
