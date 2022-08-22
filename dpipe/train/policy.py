from datetime import datetime, timedelta
from typing import Sequence, Callable, Dict, Any, List, Union

import numpy as np
from tqdm import tqdm

from dpipe.dataset.base import AbstractAttribute, ABCAttributesMeta


class Policy:
    """Interface for various policies."""

    def epoch_started(self, epoch: int):
        """
        Update the policy before an epoch will start. The epochs numeration starts at zero.
        """

    def train_step_started(self, epoch: int, iteration: int):
        """
        Update the policy before a new train step.
        ``iteration`` denotes the iteration index inside the current epoch.
        The epochs and iterations numeration starts at zero.
        """

    def train_step_finished(self, epoch: int, iteration: int, loss: Any):
        """
        Update the policy after a train step.
        ``iteration`` denotes the iteration index inside the current epoch.
        ``loss`` is the value returned by the last train step.
        The epochs and iterations numeration starts at zero.
        """

    def validation_started(self, epoch: int, train_losses: Sequence):
        """
        Update the policy after the batch iterator was depleted. The epochs numeration starts at zero.

        The history of ``train_losses`` and ``metrics`` from the entire ``epoch`` is provided as additional information.
        """

    def epoch_finished(self, epoch: int, train_losses: Sequence, metrics: dict = None, policies: dict = None):
        """
        Update the policy after an epoch is finished. The epochs numeration starts at zero.

        The history of ``train_losses`` and ``metrics`` and ``policies`` from the entire ``epoch`` is provided as additional information.
        """


class ValuePolicy(Policy, metaclass=ABCAttributesMeta):
    """
    Interface for policies that have a `value` which changes over time.

    Attributes
    ----------
    value: the current value carried by the policy.
    """
    value = AbstractAttribute('The current value')

    def __init__(self, initial):
        super().__init__()
        # TODO: turn self.value into a property
        self.value = initial


# useful alias
Constant = ValuePolicy


class DecreasingOnPlateau(ValuePolicy):
    """
    Policy that traces average train loss and if it didn't decrease according to ``atol``
    or ``rtol`` for ``patience`` epochs, multiply `value` by ``multiplier``.
    ``atol`` :- absolute tolerance for detecting change in training loss value.
    ``rtol`` :- relative tolerance for detecting change in training loss value.
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
        return max(loss - self.atol, loss * (1 - self.rtol))

    def epoch_finished(self, epoch: int, train_losses: Sequence, **kwargs) -> None:
        loss = np.mean(train_losses)
        if loss < self.margin_loss:
            self.margin_loss = self.get_margin_loss(loss)
            self.epochs_waited = 0
            return
        self.epochs_waited += 1
        if self.epochs_waited > self.patience:
            self.value *= self.lr_dec_mul
            self.epochs_waited = 0


class Exponential(ValuePolicy):
    """
    Exponentially change the `value` by a factor of ``multiplier`` each ``step_length`` epochs.
    If ``floordiv`` is False - the `value` will be changed continuously.
    """

    def __init__(self, initial: float, multiplier: float, step_length: int = 1, floordiv: bool = True,
                 min_value: float = -np.inf, max_value: float = np.inf):
        super().__init__(initial)
        self.multiplier = multiplier
        self.initial = initial
        self.step_length = step_length
        self.floordiv = floordiv
        self.min_value = min_value
        self.max_value = max_value

    def epoch_started(self, epoch: int):
        power = epoch / self.step_length
        if self.floordiv:
            power = np.floor(power)
        self.value = np.clip(self.initial * self.multiplier ** power, self.min_value, self.max_value)


class Schedule(ValuePolicy):
    """Multiply `value` by multipliers given by ``epoch2value_multiplier`` at corresponding epochs."""

    def __init__(self, initial: float, epoch2value_multiplier: Dict[int, float]):
        super().__init__(initial)
        self.initial = initial
        self.epoch2value_multiplier = sorted(epoch2value_multiplier.items())

    def epoch_started(self, epoch: int):
        value = self.initial
        for idx, multiplier in self.epoch2value_multiplier:
            if epoch < idx:
                break
            value *= multiplier

        self.value = value

    # ----------------------------------
    # Factories to build Schedule object
    @staticmethod
    def constant_multiplier(initial: float, multiplier: float, epochs: Sequence[int]):
        return Schedule(initial=initial, epoch2value_multiplier=dict(zip(epochs, [multiplier] * len(epochs))))


class Switch(ValuePolicy):
    """
    Changes the `value` at specific epochs to the values given in `epoch_to_value`.
    """

    def __init__(self, initial: float, epoch_to_value: Dict[int, Any]):
        super().__init__(initial)
        self.epoch_to_value = sorted(epoch_to_value.items())

    def epoch_started(self, epoch: int):
        for idx, value in self.epoch_to_value:
            if epoch < idx:
                break

            self.value = value


class LambdaEpoch(ValuePolicy):
    """Use the passed function to calculate the `value` for the current epoch (starting with 0)."""

    def __init__(self, func: Callable, *args, **kwargs):
        super().__init__(func(0, *args, **kwargs))
        self.func = func
        self.args = args
        self.kwargs = kwargs

    def epoch_started(self, epoch: int):
        # func(0) was already calculated
        if epoch > 0:
            self.value = self.func(epoch, *self.args, **self.kwargs)


class EarlyStopping(StopIteration):
    """Exception raised by policies in order to trigger early stopping."""


class LossStop(Policy):
    def __init__(self, max_ratio: float = 3):
        super().__init__()
        self.min_loss = np.inf
        self.max_ratio = max_ratio

    def epoch_finished(self, epoch, *, train_losses: Sequence[float] = None, metrics: dict = None, **kwargs):
        loss = np.mean(train_losses)
        self.min_loss = min(self.min_loss, loss)
        if loss > self.max_ratio * self.min_loss:
            raise EarlyStopping


class TimeProfiler(Policy):
    def __init__(self, output=None):
        self.output = output
        self.stamps: List[datetime] = []

    def _gather_stats(self):
        def delta(stop, start):
            return (stop - start).total_seconds()

        def sum_deltas(seq):
            assert len(seq) % 2 == 0
            return sum(map(delta, seq[1::2], seq[::2]))

        now = datetime.now()
        stamps = self.stamps
        assert len(stamps) % 2 == 1
        n_batches = max(len(stamps) // 2, 1)

        durations = [
            ('Epoch', delta(now, stamps[0])),
            ('Train', delta(stamps[-1], stamps[0])),
            ('Validation', delta(now, stamps[-1])),

            ('Total Batch Iter', sum_deltas(self.stamps[:-1])),
            ('Total Train Step', sum_deltas(self.stamps[1:])),

            ('Avg Batch Iter', sum_deltas(self.stamps[:-1]) / n_batches),
            ('Avg Train Step', sum_deltas(self.stamps[1:]) / n_batches),
        ]

        return [(name, timedelta(seconds=seconds)) for name, seconds in durations]

    def _display(self, epoch):
        if self.output is None:
            self._print(epoch)
        else:
            self._tensorboard(epoch)

    def _tensorboard(self, epoch):
        from tensorboard_easy import Logger
        assert isinstance(self.output, Logger)

        for name, duration in self._gather_stats():
            name = name.lower().replace(' ', '_')
            self.output.log_scalar(f'time/{name}', duration.total_seconds(), epoch)

    def _print(self, epoch):
        print(f'Epoch {epoch} time profiling:', flush=True)
        for name, duration in self._gather_stats():
            print('  ', f'{name}:', duration, flush=True)

        print(flush=True)

    def epoch_started(self, epoch: int):
        self.stamps = [datetime.now()]

    def train_step_started(self, epoch: int, iteration: int):
        self.stamps.append(datetime.now())

    def train_step_finished(self, epoch: int, iteration: int, loss: Any):
        self.stamps.append(datetime.now())

    def epoch_finished(self, epoch: int, train_losses: Sequence, metrics: dict = None, **kwargs):
        self._display(epoch)

    # this policy is stateless
    def __getstate__(self):
        return {}


class TQDM(Policy):
    """
    Adds a tqdm progressbar.
    If loss is True - the progressbar will also display the current train loss.
    """

    def __init__(self, loss: bool = True):
        self.loss = loss
        self.bar: tqdm = tqdm(disable=True)

    def epoch_started(self, epoch: int):
        self.bar = tqdm(desc=f'Epoch {epoch}')

    def train_step_finished(self, epoch: int, iteration: int, loss: Any):
        if self.loss:
            self.bar.set_description(f'Epoch {epoch}. Train loss: {loss}')
        self.bar.update()

    def validation_started(self, epoch: int, train_losses: Sequence):
        self.bar.close()
        self.bar = None


class LoggerPolicy(Policy):
    def __init__(self, logger):
        self.logger = logger

    def train_step_finished(self, epoch: int, iteration: int, loss: Union[dict, float]):
        if not isinstance(loss, dict):
            loss = {'train/loss_step': loss}
        else:
            loss = {f'train/loss_step/{name}': v for name, v in loss.items()}

        for name, value in loss.items():
            self.logger.value(name, value)

    def epoch_finished(self, epoch: int, train_losses: Sequence, metrics: dict = None, policies: dict = None):
        self.logger.train(train_losses, epoch)
        if metrics is not None:
            self.logger.metrics(metrics, epoch)
        if policies is not None:
            self.logger.policies(policies, epoch)
