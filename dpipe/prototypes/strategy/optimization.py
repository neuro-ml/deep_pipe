import torch
from torch.optim import Optimizer
from torch.cuda.amp import GradScaler
from abc import ABCMeta, abstractmethod
from typing import Sequence, Any, Generator, Union, Dict

from dpipe.torch import to_np
from dpipe.torch.utils import set_params
from .policy import PolicyHandler, Policy


class OptimizationPolicy(Policy, metaclass=ABCMeta):
    def __init__(self, optimizer: Optimizer, *, optimizer_parameters: Union[Dict, PolicyHandler],
                 set_to_none=False, scaler: GradScaler = None):
        self.scaler = scaler
        self.optimizer = optimizer
        self.set_to_none = set_to_none

        if isinstance(optimizer_parameters, PolicyHandler):
            self.optimizer_parameters = optimizer_parameters
        else:
            self.optimizer_parameters = PolicyHandler(optimizer_parameters)

    @abstractmethod
    def optimize(self, losses_gen: Generator):
        pass

    @property
    def policies(self):
        return self.optimizer_parameters.policies

    def epoch_started(self, epoch: int):
        self.optimizer_parameters.epoch_started(epoch)
        set_params(self.optimizer, **self.optimizer_parameters.current_values)

    def epoch_finished(self, epoch: int, train_losses: Sequence, metrics: dict = None, policies: dict = None):
        self.optimizer_parameters.epoch_finished(epoch, train_losses, metrics)

    def train_step_started(self, epoch: int, iteration: int):
        self.optimizer_parameters.train_step_started(epoch, iteration)

    def train_step_finished(self, epoch: int, iteration: int, loss: Any):
        self.optimizer_parameters.train_step_finished(epoch, iteration, loss)

    def validation_started(self, epoch: int, train_losses: Sequence):
        self.optimizer_parameters.validation_started(epoch, train_losses)


class GradientsAccumulator(OptimizationPolicy):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def optimize(self, losses_gen: Generator):
        assert isinstance(losses_gen, Generator)
        self.optimizer.zero_grad(set_to_none=self.set_to_none)

        total_loss = 0.
        if self.scaler is not None:
            with torch.cuda.amp.autocast(False):
                for loss in losses_gen:
                    self.scaler.scale(loss).backward()
                    total_loss += loss

                self.scaler.step(self.optimizer)
                self.scaler.update()
        else:
            for loss in losses_gen:
                loss.backward()
                total_loss += loss
            self.optimizer.step()

        return to_np(total_loss)


class LossAccumulator(OptimizationPolicy):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def optimize(self, losses_gen: Generator):
        assert isinstance(losses_gen, Generator)
        self.optimizer.zero_grad(set_to_none=self.set_to_none)

        total_loss = sum(losses_gen)
        if self.scaler is not None:
            with torch.cuda.amp.autocast(False):
                self.scaler.scale(total_loss).backward()
                self.scaler.step(self.optimizer)
                self.scaler.update()
        else:
            total_loss.backward()
            self.optimizer.step()

        return to_np(total_loss)
