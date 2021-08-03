import torch
import numpy as np

from torch import nn as nn
from typing import Dict, Callable
from abc import ABCMeta, abstractmethod

from dpipe.train.logging import Logger
from dpipe.itertools import dmap, lmap
from dpipe.train.checkpoint import Checkpoints
from dpipe.batch_iter.pipeline import Infinite
from dpipe.torch import sequence_to_var, to_np
from dpipe.train.policy import EarlyStopping

from .policy import PolicyHandler
from .optimization import OptimizationPolicy


class BatchProcessor(metaclass=ABCMeta):
    """
    Base interface for all entities involved in training procedure.
    """

    @abstractmethod
    def start_epoch(self, epoch: int):
        pass

    @abstractmethod
    def process_batch(self, epoch: int, batch_index: int):
        pass

    @abstractmethod
    def finish_epoch(self, epoch: int):
        pass

    @abstractmethod
    def validate(self, epoch: int):
        pass


class Strategy(BatchProcessor, metaclass=ABCMeta):
    """
    Interface for objects that perform both forward and backward steps.
    """

    def __init__(self, batches_per_epoch):
        self.batches_per_epoch = batches_per_epoch

    @abstractmethod
    def __enter__(self):
        pass

    @abstractmethod
    def __exit__(self, exc_type, exc_val, exc_tb):
        pass

    @abstractmethod
    def close(self):
        pass

    @abstractmethod
    def process_batch(self, epoch: int, batch_index: int):
        pass


class ForwardStrategy(Strategy):
    """
    A class for performing a forward pass only.
    """

    def __init__(self, train_step_parameters: Dict, iterator: Infinite,
                 calculate_loss: Callable, logger: Logger, validate: Callable = None):
        self.logger = logger
        self.iterator = iterator
        self.validate_step = validate
        self.calculate_loss = calculate_loss
        self.train_parameters = PolicyHandler(train_step_parameters)

        self._generator = None
        self._epoch_losses = None
        super().__init__(self.iterator.batches_per_epoch)

    def __enter__(self):
        self.iterator.pipeline.__enter__()

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.iterator.pipeline.__exit__(exc_type, exc_val, exc_tb)

    def close(self):
        self.iterator.pipeline.close()

    def start_epoch(self, epoch):
        self.train_parameters.epoch_started(epoch)
        # initialization
        self._epoch_losses = []
        self._generator = self.iterator()

    def process_batch(self, epoch: int, batch_index: int):
        self.train_parameters.train_step_started(epoch, batch_index)
        # process inputs
        inputs = next(self._generator)
        loss, detached_loss = self.calculate_loss(*inputs, **self.train_parameters.current_values)
        self._epoch_losses.append(detached_loss)
        self.train_parameters.train_step_finished(epoch, batch_index, detached_loss)
        return loss

    def validate(self, epoch):
        if self.validate_step is not None:
            self.train_parameters.validation_started(epoch, self._epoch_losses)
            metrics = self.validate_step()
            self.logger.metrics(metrics, epoch)
            return metrics
        else:
            return {}

    def finish_epoch(self, epoch):
        self.logger.train(self._epoch_losses, epoch)
        self.logger.policies(self.train_parameters.policies, epoch)
        self._epoch_losses = None

    @property
    def is_active(self):
        return self._generator is not None


class CompositeTrainStrategy(Strategy):
    """
    A Class for aggregation result of multiple ForwardStrategy and for performing an optimization step.
    """

    def __init__(self, *strategies: ForwardStrategy, optimization_policy: OptimizationPolicy, logger: Logger = None):
        self.logger = logger
        self.strategies = list(strategies)
        self.optimization_policy = optimization_policy

        self._epoch_total_losses = None
        batches_per_epoch = set()

        for x in self.strategies:
            batches_per_epoch.add(x.iterator.batches_per_epoch)

        # make sure that all iterators have same batches_per_epoch
        assert len(batches_per_epoch) == 1, len(batches_per_epoch)
        # init parent
        super().__init__(batches_per_epoch.pop())

    def start_epoch(self, epoch):
        self._epoch_total_losses = []
        self.optimization_policy.epoch_started(epoch)

        for s in self.strategies:
            s.start_epoch(epoch)

    def process_batch(self, epoch: int, batch_index: int):
        self.optimization_policy.train_step_started(epoch, batch_index)
        # create generator for losses
        losses_gen = (s.process_batch(epoch, batch_index) for s in self.strategies)
        # perform optimization step
        total_loss = self.optimization_policy.optimize(losses_gen)
        self._epoch_total_losses.append(total_loss)
        self.optimization_policy.train_step_finished(epoch, batch_index, total_loss)
        return total_loss

    def finish_epoch(self, epoch):
        for s in self.strategies:
            s.finish_epoch(epoch)

        if self.logger is not None:
            self.logger.train(self._epoch_total_losses, epoch)
            self.logger.policies(self.optimization_policy.policies, epoch)

        self.optimization_policy.epoch_finished(epoch, self._epoch_total_losses)
        self._epoch_total_losses = None

    def validate(self, epoch):
        self.optimization_policy.validation_started(epoch, self._epoch_total_losses)
        all_metrics = {}
        for s in self.strategies:
            strategy_metrics = s.validate(epoch)
            # make sure that keys do not repeat
            assert len(set(all_metrics.keys()) & set(strategy_metrics.keys())) == 0
            all_metrics.update(strategy_metrics)
        return all_metrics

    @property
    def is_active(self):
        return any(s.is_active for s in self.strategies)

    def __enter__(self):
        for s in self.strategies:
            s.__enter__()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        for s in self.strategies:
            s.__exit__(exc_type, exc_val, exc_tb)

    def __del__(self):
        self.close()

    def __len__(self):
        return len(self.strategies)

    def close(self):
        for s in self.strategies:
            s.close()


class CompleteStrategy(CompositeTrainStrategy):
    def __init__(self, strategy: ForwardStrategy, optimization_policy: OptimizationPolicy):
        super().__init__(strategy, optimization_policy=optimization_policy)

    @classmethod
    def from_parameters(cls, *, train_step_parameters: Dict, iterator: Infinite,
                        calculate_loss: Callable, logger: Logger, validate: Callable = None,
                        optimization_policy: OptimizationPolicy):
        strategy = ForwardStrategy(train_step_parameters, iterator, calculate_loss, logger, validate)
        return cls(strategy, optimization_policy)


class TrainManager(BatchProcessor):
    """
    Aggregates multiple TrainStrategies.
    """

    def __init__(self, *strategies: Strategy):
        self.strategies = strategies
        batches_per_epoch = set()
        for x in self.strategies:
            batches_per_epoch.add(x.batches_per_epoch)

        # make sure that all iterators have same batches_per_epoch
        assert len(batches_per_epoch) == 1, len(batches_per_epoch)
        self.batches_per_epoch = batches_per_epoch.pop()

    def __enter__(self):
        for s in self.strategies:
            s.__enter__()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        for s in self.strategies:
            s.__exit__(exc_type, exc_val, exc_tb)

    def start_epoch(self, epoch: int):
        for s in self.strategies:
            s.start_epoch(epoch)

    def process_batch(self, epoch: int, batch_index: int):
        total_loss = []
        for s in self.strategies:
            total_loss.append(s.process_batch(epoch, batch_index))
        return sum(total_loss)

    def finish_epoch(self, epoch: int):
        for s in self.strategies:
            s.finish_epoch(epoch)

    def validate(self, epoch: int):
        all_metrics = {}
        for s in self.strategies:
            strategy_metrics = s.validate(epoch)
            # make sure that keys do not repeat
            assert len(set(all_metrics.keys()) & set(strategy_metrics.keys())) == 0
            all_metrics.update(strategy_metrics)
        return all_metrics


def train_multiple_strategies(*strategies: Strategy, n_epochs: int = np.inf, checkpoints: Checkpoints = None):
    manager = TrainManager(*strategies)
    if checkpoints is None:
        from dpipe.train.base import _DummyCheckpoints
        checkpoints = _DummyCheckpoints()

    start_epoch = checkpoints.restore()
    with manager as m:
        try:
            for epoch in range(start_epoch, n_epochs):
                m.start_epoch(epoch)
                # store losses
                train_losses = []
                for batch_index in range(m.batches_per_epoch):
                    current_loss = m.process_batch(epoch, batch_index)
                    train_losses.append(current_loss)
                # finalize
                m.finish_epoch(epoch)
                # finalize
                metrics = m.validate(epoch)
                checkpoints.save(epoch, train_losses=train_losses, metrics=metrics)

        except EarlyStopping:
            pass


def extract_loss(loss, loss_key):
    if loss_key is not None:
        return loss[loss_key], dmap(to_np, loss)
    return loss, to_np(loss)


def split_inputs_targets(*inputs: np.ndarray, architecture: nn.Module, n_targets: int = 1):
    if n_targets >= 0:
        n_inputs = len(inputs) - n_targets
    else:
        n_inputs = -n_targets

    assert 0 <= n_inputs <= len(inputs)
    inputs = sequence_to_var(*inputs, device=architecture)
    inputs, targets = inputs[:n_inputs], inputs[n_inputs:]
    return inputs, targets


def calculate_model_loss(*inputs: np.ndarray, architecture: nn.Module, criterion: Callable, use_hf: bool,
                         n_targets: int, loss_key: str = None, **kwargs):
    architecture.train()
    inputs, targets = split_inputs_targets(*inputs, architecture=architecture, n_targets=n_targets)
    with torch.cuda.amp.autocast(use_hf):
        loss = criterion(*lmap(architecture, inputs), *targets, **kwargs)

    return extract_loss(loss, loss_key)
