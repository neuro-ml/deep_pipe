import pytest
import torch
import numpy as np

from torch.optim import SGD
from functools import partial
from torch.nn import functional as F

from dpipe.train import ConsoleLogger
from dpipe.train.policy import ValuePolicy
from dpipe.im.shape_utils import prepend_dims
from dpipe.batch_iter import Infinite, multiply
from dpipe.prototypes import ForwardStrategy, CompleteStrategy, CompositeTrainStrategy, calculate_model_loss

reg_size = 20
task_size = 11


def make_task_data():
    task_x_values = np.array([0.01 * x for x in range(task_size)]).astype(np.float32)
    task_y_values = np.array([2 * 0.01 * x + 0.02 for x in range(task_size)]).astype(np.float32)
    return task_x_values, task_y_values


def make_batch_iter(data, batches_per_epoch, batch_size):
    x_values, y_values = data

    def sample_batch():
        while True:
            for x, y in zip(x_values, y_values):
                yield x, y

    batch_iterator = Infinite(
        sample_batch(),
        multiply(prepend_dims),
        multiply(np.float32),
        batch_size=batch_size,
        batches_per_epoch=batches_per_epoch
    )
    return batch_iterator


def make_task_batch_iter(batches_per_epoch, batch_size):
    data = make_task_data()
    return make_batch_iter(data, batches_per_epoch, batch_size)


def make_model(weight, bias):
    model = torch.nn.Linear(1, 1)
    with torch.no_grad():
        model.bias = torch.nn.Parameter(torch.reshape(torch.tensor(bias), (1, 1)))
        model.weight = torch.nn.Parameter(torch.reshape(torch.tensor(weight), (1, 1)))
    return model


def make_forward_strategy(batch_size, batches_per_epoch, model, *, logger=None, weight=2, use_hf, n_targets):
    logger = logger or ConsoleLogger()
    batch_iterator = make_task_batch_iter(
        batch_size=batch_size,
        batches_per_epoch=batches_per_epoch,
    )

    def criterion(x, y, *, weight):
        return weight * F.mse_loss(x, y)

    train_parameters = dict(
        weight=weight,
        criterion=criterion,
    )
    calculate_loss = partial(calculate_model_loss, architecture=model, n_targets=n_targets, use_hf=use_hf)
    task_strategy = ForwardStrategy(
        validate=None,
        logger=logger,
        iterator=batch_iterator,
        calculate_loss=calculate_loss,
        train_step_parameters=train_parameters,
    )
    return task_strategy


def make_complete_strategy(optimization_policy, *args, **kwargs):
    forward_strategy = make_forward_strategy(*args, **kwargs)
    complete_strategy = CompleteStrategy(forward_strategy, optimization_policy)
    return complete_strategy


@pytest.fixture()
def model():
    return make_model(0.5, 0.1)


@pytest.fixture(scope='module')
def task_data():
    return make_task_data()


@pytest.fixture(scope='function')
def forward_strategy_all_ds_batch(model, task_data):
    strategy = None

    def make_strategy(use_hf, n_targets):
        if not torch.cuda.is_available():
            use_hf = False

        x_data, y_data = task_data
        s: ForwardStrategy = make_forward_strategy(
            n_targets=n_targets,
            model=model,
            use_hf=use_hf,
            batches_per_epoch=2,
            batch_size=len(x_data),
        )
        nonlocal strategy
        strategy = s
        return s, model

    yield make_strategy
    if strategy and strategy.is_active:
        strategy.close()


@pytest.fixture(scope='function')
def single_strategy(model):
    strategy = None

    def make_strategy(accum, use_hf, lr, **kwargs):
        opt = SGD(model.parameters(), lr=lr.value if isinstance(lr, ValuePolicy) else lr)
        if not torch.cuda.is_available():
            use_hf = False

        scaler = None
        if use_hf:
            scaler = torch.cuda.amp.GradScaler()
            model.cuda()

        nonlocal strategy
        accum = accum(optimizer=opt, scaler=scaler, optimizer_parameters=dict(lr=lr))
        strategy = make_complete_strategy(
            accum, batch_size=11, batches_per_epoch=5, use_hf=use_hf, model=model, **kwargs
        )
        return strategy, model

    yield make_strategy
    if strategy and strategy.is_active:
        strategy.close()


@pytest.fixture(scope='function')
def composite_strategy(model):
    strategies = []

    def make_strategy(accum, use_hf, lr, composed_logger=None, first_logger=None, second_logger=None):
        opt = SGD(model.parameters(), lr=lr.value if isinstance(lr, ValuePolicy) else lr)
        if not torch.cuda.is_available():
            use_hf = False

        scaler = None
        if use_hf:
            scaler = torch.cuda.amp.GradScaler()
            model.cuda()

        accum = accum(optimizer=opt, scaler=scaler, optimizer_parameters=dict(lr=lr))
        # task strategy
        first_strategy = make_forward_strategy(
            batch_size=11, batches_per_epoch=5, use_hf=use_hf, model=model, n_targets=1, logger=first_logger
        )
        # reg strategy
        second_strategy = make_forward_strategy(
            batch_size=11, batches_per_epoch=5, use_hf=use_hf, model=model, n_targets=0, logger=second_logger
        )
        composed = CompositeTrainStrategy(
            first_strategy, second_strategy, optimization_policy=accum, logger=composed_logger
        )
        nonlocal strategies
        strategies.extend([first_strategy, second_strategy])
        return composed, model

    yield make_strategy
    for s in strategies:
        if s.is_active:
            s.close()
