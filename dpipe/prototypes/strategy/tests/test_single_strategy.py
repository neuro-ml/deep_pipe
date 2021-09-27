import pytest
import numpy as np

from dpipe.torch import to_np
from dpipe.train import ConsoleLogger
from dpipe.torch import inference_step
from dpipe.train.policy import Schedule
from dpipe.prototypes import GradientsAccumulator, LossAccumulator, train_multiple_strategies


@pytest.mark.parametrize("use_hf", [False, True])
def test_single_forward_loss(forward_strategy_all_ds_batch, task_data, use_hf):
    task_inputs, task_gt = task_data
    s, model = forward_strategy_all_ds_batch(use_hf, n_targets=1)
    s.start_epoch(0)
    pred = inference_step(task_inputs[..., None], architecture=model)[..., 0]
    # loss weight == 2
    true_loss = 2 * np.mean((pred - task_gt) ** 2)
    first_batch_loss = to_np(s.process_batch(0, 0))
    assert np.isclose(true_loss, first_batch_loss)
    second_batch_loss = to_np(s.process_batch(0, 1))
    assert np.isclose(true_loss, second_batch_loss)


@pytest.mark.parametrize("use_hf", [False, True])
@pytest.mark.parametrize("accum", [GradientsAccumulator, LossAccumulator])
def test_single_optimization(accum, use_hf, single_strategy, task_data):
    lr = 0.1
    task_inputs, task_gt = task_data
    s, model = single_strategy(accum, use_hf, lr=lr, n_targets=1)
    x_data, y_data = task_data
    # test several iterations
    s.start_epoch(0)
    for batch_index in range(4):
        init_bias = to_np(model.bias).copy()
        init_weight = to_np(model.weight).copy()
        pred = inference_step(x_data[..., None], architecture=model)[..., 0]
        s.process_batch(0, batch_index)
        # correct gradients
        w_grad = 4 / len(x_data) * np.dot(x_data, pred - y_data)
        b_grad = 4 * np.mean((pred - y_data))
        # check gradients
        atol = 1e-3
        assert np.isclose(to_np(model.bias.grad), b_grad, atol=atol)
        assert np.isclose(to_np(model.weight.grad), w_grad, atol=atol)
        # check updates
        assert np.isclose(to_np(model.weight), init_weight - lr * w_grad, atol=atol)
        assert np.isclose(to_np(model.bias), init_bias - lr * b_grad, atol=atol)


@pytest.mark.parametrize("use_hf", [True, False])
@pytest.mark.parametrize("accum", [GradientsAccumulator, LossAccumulator])
def test_single_strategy_lifecycle(accum, use_hf, single_strategy):
    weights = []

    class Logger(ConsoleLogger):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)

        def policies(self, policies: dict, step: int):
            nonlocal weights
            weights.append(policies['weight'])
            return super().policies(policies, step)

    # check policy update
    lr = Schedule(initial=1, epoch2value_multiplier={1: 2, 2: 3, 3: 4, 4: 5})
    weight = Schedule(initial=1, epoch2value_multiplier={1: 2, 2: 3, 3: 4, 4: 5})
    s, _ = single_strategy(accum, use_hf, lr=lr, weight=weight, logger=Logger(), n_targets=1)
    counter = 0

    def validate():
        nonlocal counter
        counter += 1
        return {'counter': counter}

    s.strategies[0].validate_step = validate
    train_multiple_strategies(s, n_epochs=5)
    assert counter == 5
    assert weights == [1, 2, 6, 24, 120]

    for param_group in s.optimization_policy.optimizer.param_groups:
        assert param_group['lr'] == 120


@pytest.mark.parametrize("use_hf", [True, False])
@pytest.mark.parametrize("accum", [GradientsAccumulator, LossAccumulator])
def test_double_grad_propagation(accum, use_hf, single_strategy, task_data):
    lr = 1
    s, model = single_strategy(accum, use_hf, lr=lr, n_targets=0)
    x_data, y_data = task_data
    s.start_epoch(0)

    # test several iterations
    for batch_index in range(4):
        init_bias = to_np(model.bias).copy()
        init_weight = to_np(model.weight).copy()

        s.process_batch(0, batch_index)
        # correct gradients
        atol = 1e-4
        b_grad = 0.
        w_grad = 4 * init_weight * np.mean((x_data - y_data) ** 2)
        # check gradients
        assert np.isclose(to_np(model.bias.grad), b_grad, atol=atol)
        assert np.isclose(to_np(model.weight.grad), w_grad, atol=atol)
        # check updates
        assert np.isclose(to_np(model.weight), init_weight - lr * w_grad, atol=atol)
        assert np.isclose(to_np(model.bias), init_bias - lr * b_grad, atol=atol)
