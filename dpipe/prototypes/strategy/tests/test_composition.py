import pytest
import numpy as np

from dpipe.torch import to_np
from dpipe.train import ConsoleLogger
from dpipe.torch import inference_step
from dpipe.prototypes import GradientsAccumulator, LossAccumulator, train_multiple_strategies


@pytest.mark.parametrize("use_hf", [True, False])
@pytest.mark.parametrize("accum", [GradientsAccumulator, LossAccumulator])
def test_forward_loss(composite_strategy, use_hf, accum, task_data):
    task_inputs, task_gt = task_data
    s, model = composite_strategy(accum, use_hf, lr=0.)
    first_pred = inference_step(task_inputs[..., None], architecture=model)[..., 0]
    second_pred = inference_step(task_gt[..., None], architecture=model)[..., 0]
    # loss weight  2
    first_loss = 2 * np.mean((first_pred - task_gt) ** 2)
    second_loss = 2 * np.mean((first_pred - second_pred) ** 2)
    composed_loss = first_loss + second_loss
    s.start_epoch(0)
    first_batch_loss = s.process_batch(0, 0)
    assert np.isclose(composed_loss, first_batch_loss, atol=1e-3)
    second_batch_loss = s.process_batch(0, 1)
    assert np.isclose(composed_loss, second_batch_loss, atol=1e-3)


@pytest.mark.parametrize("use_hf", [True, False])
@pytest.mark.parametrize("accum", [GradientsAccumulator, LossAccumulator])
def test_optimization(composite_strategy, use_hf, accum, task_data):
    lr = 0.1
    s, model = composite_strategy(accum, use_hf, lr=lr)
    x_data, y_data = task_data
    # test several iterations
    s.start_epoch(0)
    for batch_index in range(4):
        init_bias = to_np(model.bias).copy()
        init_weight = to_np(model.weight).copy()
        pred = inference_step(x_data[..., None], architecture=model)[..., 0]
        s.process_batch(0, batch_index)
        # correct gradients
        b_grad = 4 * np.mean((pred - y_data)) + 0.
        w_grad = 4 / len(x_data) * np.dot(x_data, pred - y_data) + 4 * init_weight * np.mean((x_data - y_data) ** 2)
        # check gradients
        atol = 1e-3
        assert np.isclose(to_np(model.bias.grad), b_grad, atol=atol)
        assert np.isclose(to_np(model.weight.grad), w_grad, atol=atol)
        # check updates
        assert np.isclose(to_np(model.weight), init_weight - lr * w_grad, atol=atol)
        assert np.isclose(to_np(model.bias), init_bias - lr * b_grad, atol=atol)


@pytest.mark.parametrize("use_hf", [True, False])
@pytest.mark.parametrize("accum", [GradientsAccumulator, LossAccumulator])
def test_loss_logging(composite_strategy, use_hf, accum, task_data):
    logger = ConsoleLogger()
    strategy, _ = composite_strategy(accum, use_hf, lr=0.1, composed_logger=logger)
    train_multiple_strategies(strategy, n_epochs=2)
