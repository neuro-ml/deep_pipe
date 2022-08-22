import pytest

import torch
import numpy as np
from torch import nn
from dpipe.torch import train_step
from dpipe.train import train


@pytest.mark.parametrize('batch_size', [4, 16, 64])
def test_train(batch_size):
    net1 = nn.Sequential(
        nn.Conv2d(3, 4, kernel_size=3, padding=1),
        nn.LayerNorm([28, 28]),
        nn.GELU(),
        nn.Conv2d(4, 8, kernel_size=3, padding=1),
        nn.LayerNorm([28, 28]),
        nn.GELU(),
        nn.Conv2d(8, 16, kernel_size=3, padding=1),
        nn.LayerNorm([28, 28]),
        nn.GELU(),
    )
    net2 = nn.Sequential(
        nn.Conv2d(3, 4, kernel_size=3, padding=1),
        nn.LayerNorm([28, 28]),
        nn.GELU(),
        nn.Conv2d(4, 8, kernel_size=3, padding=1),
        nn.LayerNorm([28, 28]),
        nn.GELU(),
        nn.Conv2d(8, 16, kernel_size=3, padding=1),
        nn.LayerNorm([28, 28]),
        nn.GELU(),
    )
    net2.load_state_dict(net1.state_dict())

    opt1 = torch.optim.SGD(net1.parameters(), lr=3e-4)
    opt2 = torch.optim.SGD(net2.parameters(), lr=3e-4)

    n_epochs = 10
    n_batches = 10

    data = np.random.randn(n_batches, batch_size, 3, 28, 28).astype(np.float32)

    def batch_iter1():
        for x in data:
            yield x, 0

    def batch_iter2():
        for x in data:
            for batch_el in x:
                yield batch_el[None], 0

    def criterion(x, y):
        return x.mean()

    train(
        train_step,
        batch_iter1,
        n_epochs=n_epochs,
        architecture=net1,
        optimizer=opt1,
        criterion=criterion,
    )

    train(
        train_step,
        batch_iter2,
        n_epochs=n_epochs,
        architecture=net2,
        optimizer=opt2,
        criterion=criterion,
        gradient_accumulation_steps=batch_size,
    )

    for param1, param2 in zip(net1.parameters(), net2.parameters()):
        assert torch.allclose(param1, param2)
