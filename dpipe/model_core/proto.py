import torch
from torch import nn

from dpipe.config import register


@register()
class Proto(nn.Module):
    def __init__(self, cuda=True):
        super().__init__()
        self.layer = nn.Linear(28 ** 2, 10)

    def forward(self, x):
        x = x.view(-1, 28 ** 2)
        x = self.layer(x)
        return nn.functional.relu(x)
