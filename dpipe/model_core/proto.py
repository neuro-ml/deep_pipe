from torch import nn

from dpipe.config import register


@register()
class Proto(nn.Module):
    def __init__(self):
        super().__init__()
        self.layer = nn.Linear(28 ** 2, 10)

    def forward(self, x):
        return nn.functional.relu(self.layer(x.view(-1, 28 ** 2)))
