from torch import nn

from dpipe.config import register
from dpipe.model_core.enet_torch import Stage3D, InitialBlock3D


@register()
class ResRegressor(nn.Module):
    def __init__(self, n_chans_in):
        super().__init__()

        self.conv_path = nn.Sequential(
            InitialBlock3D(n_chans_in),
            Stage3D(16, 64, 5, downsample=True),
            Stage3D(64, 128, 9, downsample=True),
            Stage3D(128, 128, 8),
        )

        self._reg_path = None

    def reg_path(self, x):
        batch_size = x.size()[0]
        x = x.view(batch_size, -1)
        units = x.size()[-1]
        if self._reg_path is None:
            self._reg_path = nn.Sequential(
                nn.Linear(units, units // 10, bias=False),
                nn.ReLU(True),
                nn.Linear(units // 10, units // 100, bias=False),
                nn.ReLU(True),
                nn.Linear(units // 100, 1, bias=False),
            )
            if next(self.parameters()).is_cuda:
                self._reg_path = self._reg_path.cuda()
        return self._reg_path(x)

    def forward(self, x):
        x = self.conv_path(x)
        return self.reg_path(x)
