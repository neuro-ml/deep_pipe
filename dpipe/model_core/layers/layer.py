import numpy as np
import torch.nn as nn

from dpipe.medim.utils import build_slices


class CenteredCrop(nn.Module):
    def __init__(self, start, stop=None):
        super().__init__()

        if stop is None:
            start = np.array(start)
            stop = np.where(start, -start, None)

        self.slices = tuple([slice(None)] * 2 + build_slices(start, stop))

    def forward(self, x):
        return x[self.slices]
