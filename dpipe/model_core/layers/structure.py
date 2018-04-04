import torch
import torch.nn as nn
from torch.nn import functional


def compose_blocks(structure, get_block):
    assert all([type(s) is int for s in structure]), f'{structure}'
    return nn.Sequential(*[
        get_block(n_chans_in, n_chans_out) for n_chans_in, n_chans_out in zip(structure[:-1], structure[1:])
    ])


class SplitCat(nn.Module):
    def __init__(self, *paths):
        super().__init__()
        self.paths = nn.ModuleList(list(paths))

    def forward(self, x):
        return torch.cat([path(x) for path in self.paths], dim=1)


class SplitAdd(nn.Module):
    def __init__(self, *paths):
        super().__init__()
        self.init_path, *paths = paths
        self.other_paths = nn.ModuleList(list(paths))

    def forward(self, x):
        result = self.init_path(x)
        for path in self.other_paths:
            result += path(x)

        return result


class SplitReduce(nn.Module):
    def __init__(self, *paths, reduce):
        super().__init__()
        self.reduce = reduce
        self.paths = nn.ModuleList(list(paths))

    def forward(self, x):
        return self.reduce([path(x) for path in self.paths])


class UpsampleToInput(nn.Module):
    def __init__(self, path, mode='nearest'):
        super().__init__()
        self.path = path
        self.mode = mode

    def forward(self, x):
        shape = x.shape[2:]
        return functional.upsample(self.path(x), size=shape, mode=self.mode)
