import torch
import torch.nn as nn
from torch.nn import functional


def make_pipeline(structure, make_transformer):
    assert all([type(s) is int for s in structure]), f'{structure}'
    return nn.Sequential(*[
        make_transformer(n_chans_in, n_chans_out) for n_chans_in, n_chans_out in zip(structure[:-1], structure[1:])
    ])


def make_blocks_with_splitters(structure, make_block, make_splitter):
    if len(structure) == 1:
        return make_block(structure[0])
    else:
        return nn.Sequential(make_block(structure[0]),
                             make_splitter(),
                             make_blocks_with_splitters(structure[1:], make_block, make_splitter))


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
