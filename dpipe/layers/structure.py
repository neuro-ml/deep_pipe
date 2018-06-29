import numpy as np
import torch
import torch.nn as nn
from torch.nn import functional

from dpipe.medim.utils import build_slices


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


class CenteredCrop(nn.Module):
    def __init__(self, start, stop=None):
        super().__init__()

        if stop is None:
            start = np.asarray(start)
            stop = np.where(start, -start, None)

        self.slices = (slice(None), slice(None), *build_slices(start, stop))

    def forward(self, x):
        return x[self.slices]


class Reshape(nn.Module):
    def __init__(self, *shape):
        super().__init__()
        self.shape = shape

    def forward(self, x: torch.Tensor):
        batch_size = x.shape[0]
        return x.reshape(batch_size, *self.shape)


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
            result = result + path(x)

        return result


class SplitReduce(nn.Module):
    def __init__(self, reduce, *paths):
        super().__init__()
        self.reduce = reduce
        self.paths = nn.ModuleList(list(paths))

    def forward(self, x):
        return self.reduce(path(x) for path in self.paths)


class UpsampleToInput(nn.Module):
    def __init__(self, path, mode='nearest'):
        super().__init__()
        self.path = path
        self.mode = mode

    def forward(self, x):
        shape = x.shape[2:]
        return functional.upsample(self.path(x), size=shape, mode=self.mode)


# backwards compatibility

# Deprecated
# ----------

compose_blocks = make_pipeline
