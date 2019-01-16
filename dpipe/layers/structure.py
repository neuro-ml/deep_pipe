import numpy as np
import torch
import torch.nn as nn
from torch.nn import functional

from dpipe.medim.utils import build_slices, pam


def make_pipeline(structure, make_transformer):
    assert all(isinstance(s, int) for s in structure), f'{structure}'
    return nn.Sequential(*[
        make_transformer(n_chans_in, n_chans_out) for n_chans_in, n_chans_out in zip(structure[:-1], structure[1:])
    ])


def build_fpn(structure, make_block, make_up, make_down, split_merge):
    line, *down_structure = structure
    if len(down_structure) == 0:
        assert len(line) == 1, 'f{line}'
        return make_pipeline(line[0], make_block)
    else:
        assert len(line) == 3, f'{line}'
        inner_path = line[1] if isinstance(line[1], nn.Module) else make_pipeline(line[1], make_block)
        down_path = nn.Sequential(make_down(), *build_fpn(down_structure, make_block, make_up, make_down, split_merge),
                                  make_up())
        return nn.Sequential(*make_pipeline(line[0], make_block),
                             split_merge(down_path, inner_path),
                             *make_pipeline(line[2], make_block))


def make_blocks_with_splitters(structure, make_block, make_splitter):
    if len(structure) == 1:
        return make_block(structure)
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


class SplitReduce(nn.Module):
    def __init__(self, reduce, *paths):
        super().__init__()
        self.reduce = reduce
        self.paths = nn.ModuleList(list(paths))

    def forward(self, x):
        return self.reduce(pam(self.paths, x))


class SplitCat(SplitReduce):
    def __init__(self, *paths):
        super().__init__(lambda x: torch.cat(tuple(x), dim=1), *paths)


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


class UpsampleToInput(nn.Module):
    def __init__(self, path, mode='nearest'):
        super().__init__()
        self.path = path
        self.mode = mode

    def forward(self, x):
        shape = x.shape[2:]
        return functional.upsample(self.path(x), size=shape, mode=self.mode)
