from typing import Callable, Sequence, Union

import numpy as np
import torch
import torch.nn as nn

from dpipe.medim.itertools import zip_equal, lmap
from dpipe.medim.utils import build_slices, pam, identity
from .functional import make_consistent_seq


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


class FPN(nn.Module):
    """
    Feature Pyramid Network - a generalization of UNet.

    Parameters
    ----------
    layer: Callable
        the structural block of each level, e.g. ``torch.nn.Conv2d``.
    downsample: nn.Module
        the downsampling layer, e.g. ``torch.nn.MaxPool2d``.
    upsample: nn.Module
        the upsampling layer, e.g. ``torch.nn.Upsample``.
    merge: Callable(left, down)
        a function that merges the upsampled features map with the one coming from the left branch,
        e.g. ``torch.add``.
    structure: Sequence[Union[Sequence[int], nn.Module]]
        a collection of channels sequences, see Examples section for details.
    last_level: bool
        If True only the result of the last level is returned (as in UNet),
        otherwise the results from all levels are returned (as in FPN).
    args, kwargs
        additional arguments passed to ``layer``.

    Examples
    --------
    >>> from dpipe.layers import ResBlock2d
    >>> from functools import partial
    >>>
    >>> structure = [
    >>>     [[16, 16, 16],       [16, 16, 16]],  # level 1, left and right
    >>>     [[16, 32, 32],       [32, 32, 16]],  # level 2, left and right
    >>>                [32, 64, 32]              # final level
    >>> ]
    >>>
    >>> upsample = partial(nn.Upsample, scale_factor=2, mode='bilinear')
    >>> downsample = partial(nn.MaxPool2d, kernel_size=2)
    >>>
    >>> ResUNet = FPN(
    >>>     ResBlock2d, downsample, upsample, torch.add,
    >>>     structure, kernel_size=3, dilation=1, padding=1, last_level=True
    >>> )

    References
    ----------
    `make_consistent_seq` `FPN <https://arxiv.org/pdf/1612.03144.pdf>`_ `UNet <https://arxiv.org/pdf/1505.04597.pdf>`_
    """

    def __init__(self, layer: Callable, downsample: nn.Module, upsample: nn.Module, merge: Callable,
                 structure: Sequence[Union[Sequence[int], nn.Module]], last_level: bool = True, *args, **kwargs):
        super().__init__()

        def build_level(path):
            if not isinstance(path, nn.Module):
                path = make_consistent_seq(layer, path, *args, **kwargs)
            return path

        self.bridge = build_level(structure[-1])
        self.merge = merge
        self.last_level = last_level
        self.downsample = nn.ModuleList([downsample() for _ in structure[:-1]])
        self.upsample = nn.ModuleList([upsample() for _ in structure[:-1]])

        # group branches
        branches = []
        for paths in zip_equal(*structure[:-1]):
            branches.append(nn.ModuleList(lmap(build_level, paths)))

        if len(branches) not in [2, 3]:
            raise ValueError(f'Expected 2 or 3 branches, but {len(branches)} provided.')

        self.down_path, self.up_path = branches[0], branches[-1]
        # add middle branch if needed
        if len(branches) == 2:
            self.middle_path = [identity] * len(self.down_path)
        else:
            self.middle_path = branches[1]

    def forward(self, x):
        levels, results = [], []
        for layer, down, middle in zip_equal(self.down_path, self.downsample, self.middle_path):
            x = layer(x)
            levels.append(middle(x))
            x = down(x)

        x = self.bridge(x)
        results.append(x)

        for layer, up, left in zip_equal(reversed(self.up_path), self.upsample, reversed(levels)):
            x = layer(self.merge(left, up(x)))
            results.append(x)

        if self.last_level:
            return x

        return results


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
