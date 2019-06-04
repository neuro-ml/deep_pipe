from typing import Callable, Union

import numpy as np
import torch
import torch.nn as nn
from torch.nn import functional

from dpipe.medim.utils import build_slices, pam
from dpipe.medim.axes import AxesLike, expand_axes
from .fpn import FPN


def make_consistent_seq(layer: Callable, channels: Sequence[int], *args, **kwargs):
    """
    Builds a sequence of layers that have consistent input and output channels/features.

    ``args`` and ``kwargs`` are passed as additional parameters.

    Examples
    --------
    >>> make_consistent_seq(nn.Conv2d, [16, 32, 64, 128], kernel_size=3, padding=1)
    >>> # same as
    >>> nn.Sequential(
    >>>     nn.Conv2d(16, 32, kernel_size=3, padding=1),
    >>>     nn.Conv2d(32, 64, kernel_size=3, padding=1),
    >>>     nn.Conv2d(64, 128, kernel_size=3, padding=1),
    >>> )
    """
    return nn.Sequential(*(layer(in_, out, *args, **kwargs) for in_, out in zip(channels, channels[1:])))


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


class PyramidPooling(nn.Module):
    """
    Implements the pyramid pooling operation.

    Parameters
    ----------
    pooling: Callable
        the pooling to be applied, e.g. ``torch.nn.functional.max_pool2d``.
    levels: int
        the number of pyramid levels, default is 1 which is the global pooling operation.
    """

    def __init__(self, pooling: Callable, levels: int = 1):
        super().__init__()
        self.levels = levels
        self.pooling = pooling

    def forward(self, x):
        assert x.dim() > 2
        shape = np.array(x.shape[2:], dtype=int)
        batch_size = x.shape[0]
        pyramid = []

        for level in range(self.levels):
            level = 2 ** level
            stride = tuple(map(int, np.floor(shape / level)))
            kernel_size = tuple(map(int, np.ceil(shape / level)))
            temp = self.pooling(x, kernel_size=kernel_size, stride=stride)
            pyramid.append(temp.view(batch_size, -1))

        return torch.cat(pyramid, dim=-1)

    @staticmethod
    def get_multiplier(levels, ndim):
        return (2 ** (ndim * levels) - 1) // (2 ** ndim - 1)

    @staticmethod
    def get_out_features(in_features, levels, ndim):
        return in_features * PyramidPooling.get_multiplier(levels, ndim)


class Lambda(nn.Module):
    """Applies ``func`` to the incoming tensor."""

    def __init__(self, func):
        super().__init__()
        self.func = func

    def forward(self, *args, **kwargs):
        return self.func(*args, **kwargs)


class Reshape(nn.Module):
    """
    Reshape the incoming tensor to the given ``shape``.

    Parameters
    ----------
    shape: Union[int, str]
        the resulting shape. String values denote indices in the input tensor's shape.

    Examples
    --------
    >>> layer = Reshape('0', '1', 500, 500)
    >>> layer(x)
    >>> # same as
    >>> x.reshape(x.shape[0], x.shape[1], 500, 500)
    """

    def __init__(self, *shape: Union[int, str]):
        super().__init__()
        self.shape = shape

    def forward(self, x: torch.Tensor):
        shape = [x.shape[int(i)] if isinstance(i, str) else i for i in self.shape]
        return x.reshape(*shape)


class InterpolateToInput(nn.Module):
    """
    Interpolates the result of ``path`` to the original shape along the spatial ``axes``.

    Parameters
    ----------
    path: nn.Module
        arbitrary neural network module to calculate the result.
    mode: str
        algorithm used for upsampling in `functional.interpolate`.
        Should be one of 'nearest' | 'linear' | 'bilinear' | 'trilinear' | 'area'. Default is 'nearest'.
    axes: AxesLike, None, optional
        spatial axes to interpolate result along.
        If ``axes`` is ``None``, the result is interpolated along all the spatial axes.
    """

    def __init__(self, path: nn.Module, mode: str = 'nearest', axes: AxesLike = None):
        super().__init__()
        self.axes = axes
        self.path = path
        self.mode = mode

    def forward(self, x):
        old_shape = x.shape[2:]
        axes = expand_axes(self.axes, old_shape)
        x = self.path(x)
        new_shape = list(x.shape[2:])
        for i in axes:
            new_shape[i] = old_shape[i]

        if np.not_equal(x.shape[2:], new_shape).any():
            x = functional.interpolate(x, size=new_shape, mode=self.mode)
        return x


@np.deprecate(message='Use `dpipe.batch_iter.functional.make_consistent_seq` instead.')  # 04.06.19
def make_pipeline(structure, make_transformer):
    assert all(isinstance(s, int) for s in structure), f'{structure}'
    return nn.Sequential(*[
        make_transformer(n_chans_in, n_chans_out) for n_chans_in, n_chans_out in zip(structure[:-1], structure[1:])
    ])


@np.deprecate(message='Use `dpipe.batch_iter.structure.FPN` instead.')  # 04.06.19
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
