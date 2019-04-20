from typing import Union, Callable

import numpy as np
import torch
import torch.nn as nn
from torch.nn import functional

from dpipe.medim.axes import AxesLike, expand_axes
from dpipe.medim.utils import name_changed


class PyramidPooling(nn.Module):
    """
    Implements the pyramid pooling operation.

    Parameters
    ----------
    pooling
        the pooling to be applied, e.g. `torch.nn.functional.max_pool2d`.
    levels
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
    shape
        the final shape. String values denote indices in the input tensor's shape.
        So, (1, '2', 3) will be interpreted as (1, input.shape[2], 3).
    """

    def __init__(self, *shape: Union[int, str]):
        super().__init__()
        self.shape = shape

    def forward(self, x: torch.Tensor):
        shape = [x.shape[int(i)] if isinstance(i, str) else i for i in self.shape]
        return x.reshape(*shape)


class InterpolateToInput(nn.Module):
    """
    Interpolates the result of ``path`` to the original shape along the ``axes``.
    If ``axes`` is None - the result is interpolated along all the axes.
    """

    def __init__(self, path: nn.Module, mode='nearest', axes: AxesLike = None):
        super().__init__()
        self.axes = axes
        self.path = path
        self.mode = mode

    def forward(self, x):
        old_shape = x.shape[2:]
        x = self.path(x)
        axes = expand_axes(self.axes, old_shape)
        new_shape = list(x.shape[2:])
        for i in axes:
            new_shape[i] = old_shape[i]

        if np.not_equal(x.shape[2:], new_shape).any():
            x = functional.upsample(x, size=new_shape, mode=self.mode)
        return x


UpsampleToInput = name_changed(InterpolateToInput, 'UpsampleToInput', '16.03.2019')
