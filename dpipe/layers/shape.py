from typing import Callable, Union

import numpy as np
import torch
import torch.nn as nn
from torch.nn import functional

from ..im.axes import AxesLike, axis_from_dim
from ..torch.functional import moveaxis, softmax


class InterpolateToInput(nn.Module):
    """
    Interpolates the result of ``path`` to the original shape along the spatial ``axis``.

    Parameters
    ----------
    path: nn.Module
        arbitrary neural network module to calculate the result.
    mode: str
        algorithm used for upsampling.
        Should be one of 'nearest' | 'linear' | 'bilinear' | 'trilinear' | 'area'. Default is 'nearest'.
    axis: AxesLike, None, optional
        spatial axes to interpolate result along.
        If ``axes`` is ``None``, the result is interpolated along all the spatial axes.
    """

    def __init__(self, path: nn.Module, mode: str = 'nearest', axis: AxesLike = None):
        super().__init__()
        self.axes = axis
        self.path = path
        self.mode = mode

    def forward(self, x):
        old_shape = x.shape[2:]
        axis = axis_from_dim(self.axes, len(old_shape))

        x = self.path(x)
        new_shape = list(x.shape[2:])
        for i in axis:
            new_shape[i] = old_shape[i]

        if np.not_equal(x.shape[2:], new_shape).any():
            x = functional.interpolate(x, size=new_shape, mode=self.mode, align_corners=False)
        return x


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


class MoveAxis(nn.Module):
    def __init__(self, source: AxesLike, destination: AxesLike):
        super().__init__()
        self.source, self.destination = source, destination

    def forward(self, x):
        return moveaxis(x, self.source, self.destination)


class Softmax(nn.Module):
    """
    A multidimensional version of softmax.
    """

    def __init__(self, axis):
        super().__init__()
        self.axis = axis

    def forward(self, x):
        return softmax(x, self.axis)


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
            # adaptive pooling
            level = 2 ** level
            stride = np.floor(shape / level)
            kernel_size = shape - (level - 1) * stride
            stride, kernel_size = tuple(map(int, stride)), tuple(map(int, kernel_size))
            temp = self.pooling(x, kernel_size=kernel_size, stride=stride)
            pyramid.append(temp.view(batch_size, -1))

        return torch.cat(pyramid, dim=-1)

    @staticmethod
    def get_multiplier(levels, ndim):
        return (2 ** (ndim * levels) - 1) // (2 ** ndim - 1)

    @staticmethod
    def get_out_features(in_features, levels, ndim):
        return in_features * PyramidPooling.get_multiplier(levels, ndim)
