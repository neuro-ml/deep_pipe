from typing import Callable, Sequence

import numpy as np
import torch
import torch.nn as nn

from dpipe.im.utils import build_slices, pam, identity


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
    return ConsistentSequential(layer, channels, *args, **kwargs)


class ConsistentSequential(nn.Sequential):
    """
    A sequence of layers that have consistent input and output channels/features.

    ``args`` and ``kwargs`` are passed as additional parameters.

    Examples
    --------
    >>> ConsistentSequential(nn.Conv2d, [16, 32, 64, 128], kernel_size=3, padding=1)
    >>> # same as
    >>> nn.Sequential(
    >>>     nn.Conv2d(16, 32, kernel_size=3, padding=1),
    >>>     nn.Conv2d(32, 64, kernel_size=3, padding=1),
    >>>     nn.Conv2d(64, 128, kernel_size=3, padding=1),
    >>> )
    """

    def __init__(self, layer: Callable, channels: Sequence[int], *args, **kwargs):
        if len(channels) < 2:
            raise ValueError('`channels` must contain at least two elements.')

        super().__init__(*(layer(in_, out, *args, **kwargs) for in_, out in zip(channels, channels[1:])))


class PreActivation(nn.Module):
    """
    Runs a sequence of batch_norm, activation, and ``layer``.

        in -> (BN -> activation -> layer) -> out

    Parameters
    ----------
    in_features: int
        the number of incoming features/channels.
    out_features: int
        the number of the output features/channels.
    batch_norm_module
        module to build up batch normalization layer, e.g. ``torch.nn.BatchNorm3d``.
    activation_module
        module to build up activation layer. Default is ``torch.nn.ReLU``.
    layer_module: Callable(in_features, out_features, **kwargs)
        module to build up the main layer, e.g. ``torch.nn.Conv3d`` or ``torch.nn.Linear``.
    kwargs
        additional arguments passed to ``layer_module``.
    """

    def __init__(self, in_features: int, out_features: int, *,
                 layer_module, batch_norm_module=None, activation_module=nn.ReLU, **kwargs):
        super().__init__()
        if batch_norm_module is not None:
            self.bn = batch_norm_module(in_features)
        else:
            self.bn = identity
        self.activation = activation_module()
        self.layer = layer_module(in_features, out_features, **kwargs)

    def forward(self, x):
        return self.layer(self.activation(self.bn(x)))


class PostActivation(nn.Module):
    """
    Performs a sequence of layer, batch_norm and activation:

        in -> (layer -> BN -> activation) -> out

    Parameters
    ----------
    in_features: int
        the number of incoming features/channels.
    out_features: int
        the number of the output features/channels.
    batch_norm_module
        module to build up batch normalization layer, e.g. ``torch.nn.BatchNorm3d``.
    activation_module
        module to build up activation layer. Default is ``torch.nn.ReLU``.
    layer_module: Callable(in_features, out_features, **kwargs)
        module to build up the main layer, e.g. ``torch.nn.Conv3d`` or ``torch.nn.Linear``.
    kwargs
        additional arguments passed to ``layer_module``.

    Notes
    -----
    If ``layer`` supports a bias term, make sure to pass ``bias=False``.
    """

    def __init__(self, in_features: int, out_features: int, *,
                 layer_module, batch_norm_module=None, activation_module=nn.ReLU, **kwargs):
        super().__init__()
        self.layer = layer_module(in_features, out_features, **kwargs)
        self.activation = activation_module()
        if batch_norm_module is not None:
            self.bn = batch_norm_module(out_features)
        else:
            self.bn = identity

    def forward(self, x):
        return self.activation(self.bn(self.layer(x)))


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


class Split(SplitReduce):
    def __init__(self, *paths):
        super().__init__(tuple, *paths)


class SplitCat(SplitReduce):
    def __init__(self, *paths, axis=1):
        super().__init__(lambda x: torch.cat(tuple(x), dim=axis), *paths)


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


class Lambda(nn.Module):
    """
    Applies ``func`` to the incoming tensor.

    ``kwargs`` are passed as additional arguments.
    """

    def __init__(self, func, **kwargs):
        super().__init__()
        self.func = func
        self.kwargs = kwargs

    def forward(self, *args, **kwargs):
        return self.func(*args, **kwargs, **self.kwargs)
