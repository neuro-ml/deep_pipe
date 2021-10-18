from functools import partial
from typing import Callable, Sequence, Union
from warnings import warn

import torch
import torch.nn as nn
from torch.nn import functional
import numpy as np

from dpipe.itertools import zip_equal, lmap
from dpipe.im.utils import identity
from dpipe.torch.utils import order_to_mode
from .structure import ConsistentSequential


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
    kwargs
        additional arguments passed to ``layer``.

    Examples
    --------
    >>> from dpipe.layers import ResBlock2d
    >>>
    >>> structure = [
    >>>     [[16, 16, 16],       [16, 16, 16]],  # level 1, left and right
    >>>     [[16, 32, 32],       [32, 32, 16]],  # level 2, left and right
    >>>                [32, 64, 32]              # final level
    >>> ]
    >>>
    >>> upsample = nn.Upsample(scale_factor=2, mode='bilinear')
    >>> downsample = nn.MaxPool2d(kernel_size=2)
    >>>
    >>> ResUNet = FPN(
    >>>     ResBlock2d, downsample, upsample, torch.add,
    >>>     structure, kernel_size=3, dilation=1, padding=1, last_level=True
    >>> )

    References
    ----------
    `make_consistent_seq` `FPN <https://arxiv.org/pdf/1612.03144.pdf>`_ `UNet <https://arxiv.org/pdf/1505.04597.pdf>`_
    """

    def __init__(self, layer: Callable, downsample: Union[nn.Module, Callable], upsample: Union[nn.Module, Callable],
                 merge: Callable, structure: Sequence[Sequence[Union[Sequence[int], nn.Module]]],
                 last_level: bool = True, **kwargs):
        super().__init__()

        def build_level(path):
            if isinstance(path, nn.Module):
                return path

            elif not isinstance(path, Sequence) or not all(isinstance(x, int) for x in path):
                raise ValueError('The passed `structure` is not valid.')

            return ConsistentSequential(layer, path, **kwargs)

        def make_up_down(o):
            if not isinstance(o, nn.Module):
                o = o()
            return o

        *levels, bridge = structure
        # handling the case [[...]]
        if len(bridge) == 1 and isinstance(bridge[0], Sequence):
            bridge = bridge[0]

        self.bridge = build_level(bridge)
        self.merge = merge
        self.last_level = last_level
        self.downsample = nn.ModuleList([make_up_down(downsample) for _ in levels])
        self.upsample = nn.ModuleList([make_up_down(upsample) for _ in levels])

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


def interpolate_merge(merge: Callable, order: int = 0):
    return lambda left, down: merge(*interpolate_to_left(left, down, order))


def interpolate_to_left(left: torch.Tensor, down: torch.Tensor, order: int = 0, *, mode: str = None):
    if mode is not None:
        msg = 'Argument `mode` is deprecated. Use `order` instead.'
        warn(msg, UserWarning)
        warn(msg, DeprecationWarning)
        order = mode

    if isinstance(order, int):
        order = order_to_mode(order, len(down.shape) - 2)

    if np.not_equal(left.shape, down.shape).any():
        interpolate = functional.interpolate
        if order in ['linear', 'bilinear', ' bicubic', 'trilinear']:
            interpolate = partial(interpolate, align_corners=False)

        down = interpolate(down, size=left.shape[2:], mode=order)

    return left, down
