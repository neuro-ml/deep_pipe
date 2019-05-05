from typing import Union, Callable, Sequence

import torch
from torch import nn
from torch.nn import functional


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


def focal_loss_with_logits(logits: torch.Tensor, target: torch.Tensor, gamma: float = 2, weight: torch.Tensor = None,
                           reduce: Union[Callable, None] = torch.mean):
    """
    Function that measures Focal Loss between target and output logits.

    Parameters
    ----------
    logits: tensor of arbitrary shape.
    target: tensor of the same shape as ``logits``.
    gamma: float
        the power of focal loss factor.
    weight
        a manual rescaling weight. Must be broadcastable to ``logits``.
    reduce
        the reduction operation to be applied to the final loss. Defaults to `torch.mean`.
        If None - no reduction will be performed.
    """
    if not (target.size() == logits.size()):
        raise ValueError("Target size ({}) must be the same as logits size ({})".format(target.size(), logits.size()))

    min_val = - logits.clamp(min=0)
    max_val = (-logits).clamp(min=0)

    prob = (min_val + logits).exp() / (min_val.exp() + (min_val + logits).exp())

    loss = ((1 - 2 * prob) * target + prob) ** gamma * (
            logits - logits * target + max_val + ((-max_val).exp() + (-logits - max_val).exp()).log())

    if weight is not None:
        loss = loss * weight
    if reduce is not None:
        loss = reduce(loss)
    return loss


def linear_focal_loss_with_logits(logits: torch.Tensor, target: torch.Tensor, gamma: float, beta: float,
                                  weight: torch.Tensor = None, reduce: Union[Callable, None] = torch.mean):
    """
    Function that measures Linear Focal Loss between target and output logits.

    Parameters
    ----------
    logits: tensor of arbitrary shape.
    target: tensor of the same shape as ``logits``.
    gamma, beta: float
        focal loss parameters.
    weight
        a manual rescaling weight. Must be broadcastable to ``logits``.
    reduce
        the reduction operation to be applied to the final loss. Defaults to `torch.mean`.
        If None - no reduction will be performed.
    """
    loss = functional.binary_cross_entropy_with_logits(gamma * logits + beta, target, weight, reduction='none') / gamma
    if reduce is not None:
        loss = reduce(loss)
    return loss
