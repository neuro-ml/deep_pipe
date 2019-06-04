from typing import Union, Callable

import torch
from torch.nn import functional
from .structure import make_consistent_seq


def focal_loss_with_logits(logits: torch.Tensor, target: torch.Tensor, gamma: float = 2, weight: torch.Tensor = None,
                           reduce: Union[Callable, None] = torch.mean):
    """
    Function that measures Focal Loss between target and output logits.

    Parameters
    ----------
    logits: torch.Tensor
        tensor of arbitrary shape.
    target: torch.Tensor
        tensor of the same shape as ``logits``.
    gamma: float
        the power of focal loss factor.
    weight: torch.Tensor, None, optional
        a manual rescaling weight. Must be broadcastable to ``logits``.
    reduce: Callable, None, optional
        the reduction operation to be applied to the final loss. Defaults to ``torch.mean``.
        If ``None`` - no reduction will be performed.
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
    Equals to BinaryCrossEntropy( ``gamma`` *  ``logits`` + ``beta``, ``target`` , ``weights``).

    Parameters
    ----------
    logits: torch.Tensor
        tensor of arbitrary shape.
    target: torch.Tensor
        tensor of the same shape as ``logits``.
    gamma: float
        multiplication coefficient for ``logits`` tensor.
    beta: float
        coefficient to be added to all the elements in ``logits`` tensor.
    weight
        a manual rescaling weight. Must be broadcastable to ``logits``.
    reduce
        the reduction operation to be applied to the final loss. Defaults to ``torch.mean``.
        If None - no reduction will be performed.
    """
    loss = functional.binary_cross_entropy_with_logits(gamma * logits + beta, target, weight, reduction='none') / gamma
    if reduce is not None:
        loss = reduce(loss)
    return loss
