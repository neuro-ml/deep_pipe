from typing import Union, Callable

import torch
from torch.nn import functional


def focal_loss_with_logits(logits: torch.Tensor, target: torch.Tensor, gamma: float = 2, weight: torch.Tensor = None,
                           reduce: Union[Callable, None] = torch.mean):
    """
    Function that measures Focal Loss between target and output logits.

    Parameters
    ----------
    logits: torch.Tensor
        tensor of an arbitrary shape.
    target: torch.Tensor
        tensor of the same shape as ``logits``.
    gamma: float
        the power of focal loss factor.
    weight: torch.Tensor, None, optional
        a manual rescaling weight. Must be broadcastable to ``logits``.
    reduce: Callable, None, optional
        the reduction operation to be applied to the final loss. Defaults to ``torch.mean``.
        If ``None`` - no reduction will be performed.

    References
    ----------
    `Focal Loss <https://arxiv.org/abs/1708.02002>`_
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
        tensor of an arbitrary shape.
    target: torch.Tensor
        tensor of the same shape as ``logits``.
    gamma: float
        multiplication coefficient for ``logits`` tensor.
    beta: float
        coefficient to be added to all the elements in ``logits`` tensor.
    weight: torch.Tensor
        a manual rescaling weight. Must be broadcastable to ``logits``.
    reduce: Callable, None, optional
        the reduction operation to be applied to the final loss. Defaults to ``torch.mean``.
        If None - no reduction will be performed.

    References
    ----------
    `Focal Loss <https://arxiv.org/abs/1708.02002>`_
    """
    loss = functional.binary_cross_entropy_with_logits(gamma * logits + beta, target, weight, reduction='none') / gamma
    if reduce is not None:
        loss = reduce(loss)
    return loss


def weighted_cross_entropy_with_logits(logit: torch.Tensor, target: torch.Tensor, weight: torch.Tensor = None,
                                       alpha: float = 1, adaptive: bool = False,
                                       reduce: Union[Callable, None] = torch.mean):
    """
    Function that measures Binary Cross Entropy between target and output logits.
    This version of BCE has additional options of constant or adaptive weighting of positive examples.

    Parameters
    ----------
    logit: torch.Tensor
        tensor of an arbitrary shape.
    target: torch.Tensor
        tensor of the same shape as ``logits``.
    weight: torch.Tensor
        a manual rescaling weight. Must be broadcastable to ``logits``.
    alpha: float, optional
        a weight for the positive class examples.
    adaptive: bool, optional
        If ``True``, uses adaptive weight ``[N - sum(p_i)] / sum(p_i)`` for a positive class examples.
    reduce: Callable, None, optional
        the reduction operation to be applied to the final loss. Defaults to ``torch.mean``.
        If None - no reduction will be performed.

    References
    ----------
    `WCE <https://arxiv.org/abs/1707.03237>`_
    """
    if not (target.size() == logit.size()):
        raise ValueError("Target size ({}) must be the same as logit size ({})".format(target.size(), logit.size()))

    if adaptive:
        pos_weight = alpha * (logit.numel() - (torch.sigmoid(logit)).sum()) / (torch.sigmoid(logit)).sum()
    else:
        pos_weight = alpha

    max_val = - logit.clamp(min=0)
    loss = - pos_weight * target * (logit + max_val - (max_val.exp() + (logit + max_val).exp()).log()) \
           + (1 - target) * (-max_val + (max_val.exp() + (logit + max_val).exp()).log())

    if weight is not None:
        loss = loss * weight
    if reduce is not None:
        loss = reduce(loss)
    return loss


# TODO: this function looks too hardcoded
def dice_loss_with_logits(logit: torch.Tensor, target: torch.Tensor, weight: torch.Tensor = None):
    """
    References
    ----------
    `Dice Loss <https://arxiv.org/abs/1606.04797>`_
    """
    if not (target.size() == logit.size()):
        raise ValueError("Target size ({}) must be the same as logit size ({})".format(target.size(), logit.size()))

    preds = torch.sigmoid(logit)

    # TODO: why so complicated?
    sum_dims = [-i for i in range(logit.dim() - 1, 0, -1)]

    dice = 2 * torch.sum(preds * target, dim=sum_dims) \
           / (torch.sum(preds ** 2, dim=sum_dims) + torch.sum(target ** 2, dim=sum_dims))

    loss = 1 - dice

    return loss.mean()
