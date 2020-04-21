from typing import Union, Callable

import numpy as np
import torch
from torch.nn import functional

from dpipe.im.axes import AxesLike, check_axes

__all__ = [
    'focal_loss_with_logits', 'linear_focal_loss_with_logits', 'weighted_cross_entropy_with_logits',
    'moveaxis', 'softmax',
]


def focal_loss_with_logits(logits: torch.Tensor, target: torch.Tensor, weight: torch.Tensor = None,
                           gamma: float = 2, alpha: float = 0.25, reduce: Union[Callable, None] = torch.mean):
    """
    Function that measures Focal Loss between target and output logits.

    Parameters
    ----------
    logits: torch.Tensor
        tensor of an arbitrary shape.
    target: torch.Tensor
        tensor of the same shape as ``logits``.
    weight: torch.Tensor, None, optional
        a manual rescaling weight. Must be broadcastable to ``logits``.
    gamma: float
        the power of focal loss factor. Defaults to 2.
    alpha: float, None, optional
        weighting factor of the focal loss. If ``None``, no weighting will be performed. Defaults to 0.25.
    reduce: Callable, None, optional
        the reduction operation to be applied to the final loss. Defaults to ``torch.mean``.
        If ``None``, no reduction will be performed.

    References
    ----------
    `Focal Loss <https://arxiv.org/abs/1708.02002>`_
    """
    if not (target.size() == logits.size()):
        raise ValueError("Target size ({}) must be the same as logits size ({})".format(target.size(), logits.size()))

    if alpha is not None:
        if not (0 <= alpha <= 1):
            raise ValueError(f'`alpha` should be between 0 and 1, {alpha} was given')
        rescale_w = (2 * alpha - 1) * target + 1 - alpha
    else:
        rescale_w = 1

    min_val = - logits.clamp(min=0)
    max_val = (-logits).clamp(min=0)

    prob = (min_val + logits).exp() / (min_val.exp() + (min_val + logits).exp())

    loss = rescale_w * ((1 - 2 * prob) * target + prob) ** gamma * (
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
        # TODO: torch.sigmoid(logit).sum() can be reused
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


def dice_loss(pred: torch.Tensor, target: torch.Tensor):
    """
    References
    ----------
    `Dice Loss <https://arxiv.org/abs/1606.04797>`_
    """
    if not (target.size() == pred.size()):
        raise ValueError("Target size ({}) must be the same as logit size ({})".format(target.size(), pred.size()))

    sum_dims = list(range(1, target.dim()))

    dice = 2 * torch.sum(pred * target, dim=sum_dims) / torch.sum(pred ** 2 + target ** 2, dim=sum_dims)
    loss = 1 - dice

    return loss.mean()


def dice_loss_with_logits(logit: torch.Tensor, target: torch.Tensor):
    """
        References
        ----------
        `Dice Loss <https://arxiv.org/abs/1606.04797>`_
        """
    if not (target.size() == logit.size()):
        raise ValueError("Target size ({}) must be the same as logit size ({})".format(target.size(), logit.size()))
    pred = torch.sigmoid(logit)
    return dice_loss(pred, target)


# simply copied from np.moveaxis
def moveaxis(x: torch.Tensor, source: AxesLike, destination: AxesLike):
    """
    Move axes of a torch.Tensor to new positions.
    Other axes remain in their original order.
    """
    source = np.core.numeric.normalize_axis_tuple(source, x.ndim, 'source')
    destination = np.core.numeric.normalize_axis_tuple(destination, x.ndim, 'destination')
    if len(source) != len(destination):
        raise ValueError('`source` and `destination` arguments must have '
                         'the same number of elements')

    order = [n for n in range(x.ndim) if n not in source]
    for dest, src in sorted(zip(destination, source)):
        order.insert(dest, src)

    return x.permute(*order)


def softmax(x: torch.Tensor, axes: AxesLike):
    """
    A multidimensional version of softmax.
    """
    source = np.core.numeric.normalize_axis_tuple(axes, x.ndim, 'axes')
    dim = len(source)
    destination = range(-dim, 0)

    x = moveaxis(x, source, destination)
    shape = x.shape

    x = x.reshape(*shape[:-dim], -1)
    x = functional.softmax(x, -1).reshape(*shape)
    x = moveaxis(x, destination, source)
    return x
