import warnings
from typing import Union, Callable

import numpy as np
import torch
from torch.nn import functional

from dpipe.im.axes import AxesLike

__all__ = [
    'focal_loss_with_logits', 'linear_focal_loss_with_logits', 'weighted_cross_entropy_with_logits',
    'tversky_loss', 'focal_tversky_loss', 'tversky_loss_with_logits', 'focal_tversky_loss_with_logits',
    'dice_loss', 'dice_loss_with_logits',
    'masked_loss', 'moveaxis', 'softmax',
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


def dice_loss(pred: torch.Tensor, target: torch.Tensor, epsilon=1e-7):
    """
    References
    ----------
    `Dice Loss <https://arxiv.org/abs/1606.04797>`_
    """
    if not (target.size() == pred.size()):
        raise ValueError("Target size ({}) must be the same as logit size ({})".format(target.size(), pred.size()))

    sum_dims = list(range(1, target.dim()))

    dice = 2 * torch.sum(pred * target, dim=sum_dims) / (torch.sum(pred ** 2 + target ** 2, dim=sum_dims) + epsilon)
    loss = 1 - dice

    return loss.mean()


def tversky_loss(pred: torch.Tensor, target: torch.Tensor, alpha=0.5, epsilon=1e-7,
                 reduce: Union[Callable, None] = torch.mean):
    """
    References
    ----------
    `Tversky Loss https://arxiv.org/abs/1706.05721`_
    """
    if not (target.size() == pred.size()):
        raise ValueError("Target size ({}) must be the same as logit size ({})".format(target.size(), pred.size()))

    if alpha < 0 or alpha > 1:
        raise ValueError("Invalid alpha value, expected to be in (0, 1) interval")

    sum_dims = list(range(1, target.dim()))
    beta = 1 - alpha

    intersection = pred*target
    fps, fns = pred*(1-target), (1-pred)*target

    numerator = torch.sum(intersection, dim=sum_dims)
    denumenator = torch.sum(intersection, dim=sum_dims) + alpha*torch.sum(fps, dim=sum_dims) + beta*torch.sum(fns, dim=sum_dims)
    tversky = numerator / (denumenator + epsilon)
    loss = 1 - tversky

    if reduce is not None:
        loss = reduce(loss)
    return loss


def focal_tversky_loss(pred: torch.Tensor, target: torch.Tensor, gamma=4/3, alpha=0.5, epsilon=1e-7):
    """
    References
    ----------
    `Focal Tversky Loss https://arxiv.org/abs/1810.07842`_
    """
    if gamma <= 1:
        warnings.warn("Gamma is <=1, to focus on less accurate predictions choose gamma > 1.")
    tl = tversky_loss(pred, target, alpha, epsilon, reduce=None)

    return torch.pow(tl, 1/gamma).mean()


def loss_with_logits(criterion: Callable, logit: torch.Tensor, target: torch.Tensor, **kwargs):
    if not (target.size() == logit.size()):
        raise ValueError("Target size ({}) must be the same as logit size ({})".format(target.size(), logit.size()))
    pred = torch.sigmoid(logit)

    return criterion(pred, target, **kwargs)


def dice_loss_with_logits(logit: torch.Tensor, target: torch.Tensor):
    return loss_with_logits(dice_loss, logit, target)


def tversky_loss_with_logits(logit: torch.Tensor, target: torch.Tensor, alpha=0.5):
    return loss_with_logits(tversky_loss, logit, target, alpha=alpha)


def focal_tversky_loss_with_logits(logit: torch.Tensor, target: torch.Tensor, gamma, alpha=0.5):
    return loss_with_logits(focal_tversky_loss, logit, target, gamma=gamma, alpha=alpha)


def masked_loss(mask: torch.Tensor, criterion: Callable, prediction: torch.Tensor, target: torch.Tensor, **kwargs):
    """
    Calculates the ``criterion`` between the masked ``prediction`` and ``target``.
    ``args`` and ``kwargs`` are passed to ``criterion`` as additional arguments.

    If the ``mask`` is empty - returns 0 wrapped in a torch tensor.
    """
    if not mask.any():
        return torch.tensor(0., requires_grad=True).to(prediction)

    return criterion(prediction[mask], target[mask], **kwargs)


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


def softmax(x: torch.Tensor, axis: AxesLike):
    """
    A multidimensional version of softmax.
    """
    source = np.core.numeric.normalize_axis_tuple(axis, x.ndim, 'axis')
    dim = len(source)
    destination = range(-dim, 0)

    x = moveaxis(x, source, destination)
    shape = x.shape

    x = x.reshape(*shape[:-dim], -1)
    x = functional.softmax(x, -1).reshape(*shape)
    x = moveaxis(x, destination, source)
    return x
