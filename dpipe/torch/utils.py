from abc import ABCMeta

import numpy as np
import torch
import torch.nn as nn
from torch.nn import functional
from torch.autograd import Variable
from torch.nn.modules.loss import _Loss


def compress_3d_to_2d(x):
    return x.view(*x.size()[:-2], -1)


def softmax_cross_entropy(logits, target, weight=None, reduce=True):
    """Softmax cross entropy loss for Nd data."""
    target = target.long()
    # Convert 5d input to 4d, because it is faster in functional.cross_entropy
    if logits.dim() == 5:
        logits = compress_3d_to_2d(logits)
        target = compress_3d_to_2d(target)

    return nn.functional.cross_entropy(logits, target, weight=weight, reduce=reduce)


class Eye(nn.Module, metaclass=ABCMeta):
    def __init__(self, n_classes):
        super().__init__()
        self.n_classes = n_classes
        self.register_buffer('eye', Variable(torch.eye(n_classes)))


class NormalizedSoftmaxCrossEntropy(Eye):
    def forward(self, logits, target):
        # This is dangerous, as any force cast. Contact with Egor before using it.
        target = target.long()
        flat_target = target.view(-1)
        count = self.eye.index_select(0, flat_target).sum(0)
        weight = flat_target.size()[0] / count

        return softmax_cross_entropy(logits, target, weight=weight)


class PyramidPooling(nn.Module):
    def __init__(self, pooling, levels: int = 1):
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


class LinearFocalLoss(_Loss):
    def __init__(self, gamma, beta, size_average=True):
        super().__init__(size_average)
        self.beta = beta
        self.gamma = gamma

    def forward(self, logits, target):
        return functional.binary_cross_entropy_with_logits(
            self.gamma * logits + self.beta, target, size_average=self.size_average
        ) / self.gamma


def focal_loss_with_logits(logits, target, gamma=2, weight=None, size_average=True, reduce=True):
    """Function that measures Focal Loss between target and output
    logits.

    Args:
        logits: Variable of arbitrary shape
        target: Variable of the same shape as input
        gamma (float): The power of focal loss factor
        weight (Variable, optional): a manual rescaling weight
                if provided it's repeated to match input tensor shape
        size_average (bool, optional): By default, the losses are averaged
                over observations for each minibatch. However, if the field
                :attr:`size_average` is set to ``False``, the losses are instead summed
                for each minibatch. Default: ``True``
        reduce (bool, optional): By default, the losses are averaged or summed over
                observations for each minibatch depending on :attr:`size_average`. When :attr:`reduce`
                is ``False``, returns a loss per logits/target element instead and ignores
                :attr:`size_average`. Default: ``True``
    """
    if not (target.size() == logits.size()):
        raise ValueError("Target size ({}) must be the same as logits size ({})".format(target.size(), logits.size()))

    min_val = - logits.clamp(min=0)
    max_val = (-logits).clamp(min=0)

    prob = (min_val + logits).exp() / (min_val.exp() + (min_val + logits).exp())

    loss = ((1 - 2 * prob) * target + prob) ** gamma \
           * (logits - logits * target + max_val + ((-max_val).exp() + (-logits - max_val).exp()).log())
    # * F.binary_cross_entropy_with_logits(logits, target, reduce=False)

    if weight is not None:
        loss = loss * weight

    if not reduce:
        return loss
    elif size_average:
        return loss.mean()
    else:
        return loss.sum()
