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


class LinearFocalLoss(_Loss):
    def __init__(self, gamma, beta, size_average=True):
        super().__init__(size_average)
        self.beta = beta
        self.gamma = gamma

    def forward(self, logits, target):
        return functional.binary_cross_entropy_with_logits(
            self.gamma * logits + self.beta, target, size_average=self.size_average
        ) / self.gamma


def focal_weight(probs, target, gamma):
    weight = (1 - 2 * probs) * target + probs
    return weight ** gamma


# TODO: unify
class FocalLossWithLogits(_Loss):
    def __init__(self, gamma, size_average=True):
        super().__init__(size_average)
        self.gamma = gamma

    def forward(self, logits, target):
        bce = functional.binary_cross_entropy_with_logits(logits, target, size_average=False)
        loss = focal_weight(torch.sigmoid(logits), target, self.gamma) * bce

        if self.size_average:
            return loss.mean()
        return loss.sum()


class FocalLoss(_Loss):
    def __init__(self, gamma, size_average=True):
        super().__init__(size_average)
        self.gamma = gamma

    def forward(self, input, target):
        bce = functional.binary_cross_entropy(input, target, size_average=False)
        loss = focal_weight(input, target, self.gamma) * bce

        if self.size_average:
            return loss.mean()
        return loss.sum()
