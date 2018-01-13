from abc import ABCMeta

import torch
import torch.nn as nn
from torch.autograd import Variable


def compress_to_2d(x):
    return x.view(*x.size()[:-2], -1)


def softmax_cross_entropy(logits, target, weight=None, reduce=True):
    """Softmax cross entropy loss for Nd data."""
    # target = target.long()
    if target.dim() > 4:
        logits = compress_to_2d(logits)
        target = compress_to_2d(target)

    return nn.functional.cross_entropy(compress_to_2d(logits), compress_to_2d(target), weight=weight, reduce=reduce)


class Eye(nn.Module, metaclass=ABCMeta):
    def __init__(self, n_classes):
        super().__init__()
        self.n_classes = n_classes
        self.register_buffer('eye', Variable(torch.eye(n_classes)))


class NormalizedSoftmaxCrossEntropy(Eye):
    def forward(self, logits, target):
        # target = target.long()
        flat_target = target.view(-1)
        count = self.eye.index_select(0, flat_target).sum(0)
        weight = flat_target.size()[0] / count

        return softmax_cross_entropy(logits, target, weight=weight)
