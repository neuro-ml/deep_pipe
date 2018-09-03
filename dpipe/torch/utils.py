from numpy import deprecate
import torch
from torch import nn
from torch.nn import functional
from torch.autograd import Variable


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


class Eye(nn.Module):
    def __init__(self, n_classes):
        super().__init__()
        self.n_classes = n_classes
        self.register_buffer('eye', Variable(torch.eye(n_classes)))


@deprecate
class NormalizedSoftmaxCrossEntropy(Eye):
    def forward(self, logits, target):
        # This is dangerous, as any force cast. Contact with Egor before using it.
        target = target.long()
        flat_target = target.view(-1)
        count = self.eye.index_select(0, flat_target).sum(0)
        weight = flat_target.size()[0] / count

        return softmax_cross_entropy(logits, target, weight=weight)


Eye = deprecate(Eye)
