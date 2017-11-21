import torch
import torch.nn as nn
from torch.autograd import Variable

from dpipe.config import register


@register()
def softmax(x):
    e = torch.exp(x)
    return e / e.sum(dim=1, keepdim=True)


def swap_channels(x):
    size = x.size()
    if len(size) > 2:
        x = x.permute(0, *range(2, len(size)), 1)
    return x.contiguous().view(-1, size[-1])


@register()
def bce_logits(logits, target):
    logits = nn.functional.sigmoid(logits)
    return nn.functional.binary_cross_entropy(logits, target)


@register('softmax_cross_entropy')
class SoftmaxCrossEntropy:
    def __init__(self, n_classes):
        self.n_classes = n_classes
        self.eye = Variable(torch.eye(n_classes))

    def __call__(self, logits, target):
        return torch.sum(self.eye.index_select(0, target.long().view(-1)) * swap_channels(logits).view(-1, self.n_classes))

    def cuda(self):
        self.eye = self.eye.cuda()
        return self

    def cpu(self):
        self.eye = self.eye.cpu()
        return self


@register('normalized_softmax_cross_entropy')
class NormalizedSoftmaxCrossEntropy(SoftmaxCrossEntropy):
    def __call__(self, logits, target):
        target = target.long().view(-1)
        weight = 1 / self.eye.index_select(0, target).sum(0)

        return nn.functional.cross_entropy(swap_channels(logits).view(-1, self.n_classes),
                                           target, weight=weight)
