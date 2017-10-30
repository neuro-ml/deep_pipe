import torch
import torch.nn as nn

from dpipe.config import register


@register(module_type='torch')
def softmax(x):
    shape = x.size()
    e = torch.exp(x)
    return e / e.sum(dim=1, keepdim=True)


def swap_channels(x):
    x_new = x.permute(0, 2, 3, 4, 1)
    return x_new.contiguous().view(-1, x.size()[-1])


@register(module_type='torch')
def softmax_cross_entropy(logits, target):
    return nn.functional.cross_entropy(swap_channels(logits).view(-1, logits.size()[1]), target.view(-1))