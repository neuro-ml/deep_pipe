import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable


def pred_reshape(y):
    x = y.permute(0, 2, 3, 4, 1)
    return x.contiguous().view(-1, x.size()[-1])


def loss_cross_entropy(y_pred, y_true):
    return F.cross_entropy(pred_reshape(y_pred), y_true.view(-1))


def to_var(x, volatile=False):
    return Variable(torch.from_numpy(x), volatile=volatile).cuda()


def to_numpy(x):
    return np.array(x.cpu().data.numpy())
