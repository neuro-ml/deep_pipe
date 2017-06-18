"""
Functions necessary for working with pytorch.

"""

import numpy as np
import torch
from torch.autograd import Variable
import torch.nn.functional as F 


def _pred_reshape(y):
    """
    Reshape predictions wrt channels into vector (n_pixels*n_batch, n_classes):
    each channel represent probability map
    for corresponding class.
    """
    x = y.permute(0, 2, 3, 4, 1)
    return x.contiguous().view(-1, x.size()[-1])


def loss_cross_entropy(y_pred, y_true):
    """Log.loss with cropping for predicted shape."""
    true_shape = y_true.size()
    pred_shape = y_pred.size()
    shape_diff = np.array(true_shape)[-3:] - np.array(pred_shape)[-3:]
    if not np.all(shape_diff == 0):
        slices = [slice(i // 2, -(i // 2 + i % 2)) for i in shape_diff[-3:]]
        y_true = y_true[..., slices[0], slices[1], slices[2]].contiguous()
    return F.cross_entropy(_pred_reshape(y_pred), y_true.view(-1))


def to_var(x, volatile=False):
    return Variable(torch.from_numpy(x), volatile=volatile).cuda()


def to_numpy(x):
    return np.array(x.cpu().data.numpy())


def stochastic_step(x, y_true, model, optimizer, train=True):
    """
    Makes step with grad descent (if need)
    ------
    Input:
    x: numpy array,
    y_true; numpy array,
    model: torch.nn.Module,
    optimizer: torch...
    train: bool,
        Determines whether it is necessary to take a step.
    -----
    Return:
    batch_loss_val: float
        CE Loss on this data.
    """
    x = to_var(np.array(x)).cuda()
    y_true = to_var(np.array(y_true)).cuda()
    y_pred = model(x)
    batch_loss = loss_cross_entropy(y_pred, y_true).cpu()
    if train:
        optimizer.zero_grad()
        batch_loss.backward()
        optimizer.step()
    del x, y_true, y_pred
    batch_loss_val = batch_loss.data[0]
    return batch_loss_val