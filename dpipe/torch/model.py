from functools import partial

import torch
from torch.nn import Module

from dpipe.model import Model, FrozenModel
from .utils import *


def do_train_step(*inputs, lr, inputs2logits, optimizer, logits2loss):
    inputs2logits.train()
    *inputs, target = sequence_to_var(*inputs, cuda=is_on_cuda(inputs2logits))

    logits = inputs2logits(*inputs)
    loss = logits2loss(logits, target)

    set_lr(optimizer, lr)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    return to_np(loss)


def do_inf_step(*inputs, inputs2logits, logits2pred):
    inputs2logits.eval()
    with torch.no_grad():
        return to_np(logits2pred(inputs2logits(*sequence_to_var(*inputs, cuda=is_on_cuda(inputs2logits)))))


def do_val_step(*inputs, inputs2logits, logits2loss, logits2pred):
    inputs2logits.eval()
    *inputs, target = sequence_to_var(*inputs, cuda=is_on_cuda(inputs2logits))

    with torch.no_grad():
        logits = inputs2logits(*inputs)
        y_pred = logits2pred(logits)
        loss = logits2loss(logits, target)

        return sequence_to_np(y_pred, loss)


class TorchModel(Model):
    """
    `Model` interface implementation for the PyTorch framework.

    Parameters
    ----------
    model_core: torch.nn.Module
        torch model structure
    logits2pred: callable(logits) -> prediction
        last layer nonlinearity that maps logits to predictions
    logits2loss: callable(logits) -> loss
        the loss function
    optimize: torch.optim.Optimizer
        the optimizer
    cuda: bool, optional
        whether to move the model's parameters to CUDA
    """

    def __init__(self, model_core: torch.nn.Module, logits2pred: Callable, logits2loss: Callable,
                 optimize: torch.optim.Optimizer, cuda: bool = None):
        if isinstance(logits2loss, Module):
            to_cuda(logits2loss, cuda)

        self.cuda = cuda
        self.model_core = to_cuda(model_core, cuda)
        self.logits2pred = logits2pred
        self.logits2loss = logits2loss
        self.optimizer = optimize(model_core.parameters())

    def do_train_step(self, *inputs, lr):
        return do_train_step(*inputs, lr=lr, inputs2logits=self.model_core,
                             logits2loss=self.logits2loss, optimizer=self.optimizer)

    def do_val_step(self, *inputs):
        return do_val_step(*inputs, inputs2logits=self.model_core, logits2loss=self.logits2loss,
                           logits2pred=self.logits2pred)

    def do_inf_step(self, *inputs):
        return do_inf_step(*inputs, inputs2logits=self.model_core, logits2pred=self.logits2pred)

    def save(self, path: str):
        save_model_state(self.model_core, path)
        return self

    def load(self, path: str, modify_state_fn: callable = None):
        load_model_state(self.model_core, path, modify_state_fn=modify_state_fn)
        return self


def make_do_inf_step(inputs2logits, logits2pred, saved_model_path, cuda, modify_state_fn=None):
    inputs2logits = load_model_state(to_cuda(inputs2logits, cuda), path=saved_model_path,
                                     modify_state_fn=modify_state_fn)
    return partial(do_inf_step, inputs2logits=inputs2logits, logits2pred=logits2pred)


class TorchFrozenModel(FrozenModel):
    """
    Parameters
    ----------
    model_core: torch.nn.Module
        torch model structure
    logits2pred: callable(logits) -> prediction
        last layer nonlinearity that maps logits to predictions
    restore_model_path: str
        the path to the trained model
    cuda: bool, optional
        whether to move the model's parameters to CUDA
    """

    def __init__(self, model_core: torch.nn.Module, logits2pred: callable, restore_model_path: str,
                 cuda: bool = None, modify_state_fn: callable = None):
        self.model_core = to_cuda(model_core, cuda)
        self.logits2pred = logits2pred
        self.cuda = cuda
        self.f = make_do_inf_step(model_core, logits2pred=logits2pred, cuda=cuda, modify_state_fn=modify_state_fn,
                                  saved_model_path=restore_model_path)

    def do_inf_step(self, *inputs):
        return self.f(*inputs)
