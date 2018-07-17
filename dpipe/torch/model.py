import os
from typing import Callable
from functools import partial

import numpy as np
import torch
from torch.nn import Module

from dpipe.model import Model, FrozenModel, get_model_path


def load_model_state(module: torch.nn.Module, path: str, modify_state_fn: Callable = None):
    if is_on_cuda(module):
        map_location = None
    else:
        # load models that were trained on GPU nodes, but now run on CPU
        def map_location(storage, location):
            return storage

    state_to_load = torch.load(path, map_location=map_location)
    if modify_state_fn is not None:
        current_state = module.state_dict()
        state_to_load = modify_state_fn(current_state, state_to_load)
    module.load_state_dict(state_to_load)
    return module


def save_model_state(module: torch.nn.Module, path: str):
    # Legacy to load models from old experiments
    folder = os.path.dirname(path)
    if folder:
        os.makedirs(folder, exist_ok=True)
    torch.save(module.state_dict(), path)


def is_on_cuda(module: torch.nn.Module):
    return next(module.parameters()).is_cuda


def sequence_to_var(*data, cuda: bool = None, requires_grad: bool = False):
    return tuple(to_var(x, cuda, requires_grad) for x in data)


def sequence_to_np(*data):
    return tuple(to_np(x) if isinstance(x, torch.Tensor) else x for x in data)


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
        # TODO: need a way to pass other counters: write a wrapper
        return do_train_step(*inputs, lr=lr, inputs2logits=self.model_core,
                             logits2loss=self.logits2loss, optimizer=self.optimizer)

    def do_val_step(self, *inputs):
        return do_val_step(*inputs, inputs2logits=self.model_core, logits2loss=self.logits2loss,
                           logits2pred=self.logits2pred, )

    def do_inf_step(self, *inputs):
        return do_inf_step(*inputs, inputs2logits=self.model_core, logits2pred=self.logits2pred)

    def save(self, path: str):
        save_model_state(self.model_core, path)

    def load(self, path: str, modify_state_fn: callable = None):
        load_model_state(self.model_core, get_model_path(path), modify_state_fn=modify_state_fn)


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
        self.f = make_do_inf_step(model_core, logits2pred=logits2pred, cuda=cuda, modify_state_fn=modify_state_fn,
                                  saved_model_path=get_model_path(restore_model_path))

    def do_inf_step(self, *inputs):
        return self.f(*inputs)


def to_np(x: torch.Tensor) -> np.ndarray:
    """Convert a torch.Tensor to a numpy array."""
    return x.data.cpu().numpy()


def to_var(x: np.ndarray, cuda: bool = None, requires_grad: bool = False) -> torch.Tensor:
    """
    Convert a numpy array to a torch Tensor

    Parameters
    ----------
    x: np.ndarray
        the input tensor
    cuda: bool, optional
        move tensor to cuda. If None, torch.cuda.is_available() is used to determine that.
    requires_grad: bool, optional
    """
    x = torch.from_numpy(np.asarray(x))
    if requires_grad:
        x.requires_grad_()
    return to_cuda(x, cuda)


def to_cuda(x, cuda: bool = None):
    if cuda or (cuda is None and torch.cuda.is_available()):
        x = x.cuda()
    return x


def set_lr(optimizer: torch.optim.Optimizer, lr: float) -> torch.optim.Optimizer:
    """Change an optimizer's learning rate."""
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
    return optimizer
