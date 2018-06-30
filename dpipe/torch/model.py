import os
from typing import Callable

import numpy as np
import torch
from torch.nn import Module

from dpipe.model import Model, FrozenModel, get_model_path


def load_model_state(model_core: torch.nn.Module, path: str, cuda: bool = True, modify_state_fn: Callable = None):
    # To load models that were trained on GPU nodes, but now run on CPU
    if cuda:
        map_location = None
    else:
        def map_location(storage, location):
            return storage

    state_to_load = torch.load(path, map_location=map_location)
    if modify_state_fn is not None:
        current_state = model_core.state_dict()
        state_to_load = modify_state_fn(current_state, state_to_load)
    model_core.load_state_dict(state_to_load)
    return model_core


def save_model_state(model_core: torch.nn.Module, path: str):
    folder = os.path.dirname(path)
    if folder:
        os.makedirs(folder, exist_ok=True)
    torch.save(model_core.state_dict(), path)


def is_on_cuda(module: torch.nn.Module):
    return next(module.parameters()).is_cuda


def sequence_to_var(*data, cuda: bool = None, requires_grad: bool = False):
    return tuple(to_var(x, cuda, requires_grad) for x in data)


def sequence_to_np(*data):
    return tuple(to_np(x) if isinstance(x, torch.Tensor) else x for x in data)


def do_train_step(*inputs, lr, model_core, optimizer, logits2loss):
    model_core.train()
    *inputs, target = sequence_to_var(*inputs, cuda=is_on_cuda(model_core))

    logits = model_core(*inputs)
    loss = logits2loss(logits, target)

    set_lr(optimizer, lr)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    return to_np(loss)


def do_inf_step(*inputs, model_core, logits2pred):
    model_core.eval()
    with torch.no_grad():
        return to_np(
            logits2pred(model_core(*sequence_to_var(*inputs, cuda=is_on_cuda(model_core)))))


def do_val_step(*inputs, model_core, logits2loss, logits2pred):
    model_core.eval()
    *inputs, target = sequence_to_var(*inputs, cuda=is_on_cuda(model_core))

    with torch.no_grad():
        logits = model_core(*inputs)
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
                 optimize: torch.optim.Optimizer, cuda: bool = True):
        if isinstance(logits2loss, Module):
            to_cuda(logits2loss, cuda)

        self.cuda = cuda
        self.model_core = to_cuda(model_core, cuda)
        self.logits2pred = logits2pred
        self.logits2loss = logits2loss
        self.optimizer = optimize(model_core.parameters())

    def do_train_step(self, *inputs, lr):
        # TODO: need a way to pass other counters: write a wrapper
        return do_train_step(
            *inputs, lr=lr, model_core=self.model_core, logits2loss=self.logits2loss, optimizer=self.optimizer
        )

    def do_val_step(self, *inputs):
        return do_val_step(
            *inputs, model_core=self.model_core, logits2loss=self.logits2loss, logits2pred=self.logits2pred,
        )

    def do_inf_step(self, *inputs):
        return do_inf_step(*inputs, model_core=self.model_core, logits2pred=self.logits2pred)

    def save(self, path: str):
        save_model_state(self.model_core, path)

    def load(self, path: str, modify_state_fn: callable = None):
        load_model_state(self.model_core, get_model_path(path), modify_state_fn=modify_state_fn, cuda=self.cuda)


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
                 cuda: bool = True, modify_state_fn: callable = None):
        if cuda:
            model_core.cuda()
        self.model_core = load_model_state(model_core, get_model_path(restore_model_path),
                                           modify_state_fn=modify_state_fn, cuda=cuda)
        self.cuda = cuda
        self.logits2pred = logits2pred

    def do_inf_step(self, *inputs):
        return do_inf_step(*inputs, model_core=self.model_core, logits2pred=self.logits2pred)


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
