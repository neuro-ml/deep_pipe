import os

import numpy as np
import torch
from torch.nn import Module

from dpipe.model import Model, FrozenModel, get_model_path


def load_model_state(model_core: torch.nn.Module, path: str, cuda: bool = True, modify_state_fn: callable = None):
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


def sequence_to_var(*data, cuda: bool = None, requires_grad: bool = True):
    return tuple(to_var(x, cuda, requires_grad) for x in data)


def sequence_to_np(*data):
    return tuple(to_np(x) if isinstance(x, torch.Tensor) else x for x in data)


class TorchModel(Model):
    """`Model` interface implementation for the PyTorch framework."""

    def __init__(self, model_core: torch.nn.Module, logits2pred: callable, logits2loss: callable,
                 optimize: torch.optim.Optimizer, cuda: bool = True):
        """
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
        if cuda:
            model_core.cuda()
            if isinstance(logits2loss, Module):
                logits2loss.cuda()

        self.cuda = cuda
        self.model_core = model_core
        self.logits2pred = logits2pred
        self.logits2loss = logits2loss
        self.optimizer = optimize(model_core.parameters())

    def do_train_step(self, *inputs, lr):
        self.model_core.train()
        *inputs, target = sequence_to_var(*inputs, cuda=self.cuda)

        logits = self.model_core(*inputs)
        loss = self.logits2loss(logits, target)

        set_lr(self.optimizer, lr)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        return to_np(loss)

    def do_val_step(self, *inputs):
        self.model_core.eval()
        *inputs, target = sequence_to_var(*inputs, cuda=self.cuda, requires_grad=False)

        with torch.no_grad():
            logits = self.model_core(*inputs)
            y_pred = self.logits2pred(logits)
            loss = self.logits2loss(logits, target)

        return to_np(y_pred), to_np(loss)

    def do_inf_step(self, *inputs):
        self.model_core.eval()
        inputs = [to_var(x, self.cuda, requires_grad=False) for x in inputs]

        with torch.no_grad():
            logits = self.model_core(*inputs)
            y_pred = self.logits2pred(logits)

        return to_np(y_pred)

    def save(self, path: str):
        os.makedirs(path, exist_ok=True)
        path = get_model_path(path)
        state_dict = self.model_core.state_dict()
        torch.save(state_dict, path)

    def load(self, path: str, modify_state_fn: callable = None):
        load_model_state(self.model_core, get_model_path(path), modify_state_fn=modify_state_fn, cuda=self.cuda)


class TorchFrozenModel(FrozenModel):
    def __init__(self, model_core: torch.nn.Module, logits2pred: callable, restore_model_path: str,
                 cuda: bool = True, modify_state_fn: callable = None):
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
        if cuda:
            model_core.cuda()
        self.model_core = load_model_state(model_core, get_model_path(restore_model_path),
                                           modify_state_fn=modify_state_fn, cuda=cuda)
        self.cuda = cuda
        self.logits2pred = logits2pred

    def do_inf_step(self, *inputs):
        self.model_core.eval()
        inputs = [to_var(x, self.cuda, requires_grad=False) for x in inputs]

        with torch.no_grad():
            logits = self.model_core(*inputs)
            y_pred = self.logits2pred(logits)

        return to_np(y_pred)


def to_np(x: torch.Tensor) -> np.ndarray:
    """
    Convert a torch.Tensor to a numpy array.

    Parameters
    ----------
    x: torch.Tensor
    """
    return x.data.cpu().numpy()


def to_var(x: np.ndarray, cuda: bool = None, requires_grad: bool = True) -> torch.Tensor:
    """
    Convert a numpy array to a torch Tensor

    Parameters
    ----------
    x: np.ndarray
        the input tensor
    cuda: bool
        move tensor to cuda
    requires_grad: bool, optional
    """
    x = torch.from_numpy(np.asarray(x))
    if requires_grad:
        x.requires_grad_()
    if cuda or (cuda is None and torch.cuda.is_available()):
        x = x.cuda()
    return x


def set_lr(optimizer: torch.optim.Optimizer, lr: float):
    """
    Change an optimizer's learning rate.

    Parameters
    ----------
    optimizer: torch.optim.Optimizer
    lr: float

    Returns
    -------
    optimizer: torch.optim.Optimizer
    """
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
    return optimizer
