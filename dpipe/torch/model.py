import os

import numpy as np
import torch
from torch.nn import Module
from torch.autograd import Variable

from dpipe.model import Model, FrozenModel, get_model_path


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
        inputs = [to_var(x, self.cuda) for x in inputs]
        *inputs, target = inputs

        logits = self.model_core(*inputs)
        loss = self.logits2loss(logits, target)

        set_lr(self.optimizer, lr)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        return to_np(loss)[0]

    def do_val_step(self, *inputs):
        self.model_core.eval()
        inputs = [to_var(x, self.cuda, volatile=True) for x in inputs]
        *inputs, target = inputs

        logits = self.model_core(*inputs)
        y_pred = self.logits2pred(logits)
        loss = self.logits2loss(logits, target)

        return to_np(y_pred), to_np(loss)[0]

    def do_inf_step(self, *inputs):
        self.model_core.eval()
        inputs = [to_var(x, self.cuda, volatile=True) for x in inputs]

        logits = self.model_core(*inputs)
        y_pred = self.logits2pred(logits)

        return to_np(y_pred)

    def save(self, path: str):
        os.makedirs(path, exist_ok=True)
        path = get_model_path(path)
        state_dict = self.model_core.state_dict()
        torch.save(state_dict, path)

    def load(self, path: str, modify_state_fn: callable = None):
        path = get_model_path(path)
        state_to_load = torch.load(path)
        if modify_state_fn is not None:
            current_state = self.model_core.state_dict()
            state_to_load = modify_state_fn(current_state, state_to_load)
        self.model_core.load_state_dict(state_to_load)


class TorchFrozenModel(FrozenModel):
    def __init__(self, model_core: torch.nn.Module, logits2pred: callable, restore_model_path: str, cuda=True):
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
            map_location = None
        else:
            map_location = lambda storage, location: storage
        self.cuda = cuda
        self.model_core = model_core
        self.logits2pred = logits2pred

        path = get_model_path(restore_model_path)
        self.model_core.load_state_dict(torch.load(path, map_location=map_location))

    def do_inf_step(self, *inputs):
        self.model_core.eval()
        inputs = [to_var(x, self.cuda, volatile=True) for x in inputs]

        logits = self.model_core(*inputs)
        y_pred = self.logits2pred(logits)

        return to_np(y_pred)


def to_np(x: Variable):
    """
    Convert a autograd Variable to a numpy array.

    Parameters
    ----------
    x: Variable
    """
    return x.data.cpu().numpy()


def to_var(x: np.ndarray, cuda: bool, volatile: bool = False):
    """
    Convert a numpy array to a torch Tensor

    Parameters
    ----------
    x: np.array
        the input tensor
    cuda: bool
        move tensor to cuda
    volatile: bool, optional
        make tensor volatile
    """
    x = Variable(torch.from_numpy(np.asarray(x)), volatile=volatile)
    if (torch.cuda.is_available() and cuda is None) or cuda:
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
