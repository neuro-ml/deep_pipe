from typing import Callable

import numpy as np
import torch

from dpipe.medim.utils import makedirs_top


def load_model_state(module: torch.nn.Module, path: str, modify_state_fn: Callable = None):
    if is_on_cuda(module):
        map_location = None
    else:
        # load models that were trained on GPU, but now run on CPU
        def map_location(storage, location):
            return storage

    state_to_load = torch.load(path, map_location=map_location)
    if modify_state_fn is not None:
        current_state = module.state_dict()
        state_to_load = modify_state_fn(current_state, state_to_load)
    module.load_state_dict(state_to_load)
    return module


def save_model_state(module: torch.nn.Module, path: str):
    makedirs_top(path, exist_ok=True)
    torch.save(module.state_dict(), path)


def is_on_cuda(module: torch.nn.Module):
    return next(module.parameters()).is_cuda


def sequence_to_var(*data, cuda: bool = None, requires_grad: bool = False):
    return tuple(to_var(x, cuda, requires_grad) for x in data)


def sequence_to_np(*data):
    return tuple(to_np(x) if isinstance(x, torch.Tensor) else x for x in data)


def to_np(x: torch.Tensor) -> np.ndarray:
    """Convert a torch.Tensor to a numpy array."""
    return x.data.cpu().numpy()


def to_var(x: np.ndarray, cuda: bool = None, requires_grad: bool = False) -> torch.Tensor:
    """
    Convert a numpy array to a torch Tensor

    Parameters
    ----------
    x
    cuda
        whether to move tensor to cuda. If None, torch.cuda.is_available() is used to determine that.
    requires_grad: bool, optional
    """
    x = torch.from_numpy(np.asarray(x))
    if requires_grad:
        x.requires_grad_()
    return to_cuda(x, cuda)


def to_cuda(x, cuda: bool = None):
    """
    Move ``x`` to cuda if specified.

    Parameters
    ----------
    x
    cuda
        whether to move to cuda. If None, torch.cuda.is_available() is used to determine that.
    """
    if cuda or (cuda is None and torch.cuda.is_available()):
        x = x.cuda()
    return x


def set_lr(optimizer: torch.optim.Optimizer, lr: float) -> torch.optim.Optimizer:
    """Change an ``optimizer``'s learning rate to `lr`."""
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
    return optimizer
