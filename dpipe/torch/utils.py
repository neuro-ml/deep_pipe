import warnings
from pathlib import Path
from typing import Callable, Union

import numpy as np
import torch
from torch import nn
from torch.optim import Optimizer

from dpipe.medim.io import PathLike

Device = Union[torch.device, nn.Module, torch.Tensor, str]


def load_model_state(module: nn.Module, path: PathLike, modify_state_fn: Callable = None) -> nn.Module:
    """
    Updates the ``module``'s state dict by the one located at ``path``.

    Parameters
    ----------
    module
    path
    modify_state_fn: Callable(current_state, loaded_state)
        if not ``None``, two arguments will be passed to the function:
        current state of the model and the state loaded from the path.
        This function should modify states as needed and return the final state to load.
        For example, it could help you to transfer weights from similar but not completely equal architecture.
    """
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


def save_model_state(module: nn.Module, path: PathLike):
    """Saves the ``module``'s state dict to ``path``."""
    path = Path(path)
    path.parent.mkdir(exist_ok=True, parents=True)
    torch.save(module.state_dict(), path)


def get_device(x: Device = None) -> torch.device:
    """
    Determines the correct device based on the input.

    If ``x`` is None ``torch.cuda.is_available()`` is used to determine whether the device will be CPU or CUDA.
    """
    if isinstance(x, nn.Module):
        return next(x.parameters()).device
    if isinstance(x, torch.Tensor):
        return x.device

    if x is None:
        if torch.cuda.is_available():
            x = 'cuda'
        else:
            x = 'cpu'

    return torch.device(x)


def is_on_cuda(x: Union[nn.Module, torch.Tensor]):
    if isinstance(x, nn.Module):
        x = next(x.parameters())

    return x.is_cuda


def to_var(x: np.ndarray, device: Device = None, requires_grad: bool = False, cuda: bool = None) -> torch.Tensor:
    """
    Convert a numpy array to a torch Tensor

    Parameters
    ----------
    x
    device
        the device on which to move ``x``. If None  -``torch.cuda.is_available()`` is used
        to determine whether the device will be CPU or CUDA.
    requires_grad: bool, optional
    cuda
        whether to move tensor to cuda. If None, torch.cuda.is_available() is used to determine that.
        This argument is deprecated.
    """
    # TODO: legacy support for `cuda` argument. 13.08.2019
    if device is not None and cuda is not None:
        raise ValueError('Cannot pass both `device` and `cuda`.')
    if cuda is not None:
        warnings.warn('`cuda` is deprecated. Use `device` instead.', DeprecationWarning)
        device = cuda
    if isinstance(device, bool):
        warnings.warn('`device` cannot be of type `bool`.', DeprecationWarning)
        device = 'cuda' if device else 'cpu'

    x = torch.from_numpy(np.asarray(x))
    if requires_grad:
        x.requires_grad_()
    return to_device(x, device)


def to_np(x: torch.Tensor) -> np.ndarray:
    """Convert a torch.Tensor to a numpy array."""
    return x.data.cpu().numpy()


def sequence_to_var(*data, device: Device = None, requires_grad: bool = False, cuda: bool = None):
    return tuple(to_var(x, cuda=cuda, requires_grad=requires_grad, device=device) for x in data)


def sequence_to_np(*data):
    return tuple(to_np(x) if isinstance(x, torch.Tensor) else x for x in data)


def to_device(x: Union[nn.Module, torch.Tensor], device: Device = None):
    """
    Move ``x`` to ``device``.

    Parameters
    ----------
    x
    device
        the device on which to move ``x``. If None  -``torch.cuda.is_available()`` is used
        to determine whether the device will be CPU or CUDA.
    """
    return x.to(device=get_device(device))


def to_cuda(x, cuda: Union[nn.Module, torch.Tensor, bool] = None):
    """
    Move ``x`` to cuda if specified.

    Parameters
    ----------
    x
    cuda
        whether to move to cuda. If None, torch.cuda.is_available() is used to determine that.
    """
    if isinstance(cuda, (nn.Module, torch.Tensor)):
        cuda = is_on_cuda(cuda)
    if cuda or (cuda is None and torch.cuda.is_available()):
        x = x.cuda()
    return x


def set_lr(optimizer: Optimizer, lr: float) -> Optimizer:
    """Change an ``optimizer``'s learning rate to ``lr``."""
    return set_params(optimizer, lr=lr)


def set_params(optimizer: Optimizer, **params) -> Optimizer:
    """Change an ``optimizer``'s parameters by the ones passed in ``params``."""
    for param_group in optimizer.param_groups:
        for name, value in params.items():
            param_group[name] = value
    return optimizer
