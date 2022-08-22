from pathlib import Path
from typing import Callable, Union, Iterable, Iterator

import numpy as np
import torch
from torch import nn
from torch.optim import Optimizer
from torch.nn.parameter import Parameter
from torch.nn.modules.batchnorm import _BatchNorm

from dpipe.io import PathLike
from dpipe.itertools import squeeze_first, collect

__all__ = [
    'load_model_state', 'save_model_state',
    'get_device', 'to_device', 'is_on_cuda', 'to_cuda',
    'to_var', 'sequence_to_var', 'to_np', 'sequence_to_np',
    'set_params', 'set_lr', 'get_parameters', 'has_batchnorm',
    'order_to_mode',
]

Device = Union[torch.device, nn.Module, torch.Tensor, str]
ArrayLike = Union[np.ndarray, Iterable, int, float]


def load_model_state(module: nn.Module, path: PathLike, modify_state_fn: Callable = None, strict: bool = True):
    """
    Updates the ``module``'s state dict by the one located at ``path``.

    Parameters
    ----------
    module: nn.Module
    path: PathLike
    modify_state_fn: Callable(current_state, state_to_load)
        if not ``None``, two arguments will be passed to the function:
        current state of the model and the state loaded from the path.
        This function should modify states as needed and return the final state to load.
        For example, it could help you to transfer weights from similar but not completely equal architecture.
    strict: bool
    """
    state_to_load = torch.load(path, map_location=get_device(module))
    if modify_state_fn is not None:
        current_state = module.state_dict()
        state_to_load = modify_state_fn(current_state, state_to_load)
    module.load_state_dict(state_to_load, strict=strict)


def save_model_state(module: nn.Module, path: PathLike):
    """Saves the ``module``'s state dict to ``path``."""
    path = Path(path)
    path.parent.mkdir(exist_ok=True, parents=True)
    torch.save(module.state_dict(), path)


def get_device(x: Device = None) -> torch.device:
    """
    Determines the correct device based on the input.

    Parameters
    ----------
    x: torch.device, torch.nn.Module, torch.Tensor, str, None
        | if ``torch.Tensor`` - returns the device on which it is located
        | if ``torch.nn.Module`` - returns the device on which its parameters are located
        | if ``str`` or ``torch.device`` - returns `torch.device(x)`
        | if ``None`` - same as 'cuda' if CUDA is available, 'cpu' otherwise.
    """
    if isinstance(x, nn.Module):
        try:
            return next(x.parameters()).device
        except StopIteration:
            raise ValueError('The device could not be determined as the passed model has no parameters.') from None
    if isinstance(x, torch.Tensor):
        return x.device

    if x is None:
        x = 'cuda' if torch.cuda.is_available() else 'cpu'

    return torch.device(x)


def is_on_cuda(x: Union[nn.Module, torch.Tensor]):
    if isinstance(x, nn.Module):
        x = next(x.parameters())

    return x.is_cuda


def to_var(*arrays: ArrayLike, device: Device = 'cpu', requires_grad: bool = False):
    """
    Convert numpy arrays to torch Tensors.

    Parameters
    ----------
    arrays: array-like
        objects, that will be converted to torch Tensors.
    device
        the device on which to move ``x``. See `get_device` for details.
    requires_grad
        whether the tensors require grad.

    Notes
    -----
    If ``arrays`` contains a single argument the result will not be contained in a tuple:
    >>> x = to_var(x)
    >>> x, y = to_var(x, y)

    If this is not the desired behaviour, use `sequence_to_var`, which always returns a tuple of tensors.
    """
    return squeeze_first(tuple(sequence_to_var(*arrays, device=device, requires_grad=requires_grad)))


def to_np(*tensors: torch.Tensor):
    """
    Convert torch Tensors to numpy arrays.

    Notes
    -----
    If ``tensors`` contains a single argument the result will not be contained in a tuple:
    >>> x = to_np(x)
    >>> x, y = to_np(x, y)

    If this is not the desired behaviour, use `sequence_to_np`, which always returns a tuple of arrays.
    """
    return squeeze_first(tuple(sequence_to_np(*tensors)))


@collect
def sequence_to_var(*arrays: ArrayLike, device: Device = 'cpu', requires_grad: bool = False):
    for x in arrays:
        x = torch.from_numpy(np.asarray(x))
        if requires_grad:
            x.requires_grad_()
        yield to_device(x, device)


@collect
def sequence_to_np(*tensors: torch.Tensor):
    for x in tensors:
        yield x.data.cpu().numpy()


def to_device(x: Union[nn.Module, torch.Tensor], device: Union[Device, None] = 'cpu'):
    """
    Move ``x`` to ``device``.

    Parameters
    ----------
    x
    device
        the device on which to move ``x``. See `get_device` for details.
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
    for name, value in params.items():
        updated = False
        for param_group in optimizer.param_groups:
            if name in param_group:
                param_group[name] = value
                updated = True

        if not updated:
            raise ValueError(f"The optimizer doesn't have a parameter named `{name}`.")

    return optimizer


def order_to_mode(order: int, dim: int):
    """
    Converts the order of interpolation to a "mode" string.

    Examples
    --------
    >>> order_to_mode(1, 3)
    'trilinear'
    """
    if order == 0:
        return 'nearest'

    mapping = {
        (1, 1): 'linear',
        (1, 2): 'bilinear',
        (1, 3): 'trilinear',
        (3, 2): 'bicubic',
    }
    if (order, dim) not in mapping:
        raise ValueError(f'Invalid order of interpolation passed ({order}) for dim={dim}.')

    return mapping[order, dim]


def get_parameters(optimizer: Optimizer) -> Iterator[Parameter]:
    """Returns an iterator over model parameters stored in ``optimizer``."""
    for group in optimizer.param_groups:
        for param in group['params']:
            yield param


def has_batchnorm(architecture: nn.Module) -> bool:
    """Check whether ``architecture`` has BatchNorm module"""
    for module in architecture.modules():
        if isinstance(module, _BatchNorm):
            return True

    return False
