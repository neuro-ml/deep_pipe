from typing import Callable

import numpy as np
import torch
from torch.nn import Module
from torch.optim import Optimizer

from ..im.utils import identity
from .utils import *

__all__ = 'optimizer_step', 'train_step', 'inference_step'


def optimizer_step(optimizer: Optimizer, loss: torch.Tensor, **params) -> torch.Tensor:
    """
    Performs the backward pass with respect to ``loss``, as well as a gradient step.

    ``params`` is used to change the optimizer's parameters.

    Examples
    --------
    >>> optimizer = Adam(model.parameters(), lr=1)
    >>> optimizer_step(optimizer, loss) # perform a gradient step
    >>> optimizer_step(optimizer, loss, lr=1e-3) # set lr to 1e-3 and perform a gradient step
    >>> optimizer_step(optimizer, loss, betas=(0, 0)) # set betas to 0 and perform a gradient step

    Notes
    -----
    The incoming ``optimizer``'s parameters are not restored to their original values.
    """
    set_params(optimizer, **params)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    return loss


def train_step(*inputs: np.ndarray, architecture: Module, criterion: Callable, optimizer: Optimizer,
               n_targets: int = 1, **optimizer_params) -> np.ndarray:
    """
    Performs a forward-backward pass, and make a gradient step, according to the given ``inputs``.

    Parameters
    ----------
    inputs
        inputs batches. The last ``n_targets`` batches are passed to ``criterion``.
        The remaining batches are fed into the ``architecture``.
    architecture
        the neural network architecture.
    criterion
        the loss function.
    optimizer
    n_targets
        how many values from ``inputs`` to be considered as targets.
    optimizer_params
        additional parameters that will override the optimizer's current parameters (e.g. lr).

    Notes
    -----
    Note that both input and output are **not** of type ``torch.Tensor`` - the conversion
    to and from ``torch.Tensor`` is made inside this function.

    References
    ----------
    `optimizer_step`
    """
    architecture.train()
    n_inputs = len(inputs) - n_targets  # in case n_targets == 0

    inputs = sequence_to_var(*inputs, device=architecture)
    inputs, targets = inputs[:n_inputs], inputs[n_inputs:]

    loss = criterion(architecture(*inputs), *targets)

    optimizer_step(optimizer, loss, **optimizer_params)
    return to_np(loss)


@torch.no_grad()
def inference_step(*inputs: np.ndarray, architecture: Module, activation: Callable = identity) -> np.ndarray:
    """
    Returns the prediction for the given ``inputs``.

    Notes
    -----
    Note that both input and output are **not** of type ``torch.Tensor`` - the conversion
    to and from ``torch.Tensor`` is made inside this function.
    """
    architecture.eval()
    return to_np(activation(architecture(*sequence_to_var(*inputs, device=architecture))))


@np.deprecate
def do_train_step(*inputs, lr, inputs2logits, optimizer, logits2loss):
    return train_step(*inputs, lr=lr, architecture=inputs2logits, criterion=logits2loss, optimizer=optimizer)


@np.deprecate
def do_inf_step(*inputs, inputs2logits, logits2pred):
    return inference_step(*inputs, architecture=inputs2logits, activation=logits2pred)
