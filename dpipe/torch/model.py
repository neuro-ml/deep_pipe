from torch.nn import Module
from torch.optim import Optimizer

from ..medim.utils import identity
from ..model import Model
from .utils import *


def optimizer_step(optimizer: Optimizer, loss: torch.Tensor, lr: float = None) -> torch.Tensor:
    """
    Performs the backward pass with respect to ``loss``, as well as a gradient step.

    If ``loss`` is not None - the ``optimizer``'s learning rate will be changed to this value.
    """
    if lr is not None:
        set_lr(optimizer, lr)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    return loss


def train_step(*inputs: np.ndarray, architecture: nn.Module, criterion: Callable, optimizer: Optimizer,
               lr: float = None, n_targets: int = 1) -> np.ndarray:
    """
    Performs a forward-backward pass, as well as the gradient step, according to the given ``inputs``.

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
    lr
        if not None - the ``optimizer``'s learning rate will be changed to this value.
        Useful for non-constant learning rate policies.
    n_targets
        how many values from ``inputs`` to be considered as targets.

    Notes
    -----
    Note that both input and output are **not** of type ``torch.Tensor`` - the conversion
    to and from ``torch.Tensor`` is made inside this function.
    """
    architecture.train()
    n_inputs = len(inputs) - n_targets  # in case n_targets == 0

    inputs = sequence_to_var(*inputs, device=architecture)
    inputs, targets = inputs[:n_inputs], inputs[n_inputs:]

    loss = criterion(architecture(*inputs), *targets)

    optimizer_step(optimizer, loss, lr)
    return to_np(loss)


def inference_step(*inputs: np.ndarray, architecture: nn.Module, activation: Callable = identity) -> np.ndarray:
    """
    Returns the prediction for the given ``inputs``.

    Notes
    -----
    Note that both input and output are **not** of type ``torch.Tensor`` - the conversion
    to and from ``torch.Tensor`` is made inside this function.
    """
    architecture.eval()
    with torch.no_grad():
        return to_np(activation(architecture(*sequence_to_var(*inputs, device=architecture))))


def do_train_step(*inputs, lr, inputs2logits, optimizer, logits2loss):
    return train_step(*inputs, lr=lr, architecture=inputs2logits, criterion=logits2loss, optimizer=optimizer)


def do_inf_step(*inputs, inputs2logits, logits2pred):
    return inference_step(*inputs, architecture=inputs2logits, activation=logits2pred)


class TorchModel(Model):
    """
    `Model` interface implementation for the PyTorch framework.

    Parameters
    ----------
    model_core
        the model architecture
    logits2pred: callable(logits) -> prediction
        last layer nonlinearity that maps logits to predictions
    logits2loss: callable(logits) -> loss
        the loss function
    optimize: torch.optim.Optimizer
        the optimizer
    cuda: bool, optional
        whether to move the model's parameters to CUDA

    Warnings
    --------
    The `Model` interface is deprecated. Use `train_step` and `inference_step` instead.
    """

    def __init__(self, model_core: torch.nn.Module, logits2pred: Callable, logits2loss: Callable,
                 optimize: torch.optim.Optimizer, cuda: bool = None):
        warnings.warn('The `Model` interface is deprecated. Use `train_step` and `inference_step` instead.',
                      DeprecationWarning)

        if isinstance(logits2loss, Module):
            to_cuda(logits2loss, cuda)

        self.cuda = cuda
        self.model_core = to_cuda(model_core, cuda)
        self.logits2pred = logits2pred
        self.logits2loss = logits2loss
        self.optimizer = optimize(model_core.parameters())

    def do_train_step(self, *inputs: np.ndarray, lr):
        """
        Performs a forward-backward pass, as well as the gradient step, according to the given ``inputs``.

        Notes
        -----
        Note that both input and output are **not** of type `torch.Tensor` - the conversion
        to torch.Tensor is made inside this function.
        """
        return do_train_step(*inputs, lr=lr, inputs2logits=self.model_core,
                             logits2loss=self.logits2loss, optimizer=self.optimizer)

    def do_inf_step(self, *inputs: np.ndarray):
        """
        Returns the prediction for the given ``inputs``.

        Notes
        -----
        Note that both input and output are **not** of type `torch.Tensor` - the conversion
        to torch.Tensor is made inside this function.
        """
        return do_inf_step(*inputs, inputs2logits=self.model_core, logits2pred=self.logits2pred)

    def save(self, path: str):
        """Save the weights to ``path``."""
        save_model_state(self.model_core, path)
        return self

    def load(self, path: str, modify_state_fn: callable = None):
        """Load the weights from ``path``."""
        load_model_state(self.model_core, path, modify_state_fn=modify_state_fn)
        return self
