"""
`Model` is an interface that lets us decouple the DL framework we use from the rest of the library.

It implements the loading/saving of a neural to disk, as well as training and inference.
This implementation is a wrapper for the PyTorch framework.
"""
from torch.nn import Module

from ..model import Model
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
