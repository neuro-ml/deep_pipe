from functools import partial

import numpy as np

from .lr_base import LearningRate
from .utils import make_find_next_lr, make_check_loss_decrease


class Constant(LearningRate):
    def __init__(self, value):
        super().__init__()
        self.value = value

    def next_lr(self, **kwargs):
        return self.value


class Decreasing(LearningRate):
    def __init__(self, lr_init: float, lr_dec_mul: float, patience: int, rtol, atol):
        super().__init__()

        self.last_lr = lr_init
        self.find_next_lr = make_find_next_lr(lr_init, lambda lr: lr * lr_dec_mul,
                                              partial(make_check_loss_decrease, patience=patience, rtol=rtol,
                                                      atol=atol))

    def next_lr(self, val_losses, **kwargs):
        if self.step != 0 or self.epoch == 1:
            return self.last_lr

        assert val_losses, 'This policy requires validation losses'
        self.last_lr = self.find_next_lr(np.mean(val_losses))
        return self.last_lr
