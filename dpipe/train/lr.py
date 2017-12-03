from functools import partial

import numpy as np

from .utils import make_find_next_lr, make_check_loss_decrease


def decreasing(lr_init: float, lr_dec_mul: float, patience: int, rtol, atol):
    find_next_lr = make_find_next_lr(lr_init, lambda lr: lr * lr_dec_mul,
                                     partial(make_check_loss_decrease, patience=patience, rtol=rtol, atol=atol))

    def new_lr(epoch, val_losses, **kwargs):
        if epoch == 0:
            return lr_init
        return find_next_lr(np.mean(val_losses))

    return new_lr
