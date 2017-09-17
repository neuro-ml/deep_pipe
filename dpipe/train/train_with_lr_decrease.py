import math
from functools import partial

from dpipe.dl.model_controller import ModelController
from dpipe.utils.batch_iter_factory import BatchIterFactory
from dpipe.config import register
from .utils import make_find_next_lr, make_check_loss_decrease


@register()
def train_with_lr_decrease(
        model_controller: ModelController,
        train_batch_iter_factory: BatchIterFactory,
        val_ids, load_x, load_y, *, n_epochs, lr_init, lr_dec_mul=0.5,
        patience: int, rtol=0, atol=0):
    x_val = [load_x(p) for p in val_ids]
    y_val = [load_y(p) for p in val_ids]

    find_next_lr = make_find_next_lr(
        lr_init, lambda lr: lr * lr_dec_mul,
        partial(make_check_loss_decrease, patience=patience,
                rtol=rtol, atol=atol))

    lr = find_next_lr(math.inf)
    with train_batch_iter_factory:
        for _ in range(n_epochs):
            with next(train_batch_iter_factory) as train_batch_iter:
                train_loss = model_controller.train(train_batch_iter, lr=lr)
            y_pred, val_loss = model_controller.validate(x_val, y_val)
            lr = find_next_lr(train_loss)
