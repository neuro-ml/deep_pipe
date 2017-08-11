import math
from functools import partial

from dpipe.dl.model_controller import ModelController
from dpipe.utils.batch_iter_factory import BatchIterFactory


def make_check_loss_decrease(patience: int, rtol: float, atol: float):
    best_score = math.inf
    iters_waited = 0

    def check_decrease(score):
        nonlocal best_score, iters_waited, patience, rtol
        if score < best_score:
            iters_waited = 0
            # To get the next best result we need to beat either atol or rtol
            best_score = max([score * (1 - rtol), score - atol])
        else:
            iters_waited += 1

        return iters_waited >= patience

    return check_decrease


def make_find_next_lr(lr, decrease_lr: callable, get_check: callable):
    check = get_check()

    def find_next_lr(loss):
        nonlocal lr, check
        if check(loss):
            check = get_check()
            lr = decrease_lr(lr)
        return lr

    return find_next_lr


def train_with_lr_decrease(
        model_controller: ModelController,
        train_batch_iter_factory: BatchIterFactory,
        val_ids, data_loader, *, n_epochs, lr_init, lr_dec_mul=0.5,
        patience: int, rtol=0, atol=0):
    x_val = [data_loader.load_x(p) for p in val_ids]
    y_val = [data_loader.load_y(p) for p in val_ids]

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
            lr = find_next_lr(val_loss)
