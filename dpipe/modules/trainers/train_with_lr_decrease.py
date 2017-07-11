import math

from ..dl.model_controller import ModelController
from ..datasets import Dataset
from ..batch_iterators import BatchIter


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
        model_controller: ModelController, train_batch_iter: BatchIter, val_ids,
        dataset: Dataset, *, n_iters_per_epoch, n_epochs, lr_init,
        lr_dec_mul=0.5, patience, rtol=0, atol=0):
    x_val = [dataset.load_x(p) for p in val_ids]
    y_val = [dataset.load_y(p) for p in val_ids]

    find_next_lr = make_find_next_lr(
        lr_init, lambda lr: lr * lr_dec_mul,
        make_check_loss_decrease(patience, rtol, atol))

    lr = find_next_lr(math.inf)
    with train_batch_iter:
        for _ in range(n_epochs):
            train_loss = model_controller.train(train_batch_iter, lr=lr)
            y_pred, val_loss = model_controller.validate(x_val, y_val)
            lr = find_next_lr(val_loss)
