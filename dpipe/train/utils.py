import math


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
            # Recreate check so that we would start checking again.
            # TODO rewrite it without it
            check = get_check()
            lr = decrease_lr(lr)
        return lr

    return find_next_lr
