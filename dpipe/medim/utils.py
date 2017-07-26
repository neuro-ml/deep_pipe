import numpy as np


def extract(l, idx):
    return [l[i] for i in idx]


def calc_dice(num, den):
    if den == 0:
        temp_dice = 0
    else:
        temp_dice = num / den
    return temp_dice


def calc_max_dice(y_true, y_pred):
    dices = []
    y_true = np.squeeze(y_true)
    y_pred = np.squeeze(y_pred.squeeze)

    for p in np.linspace(0, 1, 100):
        temp_mask = (y_pred > p).astype(int)
        num = 2 * np.sum(temp_mask * y_true, (1, 2, 3))
        den = np.sum(temp_mask, (1, 2, 3)) + np.sum(y_true, (1, 2, 3))

        temp_dice = np.array(list(map(calc_dice, num, den))).mean()
        dices.append(temp_dice)
    return max(dices)
