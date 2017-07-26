import numpy as np

from dpipe.medim.metrics import dice_score


def extract(l, idx):
    return [l[i] for i in idx]


def calc_max_dice(y_true, y_pred):
    dices = []
    thresholds = np.linspace(0, 1, 20)
    for true, pred in zip(y_true, y_pred):
        temp = []
        for i in range(len(true)):
            temp.append([dice_score(pred[i] > thr, true[i].astype(bool))
                         for thr in thresholds])
        dices.append(temp)
    dices = np.asarray(dices)
    return dices.mean(axis=0).max(axis=1)
