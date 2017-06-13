import numpy as np

import medim

from sklearn.model_selection import KFold


def get_val_deepmedic(xo_val, yo_val, x_det_padding,
                      x_con_padding):
    size = np.array(xo_val.shape[1:])
    x_con_size = size + 2 * x_con_padding
    if not np.array_equal(x_con_size % 6, [0] * x_con_size.ndim):
        x_con_size = x_con_size + 6 - x_con_size % 6
    new_size = x_con_size - 2 * x_con_padding

    x_con_size[1] = size[1] + 4 * x_con_padding[1]
    if not x_con_size[1] % 6 == 0:
        x_con_size[1] = x_con_size[1] + 6 - x_con_size[1] % 6
    new_size[1] = x_con_size[1] - 4 * x_con_padding[1]

    padding = list(zip((new_size - size) // 2 + x_det_padding,
                       (new_size - size) - (
                       new_size - size) // 2 + x_det_padding))
    non_spatial = xo_val.ndim - 3
    xo_det = np.pad(xo_val, [(0, 0)] * non_spatial + padding, mode='constant')

    padding = list(zip((new_size - size) // 2 + x_con_padding,
                       (new_size - size) - (
                       new_size - size) // 2 + x_con_padding))
    xo_con = np.pad(xo_val, [(0, 0)] * non_spatial + padding, mode='constant')

    padding = list(zip((new_size - size) // 2,
                       (new_size - size) - (new_size - size) // 2))
    non_spatial = yo_val.ndim - 3
    yo = np.pad(yo_val, [(0, 0)] * non_spatial + padding, mode='constant')

    x_det1, x_det2 = medim.utils.divide(xo_det, [0, *x_det_padding],
                                        n_parts_per_axis=[1, 1, 2, 1])
    x_con1, x_con2 = medim.utils.divide(xo_con, [0, *x_con_padding],
                                        n_parts_per_axis=[1, 1, 2, 1])
    y1, y2 = medim.utils.divide(yo, [0] * yo.ndim, n_parts_per_axis=[1, 2, 1])
    return ((x_det1, x_con1, y1), (x_det2, x_con2, y2))


def sum_probs2017(yo_pred_proba):
    msegm = np.zeros((3, *yo_pred_proba.shape[1:]))
    msegm[0] = yo_pred_proba[1:].sum(axis=0)
    msegm[1] = yo_pred_proba[[1, 3]].sum(axis=0)
    msegm[2] = yo_pred_proba[3]
    return msegm


def sum_probs2015(yo_pred_proba):
    msegm = np.zeros((3, *yo_pred_proba.shape[1:]))
    msegm[0] = yo_pred_proba[1:].sum(axis=0)
    msegm[1] = yo_pred_proba[[1, 3, 4]].sum(axis=0)
    msegm[2] = yo_pred_proba[4]
    return msegm


def make_check_decrease(patience: int, rtol: float=0.0, atol: float=0):
    best_score = np.inf
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


def symmetry_padding(x, padding):
    # 3-dimensional spatial
    non_spatial = x.ndim - 3
    padding = [(0, 0)] * non_spatial + [*zip(padding, padding)]

    return np.pad(x, padding, mode='constant')


def split_data(metadata):
    patients = metadata.index.values

    sd = ~metadata.survival.isnull().values

    np.random.seed(17)

    train = ~sd
    val = ~train * (np.random.randint(2, size=len(patients), dtype=bool))
    test = (~train) & (~val)

    train = np.argwhere(train)[:, 0]
    val = np.argwhere(val)[:, 0]
    test = np.argwhere(test)[:, 0]

    train_val = train[-5:]
    train = train[:-5]

    return list(train), list(train_val), list(val), list(test)


def split_data_2015(metadata):
    trains, vals, tests = [], [], []
    cv5 = KFold(5, shuffle=True, random_state=42)
    for train, test in cv5.split(range(len(metadata))):
        train, val = train[:-5], train[-5:]
        trains.append(train)
        vals.append(val)
        tests.append(test)
    return trains, vals, tests


def compute_dices_msegm(msegm_pred, msegm_true):
    dices = [medim.metrics.dice_score(msegmo_pred, msegmo_true)
             for msegmo_pred, msegmo_true
             in zip(msegm_pred, msegm_true)]
    return dices


def get_dice_threshold(msegms, msegms_true):
    thresholds = []

    n_chans_msegm = len(msegms[0])
    for i in range(n_chans_msegm):
        ps = np.linspace(0, 0.99, 20)
        best_p = 0
        best_score = 0
        for p in ps:
            score = np.mean(
                [medim.metrics.dice_score(pred[i] > p, true[i], empty_val=0)
                 for pred, true in zip(msegms, msegms_true)], axis=0)
            if score is np.nan or None:
                print('None')
                score = 1

            if score > best_score:
                best_p = p
                best_score = score
        thresholds.append(best_p)
    return thresholds
