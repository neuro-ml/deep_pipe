from functools import partial, wraps
from typing import Sequence, Dict, Callable

import numpy as np


def check_bool(*arrays):
    for i, a in enumerate(arrays):
        assert a.dtype == bool, f'{i}: {a.dtype}'


def add_check_bool(func):
    """Check that all function arguments are boolean arrays."""

    @wraps(func)
    def new_func(*arrays):
        check_bool(*arrays)
        return func(*arrays)

    return new_func


def fraction(numerator, denominator, empty_val: float = 1):
    assert numerator <= denominator, f'{numerator}, {denominator}'
    return numerator / denominator if denominator != 0 else empty_val


def dice_score(x: np.ndarray, y: np.ndarray, empty_val: float = 1) -> float:
    """
    Dice score between two binary masks.

    Parameters
    ----------
    x,y : binary tensor
    empty_val: float, optional
        Default value, which is returned if the dice score is undefined (i.e. division by zero).
    """
    assert x.shape == y.shape
    check_bool(x, y)

    return fraction(2 * np.sum(x & y), np.sum(x) + np.sum(y), empty_val)


@add_check_bool
def sensitivity(y_true, y_pred):
    return fraction(np.sum(y_pred & y_true), np.sum(y_true))


@add_check_bool
def specificity(y_true, y_pred):
    return fraction(np.sum(y_pred & y_true), np.sum(y_pred), empty_val=0)


def get_area(start, stop):
    return np.product(np.maximum(stop - start, 0))


def box_iou(a_start_stop, b_start_stop):
    i = get_area(np.maximum(a_start_stop[0], b_start_stop[0]), np.minimum(a_start_stop[1], b_start_stop[1]))
    u = get_area(*a_start_stop) + get_area(*b_start_stop) - i
    if u <= 0:
        print(f'{a_start_stop} {b_start_stop}')
    return fraction(i, u)


# TODO: replace by a more general function
def multichannel_dice_score(a, b, empty_val: float = 1) -> [float]:
    """
    Channelwise dice score between two binary masks.
    The first dimension of the tensors is assumed to be the channels.

    Parameters
    ----------
    a,b : binary tensor
    empty_val: float, optional
        Default value, which is returned if the dice score
        is undefined (i.e. division by zero).

    Returns
    -------
    dice_score: [float]
    """
    assert len(a) == len(b), f'number of channels is different: {len(a)} != {len(b)}'
    dices = [dice_score(x, y, empty_val=empty_val) for x, y in zip(a, b)]
    return dices


def calc_max_dices(true_masks: Sequence, predicted_masks: Sequence) -> float:
    """
    Calculates the dice score between the true and predicted masks.
    The threshold is selected to maximize the mean dice.

    Parameters
    ----------
    true_masks: bool tensor
    predicted_masks: float tensor

    Returns
    -------
    dice_score: float
    """
    assert len(true_masks) == len(predicted_masks)

    dices = []
    thresholds = np.linspace(0, 1, 20)
    for true, pred in zip(true_masks, predicted_masks):
        temp = []
        for i in range(len(true)):
            temp.append([dice_score(pred[i] > thr, true[i].astype(bool))
                         for thr in thresholds])
        dices.append(temp)
    dices = np.asarray(dices)
    return dices.mean(axis=0).max(axis=1)


def aggregate_metric(xs, ys, metric, aggregate_fn=np.mean):
    """Compute metric for array of objects from metric on couple of objects."""
    return aggregate_fn([metric(x, y) for x, y in zip(xs, ys)])


def convert_to_aggregated(metrics: Dict[str, Callable], aggregate_fn=np.mean, key_prefix=''):
    return dict(zip(
        [key_prefix + key for key in metrics],
        [partial(aggregate_metric, metric=metric, aggregate_fn=aggregate_fn) for metric in metrics.values()]
    ))
