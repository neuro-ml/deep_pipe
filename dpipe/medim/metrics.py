from typing import Sequence

import numpy as np


def dice_score(x, y, empty_val: float = 1) -> float:
    """
    Dice score between two binary masks.

    Parameters
    ----------
    x,y : binary tensor
    empty_val: float, optional
        Default value, which is returned if the dice score
        is undefined (i.e. division by zero).

    Returns
    -------
    dice_score: float
    """
    assert x.dtype == y.dtype == np.bool
    assert x.shape == y.shape

    num = 2 * np.sum(x & y)
    den = np.sum(x) + np.sum(y)

    return empty_val if den == 0 else num / den


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
