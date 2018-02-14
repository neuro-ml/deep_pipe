from typing import Sequence, Union

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


def hausdorff(a, b, weights: Union[float, Sequence] = 1) -> float:
    """
    Calculates the Hausdorff distance between two masks.

    Parameters
    ----------

    a,b : bool tensor
        The tensors containing the masks. The tensors dimensionality must match.
    weights: number, Sequence
        The weight along each axis (for anisotropic grids). If array, its length must
        match the dimensionality of the arrays. If number, all the axes will have the same weight

    Notes
    -----
    Before using this module, install the dependencies:
        > git clone https://github.com/mavillan/py-hausdorff.git
        > pip install Cython
        > cd py-hausdorff
        > python setup.py build && python setup.py install

    Examples
    --------
    >>> hausdorff(x, y, weights=2) # isotropic, but weighted
    >>> hausdorff(x, y, weights=(1,1,1,5)) # anisotropic
    """
    from hausdorff import weighted_hausdorff

    try:
        # check if array
        len(weights)
    except TypeError:
        weights = [weights] * a.ndim
    weights = np.array(weights, 'float64')

    def prep(x):
        return x.copy(order='C').astype('float64')

    return weighted_hausdorff(prep(a), prep(b), weights)


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


def compute_dices_from_segm_prob(segm_true, segm_prob, segm2msegm, empty_val: float = 1):
    """
    Channelwise dice score between msegms for predicted segmentation and true segmentation.
    """
    return multichannel_dice_score(segm2msegm(segm_true), segm2msegm(np.argmax(segm_prob, axis=0)), empty_val=empty_val)


def compute_dices_from_msegm_prob(msegm_true, msegm_prob, empty_val: float = 1):
    """
    Channelwise dice score between msegms for predicted msegm and true msegm.
    """
    return multichannel_dice_score(msegm_true, msegm_prob > 0.5, empty_val=empty_val)
