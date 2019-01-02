from functools import partial
from typing import Sequence, Dict, Callable

import numpy as np
from scipy.ndimage.morphology import distance_transform_edt, binary_erosion

from .checks import add_check_bool, add_check_shapes, check_shapes, check_bool
from .utils import zip_equal


def fraction(numerator, denominator, empty_val: float = 1):
    assert numerator <= denominator, f'{numerator}, {denominator}'
    return numerator / denominator if denominator != 0 else empty_val


@add_check_bool
@add_check_shapes
def dice_score(x: np.ndarray, y: np.ndarray) -> float:
    return fraction(2 * np.sum(x & y), np.sum(x) + np.sum(y))


@add_check_bool
@add_check_shapes
def sensitivity(y_true, y_pred):
    return fraction(np.sum(y_pred & y_true), np.sum(y_true))


@add_check_bool
@add_check_shapes
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
@np.deprecate
def multichannel_dice_score(a, b) -> [float]:
    """
    Channelwise dice score between two binary masks.
    The first dimension of the tensors is assumed to be the channels.
    """
    assert len(a) == len(b), f'number of channels is different: {len(a)} != {len(b)}'
    return list(map(dice_score, a, b))


@np.deprecate
def calc_max_dices(true_masks: Sequence, predicted_masks: Sequence) -> float:
    """
    Calculates the dice score between the true and predicted masks.
    The threshold is selected to maximize the mean dice.

    Parameters
    ----------
    true_masks: bool tensor
    predicted_masks: float tensor
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
    """Aggregate a `metric` computed on pairs from `xs` and `ys`"""
    return aggregate_fn([metric(x, y) for x, y in zip_equal(xs, ys)])


def convert_to_aggregated(metrics: Dict[str, Callable], aggregate_fn: Callable = np.mean, key_prefix: str = ''):
    return {
        key_prefix + key: partial(aggregate_metric, metric=metric, aggregate_fn=aggregate_fn)
        for key, metric in metrics.items()
    }


@add_check_bool
@add_check_shapes
def recall(y_true, y_pred):
    tp = np.count_nonzero(np.logical_and(y_pred, y_true))
    fn = np.count_nonzero(np.logical_and(~y_pred, y_true))

    return fraction(tp, tp + fn, 0)


@add_check_bool
@add_check_shapes
def precision(y_true, y_pred):
    tp = np.count_nonzero(y_pred & y_true)
    fp = np.count_nonzero(y_pred & ~y_true)

    return fraction(tp, tp + fp, 0)


def surface_distances(y_true, y_pred, voxel_shape=None):
    check_bool(y_pred, y_true)
    check_shapes(y_pred, y_true)

    pred_border = np.logical_xor(y_pred, binary_erosion(y_pred))
    true_border = np.logical_xor(y_true, binary_erosion(y_true))

    dt = distance_transform_edt(~true_border, sampling=voxel_shape)
    return dt[pred_border]


def assd(x, y, voxel_shape=None):
    sd1 = surface_distances(y, x, voxel_shape=voxel_shape)
    sd2 = surface_distances(x, y, voxel_shape=voxel_shape)
    if sd1.size == 0 and sd2.size == 0:
        return 0
    if sd1.size == 0 or sd2.size == 0:
        return np.nan

    return np.mean([sd1.mean(), sd2.mean()])


def hausdorff_distance(x, y, voxel_shape=None):
    sd1 = surface_distances(y, x, voxel_shape=voxel_shape)
    sd2 = surface_distances(x, y, voxel_shape=voxel_shape)
    if sd1.size == 0 and sd2.size == 0:
        return 0
    if sd1.size == 0 or sd2.size == 0:
        return np.nan

    return max(sd1.max(), sd2.max())
