from typing import Dict, Callable, Union, Sequence

import numpy as np
from scipy.ndimage.morphology import distance_transform_edt, binary_erosion

from ..checks import add_check_bool, add_check_shapes, check_shapes, check_bool
from dpipe.itertools import zip_equal

__all__ = [
    'dice_score', 'sensitivity', 'specificity', 'precision', 'recall', 'iou', 'assd', 'hausdorff_distance',
    'cross_entropy_with_logits',
    'convert_to_aggregated', 'to_aggregated', 'fraction',
]


def to_aggregated(metric: Callable, aggregate: Callable = np.mean, *args, **kwargs):
    """
    Converts a ``metric`` that receives two values to a metric
    that receives two sequences and returns an aggregated value.

    ``args`` and ``kwargs`` are passed as additional arguments ot ``aggregate``.

    Examples
    --------
    >>> mean_dice = to_aggregated(dice_score)
    >>> worst_dice = to_aggregated(dice_score, aggregate=np.min)
    """

    def wrapper(xs: Sequence, ys: Sequence, *args_, **kwargs_):
        return aggregate([metric(x, y, *args_, **kwargs_) for x, y in zip(xs, ys)], *args, **kwargs)

    return wrapper


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
    tn = np.sum((~y_true) & (~y_pred))
    fp = np.sum(y_pred & (~y_true))
    return fraction(tn, tn + fp, empty_val=0)


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


@add_check_bool
@add_check_shapes
def iou(x: np.ndarray, y: np.ndarray) -> float:
    return fraction(np.sum(x & y), np.sum(x | y))


def get_area(start, stop):
    return np.product(np.maximum(stop - start, 0))


def box_iou(a_start_stop, b_start_stop):
    i = get_area(np.maximum(a_start_stop[0], b_start_stop[0]), np.minimum(a_start_stop[1], b_start_stop[1]))
    u = get_area(*a_start_stop) + get_area(*b_start_stop) - i
    if u <= 0:
        print(f'{a_start_stop} {b_start_stop}')
    return fraction(i, u)


def aggregate_metric(xs, ys, metric, aggregate_fn=np.mean):
    """Aggregate a `metric` computed on pairs from `xs` and `ys`"""
    return aggregate_fn([metric(x, y) for x, y in zip_equal(xs, ys)])


def convert_to_aggregated(metrics: Dict[str, Callable], aggregate_fn: Callable = np.mean,
                          key_prefix: str = '', key_suffix: str = '', *args, **kwargs):
    return {
        key_prefix + key + key_suffix: to_aggregated(metric, aggregate_fn, *args, **kwargs)
        for key, metric in metrics.items()
    }


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


def cross_entropy_with_logits(target: np.ndarray, logits: np.ndarray, axis: int = 1,
                              reduce: Union[Callable, None] = np.mean):
    """
    A numerically stable cross entropy for numpy arrays.
    ``target`` and ``logits`` must have the same shape except for ``axis``.

    Parameters
    ----------
    target
        integer array of shape (d1, ..., di, dj, ..., dn)
    logits
        array of shape (d1, ..., di, k, dj, ..., dn)
    axis
        the axis containing the logits for each class: ``logits.shape[axis] == k``
    reduce
        the reduction operation to be applied to the final loss.
        If None - no reduction will be performed.
    """
    main = np.take_along_axis(logits, np.expand_dims(target, axis), axis)
    max_ = np.maximum(0, logits.max(axis, keepdims=True))

    loss = -main + max_ + np.log(np.exp(logits - max_).sum(axis, keepdims=True))
    loss = loss.squeeze(axis)

    if reduce is not None:
        loss = reduce(loss)
    return loss
