"""
Functions to work with boxes: immutable numpy arrays of shape (2, n) which represent
the coordinates of the upper left and lower right corners of an n-dimensional rectangle.

In slicing operations, as everywhere in Python, the left corner is inclusive, and the right one is non-inclusive.
"""
import itertools
from functools import wraps
from typing import Callable

import numpy as np

from ..checks import check_len
from .shape_utils import compute_shape_from_spatial
from .utils import build_slices

# box type
Box = np.ndarray


def make_box_(iterable) -> Box:
    """
    Returns a box, generated inplace from the ``iterable``. If ``iterable`` was a numpy array, will make it
    immutable and return.
    """
    box = np.asarray(iterable)
    box.setflags(write=False)

    assert box.ndim == 2 and len(box) == 2, box.shape
    assert np.all(box[0] <= box[1]), box

    return box


def get_volume(box: Box):
    return np.prod(box[1] - box[0], axis=0)


def returns_box(func: Callable) -> Callable:
    """Returns function, decorated so that it returns a box."""

    @wraps(func)
    def func_returning_box(*args, **kwargs):
        return make_box_(func(*args, **kwargs))

    func_returning_box.__annotations__['return'] = Box
    return func_returning_box


@returns_box
def get_containing_box(shape: tuple):
    """Returns box that contains complete array of shape ``shape``."""
    return [0] * len(shape), shape


@returns_box
def broadcast_box(box: Box, shape: tuple, dims: tuple):
    """
    Returns box, such that it contains ``box`` across ``dims`` and whole array
    with shape ``shape`` across other dimensions.
    """
    return (compute_shape_from_spatial([0] * len(shape), box[0], dims),
            compute_shape_from_spatial(shape, box[1], dims))


@returns_box
def limit_box(box, limit):
    """
    Returns a box, maximum subset of the input ``box`` so that start would be non-negative and
    stop would be limited by the ``limit``.
    """
    check_len(*box, limit)
    return np.maximum(box[0], 0), np.minimum(box[1], limit)


def get_box_padding(box: Box, limit):
    """
    Returns padding that is necessary to get ``box`` from array of shape ``limit``.
     Returns padding in numpy form, so it can be given to `numpy.pad`.
     """
    check_len(*box, limit)
    return np.maximum([-box[0], box[1] - limit], 0).T


@returns_box
def get_union_box(*boxes):
    start = np.min([box[0] for box in boxes], axis=0)
    stop = np.max([box[1] for box in boxes], axis=0)
    return start, stop


@returns_box
def add_margin(box: Box, margin):
    """
    Returns a box with size increased by the ``margin`` (need to be broadcastable to the box)
    compared to the input ``box``.
    """
    margin = np.broadcast_to(margin, box.shape)
    return box[0] - margin[0], box[1] + margin[1]


@returns_box
def get_centered_box(center: np.ndarray, box_size: np.ndarray):
    """
    Get box of size ``box_size``, centered in the ``center``.
    If ``box_size`` is odd, ``center`` will be closer to the right.
    """
    start = center - box_size // 2
    stop = center + box_size // 2 + box_size % 2
    return start, stop


@returns_box
def mask2bounding_box(mask: np.ndarray):
    """
    Find the smallest box that contains all true values of the ``mask``.
    """
    if not mask.any():
        raise ValueError('The mask is empty.')

    start, stop = [], []
    for ax in itertools.combinations(range(mask.ndim), mask.ndim - 1):
        nonzero = np.any(mask, axis=ax)
        if np.any(nonzero):
            left, right = np.where(nonzero)[0][[0, -1]]
        else:
            left, right = 0, 0
        start.insert(0, left)
        stop.insert(0, right + 1)
    return start, stop


def box2slices(box: Box):
    return build_slices(*box)
