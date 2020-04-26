from typing import Union, Sequence

import numpy as np

from dpipe.itertools import lmap
from ..checks import join

AxesLike = Union[int, Sequence[int]]
AxesParams = Union[float, Sequence[float]]


def fill_by_indices(target, values, indices):
    """Replace the values in ``target`` located at ``indices`` by the ones from ``values``."""
    indices = expand_axes(indices, values)
    target = np.array(target)
    target[list(indices)] = values
    return tuple(target)


def broadcast_to_axes(axes: Union[AxesLike, None], *arrays: AxesParams):
    """Broadcast ``arrays`` to the length of ``axes``. Axes are inferred from the arrays if necessary."""
    if not arrays:
        raise ValueError('No arrays provided.')

    arrays = lmap(np.atleast_1d, arrays)
    lengths = lmap(len, arrays)
    if axes is None:
        axes = list(range(-max(lengths), 0))
    axes = check_axes(axes)

    if not all(len(axes) == x or x == 1 for x in lengths):
        raise ValueError(f'Axes and arrays are not broadcastable: {len(axes)} vs {join(lengths)}.')

    arrays = [np.repeat(x, len(axes) // len(x), 0) for x in arrays]
    return (axes, *arrays)


def check_axes(axes) -> tuple:
    axes = np.atleast_1d(axes)
    if axes.ndim != 1:
        raise ValueError(f'Axes must be 1D, but {axes.ndim}D provided.')
    if not np.issubdtype(axes.dtype, np.integer):
        raise ValueError(f'Axes must be integer, but {axes.dtype} provided.')
    axes = tuple(axes)
    if len(axes) != len(set(axes)):
        raise ValueError(f'Axes contain duplicates: {axes}.')
    return axes


def expand_axes(axes, values) -> tuple:
    return broadcast_to_axes(axes, values)[0]


def ndim2spatial_axes(ndim):
    """
    >>> ndim2spatial_axes(3)
    (-3, -2, -1)

    >>> ndim2spatial_axes(1)
    (-1,)
    """
    return tuple(range(-ndim, 0))
