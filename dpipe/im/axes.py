from typing import Union, Sequence

import numpy as np

from dpipe.itertools import lmap
from ..checks import join

AxesLike = Union[int, Sequence[int]]
AxesParams = Union[float, Sequence[float]]


def fill_by_indices(target, values, indices):
    """Replace the values in ``target`` located at ``indices`` by the ones from ``values``."""
    assert indices is not None
    indices = check_axes(indices)
    target = np.array(target)
    target[list(indices)] = values
    return tuple(target)


def broadcast_to_axis(axis: AxesLike, *arrays: AxesParams):
    """Broadcast ``arrays`` to the length of ``axes``. Axes are inferred from the arrays if necessary."""
    if not arrays:
        raise ValueError('No arrays provided.')

    arrays = lmap(np.atleast_1d, arrays)
    lengths = lmap(len, arrays)
    if axis is None:
        raise ValueError('`axis` cannot be None')

    axis = check_axes(axis)

    if not all(len(axis) == x or x == 1 for x in lengths):
        raise ValueError(f'Axes and arrays are not broadcastable: {len(axis)} vs {join(lengths)}.')

    return tuple(np.repeat(x, len(axis) // len(x), 0) for x in arrays)


def axis_from_dim(axis: Union[AxesLike, None], dim: int) -> tuple:
    if axis is None:
        return tuple(range(dim))

    axis = check_axes(axis)
    left, right = -dim, dim - 1
    if min(axis) < left or max(axis) > right:
        raise ValueError(f'For dim={dim} axis must be within ({left}, {right}): but provided {axis}.')

    return np.core.numeric.normalize_axis_tuple(axis, dim, 'axis')


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


def ndim2spatial_axes(ndim):
    """
    >>> ndim2spatial_axes(3)
    (-3, -2, -1)

    >>> ndim2spatial_axes(1)
    (-1,)
    """
    return tuple(range(-ndim, 0))


def resolve_deprecation(axis, ndim, *values):
    # in this case the behaviour is not consistent
    if axis is None and any(ndim != len(np.atleast_1d(v)) for v in values):
        raise ValueError('In the future the last axes will not be used to infer `axis`. '
                         'Pass the appropriate `axis` to suppress this error.')

    return axis_from_dim(axis, ndim)


@np.deprecate
def expand_axes(axes, values) -> tuple:
    if axes is None:
        raise ValueError('`axis` cannot be None')
    return check_axes(axes)
