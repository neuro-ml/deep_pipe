"""
Module for calculation of various statistics given a discrete or piecewise-linear distribution.
"""
from typing import Union, Sequence, Callable

import torch
import numpy as np

from .axes import fill_by_indices, AxesLike, axis_from_dim
from ..itertools import zip_equal, collect
from ..torch import to_var

__all__ = 'weighted_sum', 'expectation', 'marginal_expectation', 'polynomial'

Tensor = Union[np.ndarray, torch.Tensor]


def weighted_sum(weights: Tensor, axis: AxesLike, values_range: Callable) -> Tensor:
    """
    Calculates a weighted sum of values returned by ``values_range`` with the corresponding
    ``weights`` along a given ``axis``.

    Parameters
    ----------
    weights
    axis
    values_range
        takes ``n`` as input and returns an array of ``n`` values where ``n = weights.shape[axis]``.
    """
    if not isinstance(axis, int):
        axis = list(axis)

    values = values_range(np.array(weights.shape)[axis])

    shape = fill_by_indices(np.ones_like(weights.shape), values.shape, axis)
    values = values.reshape(*shape)
    if isinstance(weights, torch.Tensor) and not isinstance(values, torch.Tensor):
        values = to_var(values).to(weights)

    return (weights * values).sum(axis)


def polynomial(n: int, order=1) -> np.ndarray:
    """
    The definite integral for a polynomial function of a given ``order`` from 0 to ``n - 1``.

    Examples
    --------
    >>> polynomial(10, 1) # x ** 2 / 2 from 0 to 9
    array([ 0. ,  0.5,  2. ,  4.5,  8. , 12.5, 18. , 24.5, 32. , 40.5])
    """
    power = order + 1
    return np.arange(n) ** power / power


def expectation(distribution: Tensor, axis: int, integral: Callable = polynomial, *args, **kwargs) -> Tensor:
    r"""
    Calculates the expectation of a function ``h`` given its ``integral`` and a ``distribution``.

    ``args`` and ``kwargs`` are passed to ``integral`` as  additional arguments.

    Parameters
    ----------
    distribution:
        the distribution by which the expectation will be calculated.
        Must sum to 1 along the ``axis``.
    axis:
        the axis along which the expectation is calculated.
    integral:
        the definite integral of the function ``h``.
        See `polynomial` for an example.

    Notes
    -----
    This function calculates the expectation by a piecewise-linear distribution in the range :math:`[0, N]`
    where ``N = distribution.shape[axis] + 1``:

    .. math::
        \mathbb{E}_F[h] = \int\limits_0^N h(x) dF(x) = \sum\limits_0^{N-1} \int\limits_i^{i+1} h(x) dF(x) =
        \sum\limits_0^{N-1} distribution_i \int\limits_i^{i+1} h(x) dx =
        \sum\limits_0^{N-1} distribution_i \cdot (H(i+1) - H(i)),

    where :math:`distribution_i` are taken along ``axis``, :math:`H(i) = \int\limits_0^{i} h(x) dx` are
    returned by ``integral``.

    References
    ----------
    `polynomial`
    """

    def integral_delta(n):
        values = integral(n + 1, *args, **kwargs)
        return values[1:] - values[:-1]  # can't use np.diff for compatibility with pytorch

    return weighted_sum(distribution, axis, integral_delta)


@collect
def marginal_expectation(distribution: Tensor, axis: AxesLike,
                         integrals: Union[Callable, Sequence[Callable]] = polynomial, *args, **kwargs):
    """
    Computes expectations along the ``axis`` according to ``integrals`` independently.

    ``args`` and ``kwargs`` are passed to ``integral`` as  additional arguments.
    """
    axis = axis_from_dim(axis, distribution.ndim)
    if callable(integrals):
        integrals = [integrals]
    if len(integrals) == 1:
        integrals = [integrals[0]] * len(axis)

    for ax, integral in zip_equal(axis, integrals):
        # sum over other axes, but allow for reduction of `axis`
        other_axes = list(axis)
        other_axes.remove(ax)
        other_axes = np.array(other_axes)
        other_axes[other_axes > ax] -= 1

        yield expectation(distribution, ax, integral, *args, **kwargs).sum(tuple(other_axes))
