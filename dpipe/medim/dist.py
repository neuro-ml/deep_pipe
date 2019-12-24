from collections import Callable
from typing import Union, Sequence

import torch
import numpy as np

from .axes import fill_by_indices, AxesLike
from .itertools import zip_equal, collect
from ..torch import to_var

Tensor = Union[np.ndarray, torch.Tensor]


def weighted_sum(weights: Tensor, axes: AxesLike, values_range: Callable) -> Tensor:
    if not isinstance(axes, int):
        axes = tuple(axes)

    values = values_range(np.array(weights.shape)[axes])

    shape = fill_by_indices(np.ones_like(weights.shape), values.shape, axes)
    values = values.reshape(*shape)
    if isinstance(weights, torch.Tensor) and not isinstance(values, torch.Tensor):
        values = to_var(values, weights).to(weights)

    return (weights * values).sum(axes)


def polynomial(n: int, order=1) -> np.ndarray:
    power = order + 1
    return np.arange(n) ** power / power


def expectation(distribution: Tensor, axis: int, integral: Callable = polynomial, *args, **kwargs) -> Tensor:
    """
    Calculates the expectation of a given function ``f``.
    integral: Callable(x) -> int_0^x f(t)dt
    """

    def integral_delta(n):
        values = integral(n + 1, *args, **kwargs)
        return values[1:] - values[:-1]  # can't use np.diff for compatibility with pytorch

    return weighted_sum(distribution, axis, integral_delta)


# TODO: add tests
@collect
def conditional_expectation(distribution: Tensor, integrals: Union[Callable, Sequence[Callable]], axes: AxesLike):
    axes = np.core.numeric.normalize_axis_tuple(axes, distribution.ndim, 'axes')
    if callable(integrals):
        integrals = [integrals]
    if len(integrals) == 1:
        integrals = [integrals[0]] * len(axes)

    for axis, integral in zip_equal(axes, integrals):
        # sum over other axes, but allow for reduction of `axis`
        other_axes = list(axes)
        other_axes.remove(axis)
        other_axes = np.array(other_axes)
        other_axes[other_axes > axis] -= 1

        yield expectation(distribution, axis, integral).sum(tuple(other_axes))
