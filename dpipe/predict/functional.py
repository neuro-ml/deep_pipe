"""
Various functions that can be used to build predictors.
"""
from typing import Callable

import numpy as np

from dpipe.medim.utils import pad, build_slices, unpack_args
from dpipe.medim.axes import ndim2spatial_axes


def chain_decorators(*decorators: Callable, predict: Callable):
    """
    Wraps ``predict`` into a series of ``decorators``.

    Examples
    --------
    >>> @decorator1
    >>> @decorator2
    >>> def f(x):
    >>>     return x + 1
    >>> # same as:
    >>> def f(x):
    >>>     return x + 1
    >>>
    >>> f = chain_decorators(decorator1, decorator2, predict=f)
    """
    for decorator in reversed(decorators):
        predict = decorator(predict)
    return predict


def preprocess(func, *args, **kwargs):
    """
    Applies function ``func`` with given parameters before making a prediction.

    Examples
    --------
        >>> from dpipe.medim.shape_ops import pad
        >>> from dpipe.predict.functional import preprocess
        >>>
        >>> @preprocess(pad, padding=[10, 10, 10], padding_values=np.min)
        >>> def predict(x):
        >>>     return model.do_inf_step(x)
        performs spatial padding before prediction.

    References
    ----------
    `postprocess`
    """

    def decorator(predict):
        def wrapper(x):
            x = func(x, *args, **kwargs)
            x = predict(x)
            return x

        return wrapper

    return decorator


def postprocess(func, *args, **kwargs):
    """
    Applies function ``func`` with given parameters after making a prediction.

    References
    ----------
    `preprocess`
    """

    def decorator(predict):
        def wrapper(x):
            x = predict(x)
            x = func(x, *args, **kwargs)
            return x

        return wrapper

    return decorator


def pad_spatial_size(x, spatial_size: np.array):
    ndim = len(spatial_size)
    padding = np.zeros((x.ndim, 2), dtype=int)
    padding[-ndim:, 1] = spatial_size - x.shape[-ndim:]
    return pad(x, padding, np.min(x, axis=ndim2spatial_axes(ndim), keepdims=True))


def trim_spatial_size(x, spatial_size):
    return x[(..., *build_slices(spatial_size))]


def pad_to_dividable(x, divisor, ndim=3):
    """Pads `x`'s last `ndim` dimensions to be dividable by `divisor` and returns it."""
    spatial_shape = np.array(x.shape[-ndim:])
    return pad_spatial_size(x, spatial_size=spatial_shape + (divisor - spatial_shape) % divisor)


# deprecated: 20.04.2019
@np.deprecate
def predict_input_parts(batch_iterator, *, predict):
    return map(predict, batch_iterator)


@np.deprecate
def predict_inputs_parts(batch_iterator, *, predict):
    return predict_input_parts(batch_iterator, predict=unpack_args(predict))
