"""
Various functions that can be used to build predictors.
"""
from typing import Callable

import numpy as np


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
