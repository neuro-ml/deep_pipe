import warnings
from functools import wraps, partial
from typing import Union, Callable

import numpy as np

from dpipe.im.axes import broadcast_to_axes, AxesLike, AxesParams
from dpipe.im.grid import divide, combine
from dpipe.itertools import extract
from dpipe.im.shape_ops import pad_to_shape, crop_to_shape, pad_to_divisible
from dpipe.im.shape_utils import prepend_dims, extract_dims
from dpipe.itertools import pmap

__all__ = 'add_extract_dims', 'divisible_shape', 'patches_grid'


def add_extract_dims(n_add: int = 1, n_extract: int = None, sequence: bool = False):
    """
    Adds ``n_add`` dimensions before a prediction and extracts ``n_extract`` dimensions after this prediction.

    Parameters
    ----------
    n_add: int
        number of dimensions to add.
    n_extract: int, None, optional
        number of dimensions to extract. If ``None``, extracts the same number of dimensions as were added (``n_add``).
    sequence:
        if True - the output is expected to be a sequence, and the dims are extracted for each element of the sequence.
    """
    if n_extract is None:
        n_extract = n_add

    def decorator(predict):
        @wraps(predict)
        def wrapper(*xs, **kwargs):
            result = predict(*[prepend_dims(x, n_add) for x in xs], **kwargs)
            if sequence:
                return [extract_dims(entry, n_extract) for entry in result]

            return extract_dims(result, n_extract)

        return wrapper

    return decorator


def divisible_shape(divisor: AxesLike, axis: AxesLike = None, padding_values: Union[AxesParams, Callable] = 0,
                    ratio: AxesParams = 0.5, *, axes: AxesLike = None):
    """
    Pads an incoming array to be divisible by ``divisor`` along the ``axes``. Afterwards the padding is removed.

    Parameters
    ----------
    divisor
        a value an incoming array should be divisible by.
    axis
        axes along which the array will be padded. If None - the last ``len(divisor)`` axes are used.
    padding_values
        values to pad with. If Callable (e.g. ``numpy.min``) - ``padding_values(x)`` will be used.
    ratio
        the fraction of the padding that will be applied to the left, ``1 - ratio`` will be applied to the right.

    References
    ----------
    `pad_to_divisible`
    """
    if axes is not None:
        assert axis is None
        warnings.warn('`axes` has been renamed to `axis`', UserWarning)
        axis = axes

    axis, divisor, ratio = broadcast_to_axes(axis, divisor, ratio)

    def decorator(predict):
        @wraps(predict)
        def wrapper(x, *args, **kwargs):
            shape = np.array(x.shape)[list(axis)]
            x = pad_to_divisible(x, divisor, axis, padding_values, ratio)
            result = predict(x, *args, **kwargs)
            return crop_to_shape(result, shape, axis, ratio)

        return wrapper

    return decorator


def patches_grid(patch_size: AxesLike, stride: AxesLike, axis: AxesLike = None,
                 padding_values: Union[AxesParams, Callable] = 0, ratio: AxesParams = 0.5, *, axes: AxesLike = None):
    """
    Divide an incoming array into patches of corresponding ``patch_size`` and ``stride`` and then combine
    predicted patches by averaging the overlapping regions.

    If ``padding_values`` is not None, the array will be padded to an appropriate shape to make a valid division.
    Afterwards the padding is removed.

    References
    ----------
    `grid.divide`, `grid.combine`, `pad_to_shape`
    """
    if axes is not None:
        assert axis is None
        warnings.warn('`axes` has been renamed to `axis`', UserWarning)
        axis = axes

    axis, patch_size, stride = broadcast_to_axes(axis, patch_size, stride)
    valid = padding_values is not None

    def decorator(predict):
        @wraps(predict)
        def wrapper(x, *args, **kwargs):
            if valid:
                shape = np.array(x.shape)[list(axis)]
                padded_shape = np.maximum(shape, patch_size)
                new_shape = padded_shape + (stride - padded_shape + patch_size) % stride
                x = pad_to_shape(x, new_shape, axis, padding_values, ratio)

            patches = pmap(predict, divide(x, patch_size, stride, axis), *args, **kwargs)
            prediction = combine(patches, extract(x.shape, axis), stride, axis)

            if valid:
                prediction = crop_to_shape(prediction, shape, axis, ratio)
            return prediction

        return wrapper

    return decorator
