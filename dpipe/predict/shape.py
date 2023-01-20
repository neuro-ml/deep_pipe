from functools import wraps
from typing import Union, Callable, Type

import numpy as np

from ..im.axes import broadcast_to_axis, AxesLike, AxesParams, axis_from_dim, resolve_deprecation
from ..im.grid import divide, combine, get_boxes, PatchCombiner, Average
from ..itertools import extract, pmap
from ..im.shape_ops import pad_to_shape, crop_to_shape, pad_to_divisible
from ..im.shape_utils import prepend_dims, extract_dims

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
                    ratio: AxesParams = 0.5):
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

    def decorator(predict):
        @wraps(predict)
        def wrapper(x, *args, **kwargs):
            local_axis = axis_from_dim(axis, x.ndim)
            local_divisor, local_ratio = broadcast_to_axis(local_axis, divisor, ratio)

            shape = np.array(x.shape)[list(local_axis)]
            x = pad_to_divisible(x, local_divisor, local_axis, padding_values, local_ratio)
            result = predict(x, *args, **kwargs)
            return crop_to_shape(result, shape, local_axis, local_ratio)

        return wrapper

    return decorator


def patches_grid(patch_size: AxesLike, stride: AxesLike, axis: AxesLike = None,
                 padding_values: Union[AxesParams, Callable] = 0, ratio: AxesParams = 0.5,
                 combiner: Type[PatchCombiner] = Average, get_boxes: Callable = get_boxes):
    """
    Divide an incoming array into patches of corresponding ``patch_size`` and ``stride`` and then combine
    the predicted patches by aggregating the overlapping regions using the ``combiner`` - Average by default.

    If ``padding_values`` is not None, the array will be padded to an appropriate shape to make a valid division.
    Afterwards the padding is removed. Otherwise if input cannot be patched without remainder
    ``ValueError`` is raised.

    References
    ----------
    `grid.divide`, `grid.combine`, `pad_to_shape`
    """
    valid = padding_values is not None

    def decorator(predict):
        @wraps(predict)
        def wrapper(x, *args, **kwargs):
            input_axis = resolve_deprecation(axis, x.ndim, patch_size, stride)
            local_size, local_stride = broadcast_to_axis(input_axis, patch_size, stride)
            shape = extract(x.shape, input_axis)

            if valid:
                padded_shape = np.maximum(shape, local_size)
                new_shape = padded_shape + (local_stride - padded_shape + local_size) % local_stride
                x = pad_to_shape(x, new_shape, input_axis, padding_values, ratio)
            elif ((shape - local_size) < 0).any() or ((local_stride - shape + local_size) % local_stride).any():
                raise ValueError('Input cannot be patched without remainder.')


            patches = pmap(
                predict,
                divide(x, local_size, local_stride, input_axis, get_boxes=get_boxes),
                *args, **kwargs
            )
            prediction = combine(
                patches, extract(x.shape, input_axis), local_stride, axis,
                combiner=combiner, get_boxes=get_boxes
            )

            if valid:
                prediction = crop_to_shape(prediction, shape, axis, ratio)
            return prediction

        return wrapper

    return decorator
