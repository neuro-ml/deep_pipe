import numpy as np

from dpipe.medim.axes import broadcast_to_axes, AxesLike, AxesParams
from dpipe.medim.grid import divide, combine
from dpipe.medim.itertools import extract
from dpipe.medim.shape_ops import pad_to_shape, crop_to_shape
from dpipe.medim.utils import extract_dims
from dpipe.predict.utils import add_dims
from dpipe.predict.functional import trim_spatial_size, pad_to_dividable


def preprocess(func, *args, **kwargs):
    """
    Applies function ``func`` with given parameters before making a prediction.

    Examples
    --------
        >>> from dpipe.medim.shape_ops import pad
        >>> from dpipe.predict.shape import preprocess
        >>>
        >>> @preprocess(pad, padding=[[24] * 2] * 3, padding_values=np.min)
        >>> def predict(x):
        >>>     return model.do_inf_step(x)
        performs spatial padding before prediction.
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

    See Also
    --------
    `preprocess`
    """
    def decorator(predict):
        def wrapper(x):
            x = predict(x)
            x = func(x, *args, **kwargs)
            return x

        return wrapper

    return decorator


def add_extract_dims(n_add: int = 1, n_extract: int = None):
    """
    Adds ``n_add`` dimensions before a prediction and extracts ``n_extract`` dimensions after this prediction.

    Parameters
    ----------
    n_add: int
        number of dimensions to add.
    n_extract: int, None, optional
        number of dimensions to extract. If ``None``, extracts the same number of dimensions as were added (``n_add``).
    """
    if n_extract is None:
        n_extract = n_add

    def decorator(predict):
        def wrapper(x):
            x = add_dims(x, ndims=n_add)[0]
            x = predict(x)
            return extract_dims(x, n_extract)

        return wrapper

    return decorator


def dividable_shape(divisor, ndim: int = 3):
    """
    Pads an incoming array to be dividable by ``divisor`` in last ``ndim`` dimensions. Afterwards the padding is
    removed.

    Parameters
    ----------
    divisor: int
        a value an incoming array should be divided by.
    ndim: int
        a number of last array dimensions which will be padded.
    """
    def decorator(predict):
        def wrapper(x):
            x_padded = pad_to_dividable(x, divisor, ndim=ndim)
            y = predict(x_padded)
            return trim_spatial_size(y, spatial_size=np.array(x.shape[-ndim:]))

        return wrapper

    return decorator


def patches_grid(patch_size: AxesLike, stride: AxesLike, axes: AxesLike = None, padding_values: AxesParams = 0,
                 ratio: AxesParams = 0.5):
    """
    Divide an incoming array into patches of corresponding ``patch_size`` and ``stride`` and then combine
    predicted patches by averaging the overlapping regions.

    If ``padding_values`` is not None, the array will be padded to an appropriate shape to make a valid division.
    Afterwards the padding is removed.

    See Also
    --------
    `grid.divide` `grid.combine` `pad_to_shape`
    """
    axes, path_size, stride = broadcast_to_axes(axes, patch_size, stride)
    valid = padding_values is not None

    def decorator(predict):
        def wrapper(x):
            if valid:
                shape = np.array(x.shape)[list(axes)]
                new_shape = shape + (stride - shape + patch_size) % stride
                x = pad_to_shape(x, new_shape, axes, padding_values, ratio)

            patches = map(predict, divide(x, patch_size, stride, axes))
            prediction = combine(patches, extract(x.shape, axes), stride, axes)

            if valid:
                prediction = crop_to_shape(prediction, shape, axes, ratio)
            return prediction

        return wrapper

    return decorator
