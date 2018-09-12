import numpy as np

from dpipe.medim.utils import pad, build_slices, ndim2spatial_axes


def predict_input_parts(batch_iterator, *, predict):
    return map(predict, batch_iterator)


def predict_inputs_parts(batch_iterator, *, predict):
    return predict_input_parts(batch_iterator, predict=lambda inputs: predict(*inputs))


def pad_spatial_size(x, spatial_size: np.array):
    ndim = len(spatial_size)
    padding = np.zeros((len(x.shape), 2), dtype=int)
    padding[-ndim:, 1] = spatial_size - x.shape[-ndim:]
    return pad(x, padding, np.min(x, axis=ndim2spatial_axes(ndim), keepdims=True))


def trim_spatial_size(x, spatial_size):
    return x[(..., *build_slices(spatial_size))]


def pad_to_dividable(x, divisor, ndim=3):
    """Pads `x`'s last `ndim` dimensions to be dividable by `divisor` and returns it."""
    spatial_shape = np.array(x.shape[-ndim:])
    return pad_spatial_size(x, spatial_size=spatial_shape + (divisor - spatial_shape) % divisor)
