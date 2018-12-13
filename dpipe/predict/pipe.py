"""
Functions that build prediction pipeline.
Each prepares input in certain way, predicts and then fixes output.
"""
import numpy as np

from dpipe.medim.utils import extract_dims, pad
from dpipe.medim.axes import ndim2spatial_axes
from dpipe.medim import grid
from dpipe.medim.checks import check_len
from .functional import trim_spatial_size, pad_to_dividable
from .utils import add_dims


# TODO: add a better description and details about predicts' interface

def add_remove_first_dims(ndim=1, *, predict):
    """Adds `ndim` first dimensions to `inputs`, predicts with `predict` and removes added dimensions.
    Useful to add batch dimension for neural networks."""
    return lambda *inputs: extract_dims(predict(*add_dims(*inputs, ndims=ndim)), ndim=ndim)


def pad_trim_last_dims_to_dividable(divisor, ndim=3, *, predict):
    """Pads `x`'s last `ndim` dimensions to be dividable by ``divisor`, predicts with `predict` and then trims spatial
    dimensions to have the same shape as in `x`."""

    def predictor(x):
        x_padded = pad_to_dividable(x, divisor, ndim=ndim)
        y = predict(x_padded)
        return trim_spatial_size(y, spatial_size=np.array(x.shape[-ndim:]))

    return predictor


def divide_combine_patches(patch_size: np.ndarray, stride: np.ndarray, predict):
    """Split `x` into patches, usually because network can't process whole image because of the memory limitations.
    Network is assumed to return `y_patch_size`, given that `x_patch_size` input is provided.
    So, if `x_patch_size` is more than `y_patch_size`, network is assumed to consume
    border voxels and there will be appropriate padding on the image borders and x patches will
    intersect accordingly.

    Padding is a minimal value across spatial dimensions.

    Gives iterator over patches to the predict function, expects to get the same thing from the `predict` function."""
    check_len(patch_size, stride)
    ndim = len(patch_size)

    def predictor(x):
        padding = np.array((x.ndim - ndim) * [0] + list((patch_size - stride) // 2))[None][[0, 0]].T
        x_padded = pad(x, padding=padding, padding_values=np.min(x, axis=ndim2spatial_axes(ndim), keepdims=True))
        x_parts = grid.divide(x_padded, patch_size=patch_size, stride=stride, valid=False)
        return grid.combine(predict(x_parts), x.shape[-ndim:])

    return predictor


# TODO: test this
def chain_predicts(*predicts):
    """Chain functions from this module together."""

    def make_predictor(predict):
        chain = predict
        for predict_ in reversed(predicts):
            chain = predict_(predict=chain)
        return chain

    return make_predictor
