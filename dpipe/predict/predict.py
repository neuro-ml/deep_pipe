from functools import partial

import numpy as np

from .pipe import pad_trim_last_dims_to_dividable, add_remove_first_dims, divide_combine_patches, chain_predicts
from .functional import predict_input_parts

predict_network = add_remove_first_dims


def predict_dividable(x, predict, divisor, ndim=3):
    """Use when need to predict with network that requires input to be dividable, usually because of the upsampling.
    """
    return chain_predicts(partial(pad_trim_last_dims_to_dividable, divisor=divisor, ndim=ndim),
                          add_remove_first_dims, predict)(x)


def predict_patches_dividable(x, predict, patch_size: np.ndarray, stride: np.ndarray, divisor: int):
    """Predictor that can be used when need to split image into patches, usually because network can't process
    whole image because of the memory limitations.

    Network is assumed to return `y_patch_size`, given that `x_patch_size` input is provided.

    Often networks require output to be dividable by some value, input will be padded to be dividable by `divisor`.

    So, if `x_patch_size` is more than `y_patch_size`, network is assumed to consume
    border voxels and there will be appropriate padding `padding_mode` on the image borders and x patches will
    intersect accordingly.
    """

    return chain_predicts(partial(pad_trim_last_dims_to_dividable, divisor=divisor, ndim=3),
                          partial(divide_combine_patches, patch_size=patch_size, stride=stride),
                          predict_input_parts, add_remove_first_dims, predict)(x)
