import numpy as np

import dpipe.medim as medim
from dpipe.config import register


def split(x, xs_padding: np.array, x_patch_sizes: np.array, padding_values):
    xs_parts = []
    for x_padding, x_patch_size in zip(xs_padding, x_patch_sizes):
        complete_patch_size = np.array((x.shape[0], *x_patch_size))
        x_parts = medim.split.divide(x, patch_size=complete_patch_size,
                                     intersection_size=complete_patch_size,
                                     padding_values=padding_values)
        xs_parts.append(x_parts)

    assert all([len(x_parts) == len(xs_parts[0]) for x_parts in xs_parts])
    return xs_parts


def compute_y_shape(x, y_parts):
    x_shape = np.array(x.shape)
    if y_parts[0].ndim == 3:
        return x_shape[-3:]
    else:
        x_shape[0] = y_parts[0].shape[0]
        return x_shape

# FIXME Batches should be put into the model

@register(module_name='patch_3d')
def make_patch_3d_predict(
        model, x_patch_sizes: list, y_patch_size: list, padding_mode: str):
    x_patch_sizes = np.array(x_patch_sizes)
    y_patch_size = np.array(y_patch_size)
    assert np.all(x_patch_sizes % 2 == 1) and np.all(y_patch_size % 2 == 1)

    xs_padding = [np.array((0, *(p // 20)))[:, None].repeat(2, axis=1)
                  for p in (x_patch_sizes - y_patch_size)]

    assert padding_mode == 'min'

    def predict(x):
        padding_values = x.min(axis=(1, 2, 3), keepdims=True)
        xs_parts = split(x, xs_padding, x_patch_sizes, padding_values)
        y_parts = [model.predict(*inputs) for inputs in zip(*xs_parts)]
        y_shape = compute_y_shape(x, y_parts)
        return medim.split.combine(y_parts, y_shape)

    return predict
