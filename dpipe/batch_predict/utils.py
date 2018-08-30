import numpy as np

from dpipe.medim.divide import compute_n_parts_per_axis
from dpipe.medim.utils import pad, extract_dims


def pad_spatial_size(x, spatial_size: np.array, spatial_dims):
    padding = np.zeros((len(x.shape), 2), dtype=int)
    padding[spatial_dims, 1] = spatial_size - np.array(x.shape)[list(spatial_dims)]
    return pad(x, padding, np.min(x, axis=spatial_dims, keepdims=True))


def find_fixed_spatial_size(spatial_size, spatial_patch_size):
    return compute_n_parts_per_axis(spatial_size, spatial_patch_size) * spatial_patch_size


def slice_spatial_size(x, spatial_size, spatial_dims):
    slices = np.array([slice(None)] * len(x.shape))
    slices[list(spatial_dims)] = list(map(slice, [0] * len(spatial_size), spatial_size))
    return x[tuple(slices)]
