import numpy as np

from dpipe.config import register
from dpipe.medim.divide import compute_n_parts_per_axis
from dpipe.medim.utils import pad
from .patch_3d import Patch3DPredictor, spatial_dims


def pad_spatial_size(x, spatial_size: np.array, spatial_dims):
    padding = np.zeros((len(x.shape), 2), dtype=int)
    padding[spatial_dims, 1] = spatial_size - np.array(x.shape)[list(spatial_dims)]
    return pad(x, padding, np.min(x, axis=spatial_dims, keepdims=True))


def slice_spatial_size(x, spatial_size, spatial_dims):
    slices = np.array([slice(None)] * len(x.shape))
    slices[list(spatial_dims)] = list(map(slice, [0] * len(spatial_size), spatial_size))
    return x[tuple(slices)]


def find_fixed_spatial_size(spatial_size, spatial_patch_size):
    return compute_n_parts_per_axis(spatial_size, spatial_patch_size) * spatial_patch_size


@register(module_name='patch_3d_fixed')
class Patch3DFixedPredictor(Patch3DPredictor):
    def validate(self, x, y, *, validate_fn):
        spatial_size = np.array(x.shape)[list(spatial_dims)]
        fixed_spatial_size = find_fixed_spatial_size(spatial_size, self.y_spatial_patch_size)
        x_padded = pad_spatial_size(x, fixed_spatial_size, spatial_dims)
        y_padded = pad_spatial_size(y, fixed_spatial_size, spatial_dims)

        y_pred, loss = super().validate(x_padded, y_padded, validate_fn=validate_fn)
        y_pred = slice_spatial_size(y_pred, spatial_size, spatial_dims)

        return y_pred, loss

    def predict(self, x, *, predict_fn):
        spatial_size = np.array(x.shape)[list(spatial_dims)]
        fixed_spatial_size = find_fixed_spatial_size(spatial_size, self.y_spatial_patch_size)

        x_padded = pad_spatial_size(x, fixed_spatial_size, spatial_dims)

        y_pred = super().predict(x_padded, predict_fn=predict_fn)
        return slice_spatial_size(y_pred, spatial_size, spatial_dims)
