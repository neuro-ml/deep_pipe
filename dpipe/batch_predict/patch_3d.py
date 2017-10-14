import numpy as np

from dpipe.medim.divide import divide_spatial, combine
from dpipe.medim.shape_utils import compute_shape_from_spatial
from dpipe.config import register
from .base import BatchPredict
from .utils import validate_object, predict_object

spatial_dims = (-3, -2, -1)
zero_spatial_intersection_size = np.zeros(len(spatial_dims), dtype=int)


def divide_x(x, x_intersection_sizes: np.array, x_patch_sizes: np.array, padding_values):
    xs_parts = []
    for x_intersection_size, x_patch_size in zip(x_intersection_sizes, x_patch_sizes):
        x_parts = divide_spatial(x, spatial_patch_size=x_patch_size, spatial_dims=list(spatial_dims),
                                 spatial_intersection_size=x_intersection_size, padding_values=padding_values)
        xs_parts.append(x_parts)

    assert all([len(x_parts) == len(xs_parts[0]) for x_parts in xs_parts])
    return xs_parts


def divide_y(y, y_spatial_patch_size):
    return divide_spatial(y, spatial_patch_size=y_spatial_patch_size, spatial_dims=list(spatial_dims),
                          spatial_intersection_size=zero_spatial_intersection_size)


def combine_y(y_preds, x_shape):
    complete_shape = compute_shape_from_spatial(y_preds[0].shape, x_shape[-3:], spatial_dims=spatial_dims)
    return combine(y_preds, complete_shape)


@register(module_name='patch_3d')
class Patch3DPredictor(BatchPredict):
    def __init__(self, x_spatial_patch_sizes: list, y_spatial_patch_size: list, padding_mode: str):
        self.x_spatial_patch_sizes = np.array(x_spatial_patch_sizes)
        self.y_spatial_patch_size = np.array(y_spatial_patch_size)

        assert self.x_spatial_patch_sizes.shape[1] == len(self.y_spatial_patch_size) == 3
        np.testing.assert_equal(np.unique(self.x_spatial_patch_sizes % 2), np.unique(self.y_spatial_patch_size % 2))

        self.x_spatial_intersection_sizes = (self.x_spatial_patch_sizes - self.y_spatial_patch_size) // 2

        assert padding_mode == 'min'

    def validate(self, x, y, *, validate_fn):
        xs_parts = divide_x(x, self.x_spatial_intersection_sizes, self.x_spatial_patch_sizes,
                            x.min(axis=spatial_dims, keepdims=True))
        y_parts_true = divide_y(y, self.y_spatial_patch_size)

        y_preds, loss = validate_object(zip(*xs_parts, y_parts_true), validate_fn=validate_fn)

        return combine_y(y_preds, x.shape), loss

    def predict(self, x, *, predict_fn):
        xs_parts = divide_x(x, self.x_spatial_intersection_sizes, self.x_spatial_patch_sizes,
                            x.min(axis=spatial_dims, keepdims=True))

        y_preds = predict_object(zip(*xs_parts), predict_fn=predict_fn)
        return combine_y(y_preds, x.shape)
