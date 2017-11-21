from abc import ABC, abstractmethod

import numpy as np

from dpipe.medim.divide import divide_spatial, combine
from dpipe.medim.shape_utils import compute_shape_from_spatial
from dpipe.config import register
from .base import BatchPredict
from .utils import validate_object, predict_object

spatial_dims = (-3, -2, -1)
zero_spatial_intersection_size = np.zeros(len(spatial_dims), dtype=int)


class BatchPredictorShapeState(BatchPredict):
    @abstractmethod
    def divide_x(self, x):
        pass

    @abstractmethod
    def divide_y(self, y):
        pass

    @abstractmethod
    def combine_y(self, y_parts, x_shape):
        pass

    def validate(self, x, y, *, validate_fn):
        xs_parts = self.divide_x(x)
        y_parts_true = self.divide_y(y)

        assert len(xs_parts[0]) == len(y_parts_true)

        y_preds, loss = validate_object(zip(*xs_parts, y_parts_true), validate_fn=validate_fn)

        return self.combine_y(y_preds, x.shape), loss

    def predict(self, x, *, predict_fn):
        xs_parts = self.divide_x(x)
        y_preds = predict_object(zip(*xs_parts), predict_fn=predict_fn)
        return self.combine_y(y_preds, x.shape)


@register(module_name='patch_3d')
class Patch3DPredictor(BatchPredictorShapeState):
    def __init__(self, x_patch_sizes: list, y_patch_size: list, padding_mode: str = 'min'):
        self.x_patch_sizes = np.array(x_patch_sizes)
        self.y_patch_size = np.array(y_patch_size)

        assert self.x_patch_sizes.shape[1] == len(self.y_patch_size) == 3
        np.testing.assert_equal(np.unique(self.x_patch_sizes % 2), np.unique(self.y_patch_size % 2))

        self.x_spatial_intersection_sizes = (self.x_patch_sizes - self.y_patch_size) // 2

        assert padding_mode == 'min'

    def divide_x(self, x):
        xs_parts = []
        padding_values = x.min(axis=spatial_dims, keepdims=True)
        for x_intersection_size, x_patch_size in zip(self.x_spatial_intersection_sizes, self.x_patch_sizes):
            x_parts = divide_spatial(x, spatial_patch_size=x_patch_size, spatial_dims=list(spatial_dims),
                                     spatial_intersection_size=x_intersection_size, padding_values=padding_values)
            xs_parts.append(x_parts)

        assert all([len(x_parts) == len(xs_parts[0]) for x_parts in xs_parts])
        return xs_parts

    def divide_y(self, y):
        return divide_spatial(y, spatial_patch_size=self.y_patch_size, spatial_dims=list(spatial_dims),
                              spatial_intersection_size=zero_spatial_intersection_size)

    def combine_y(self, y_parts, x_shape):
        complete_shape = compute_shape_from_spatial(y_parts[0].shape, x_shape[-3:], spatial_dims=spatial_dims)
        return combine(y_parts, complete_shape)
