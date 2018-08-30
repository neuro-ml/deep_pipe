from abc import abstractmethod

import numpy as np

from dpipe.medim.utils import extract_dims

from .base import BatchPredict, validate_parts, predict_parts
from .simple import add_dims
from .utils import pad_spatial_size, slice_spatial_size


class BatchPredictorShapeState(BatchPredict):
    """Base class for batch predictors that need x.shape at combine_y stage.
    Works only with networks with one input and output."""

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
        x_parts = self.divide_x(x)
        y_true_parts = self.divide_y(y)

        assert len(x_parts) == len(y_true_parts)

        y_preds, loss = validate_parts(map(add_dims, x_parts, y_true_parts), validate_fn=validate_fn)
        y_preds = list(map(extract_dims, y_preds))

        return self.combine_y(y_preds, x.shape), loss

    def predict(self, x, *, predict_fn):
        x_parts = self.divide_x(x)
        y_preds = predict_parts(map(add_dims, x_parts), predict_fn=predict_fn)
        y_preds = list(map(extract_dims, y_preds))
        return self.combine_y(y_preds, x.shape)


class PatchDividable(BatchPredictorShapeState):
    """Batch Predictor that will pad input, so that spatial dimensions size would be dividable by `divisor`.
    `ndim` reflects number of trailing spatial dimensions."""
    def __init__(self, divisor, ndim=3):
        self.divisor = divisor
        self.spatial_dims = tuple(range(-ndim, 0))

    def prepare_data(self, x):
        spatial_shape = np.array(x.shape)[list(self.spatial_dims)]
        return pad_spatial_size(x, spatial_size=spatial_shape + (self.divisor - spatial_shape) % self.divisor,
                                spatial_dims=self.spatial_dims)

    def divide_x(self, x):
        return [self.prepare_data(x)]

    def divide_y(self, y):
        return [self.prepare_data(y)]

    def combine_y(self, y_parts, x_shape):
        assert len(y_parts) == 1
        y = y_parts[0]
        return slice_spatial_size(y, spatial_size=np.array(x_shape)[list(self.spatial_dims)],
                                  spatial_dims=self.spatial_dims)
