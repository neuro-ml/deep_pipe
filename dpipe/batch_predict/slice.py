import numpy as np

from dpipe.medim.slices import iterate_slices
from .base import BatchPredict
from dpipe.config import register


@register('slice2d')
class Slice2D(BatchPredict):
    """
    A predictor that feeds 2D slices along the OZ axis into the network and stacks the predictions.
    """

    def validate(self, *inputs, validate_fn):
        x, y = inputs
        predicted, losses, weights = [], [], []
        for x_slice, y_slice in iterate_slices(x, y, concatenate=0):
            y_pred, loss = validate_fn(x_slice[None], y_slice[None])

            predicted.extend(y_pred)
            losses.append(loss)
            weights.append(y_pred.size)

        loss = np.average(losses, weights=weights)
        return np.stack(predicted, axis=-1), loss

    def predict(self, *inputs, predict_fn):
        x = inputs[0]
        predicted = []
        for x_slice in iterate_slices(x, concatenate=0):
            y_pred = predict_fn(x_slice[None])
            predicted.extend(y_pred)

        return np.stack(predicted, axis=-1)
