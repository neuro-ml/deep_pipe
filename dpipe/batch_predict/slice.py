import numpy as np

from dpipe.medim.slices import iterate_slices
from .base import BatchPredict


class Slice2D(BatchPredict):
    """
    Breaks the incoming 3D image into slices along the OZ axis
    and feeds them into the network.
    """

    def validate(self, x, y, validate_fn):
        predicted, losses, weights = [], [], []
        for x_slice, y_slice in iterate_slices(x, y, concatenate=0):
            y_pred, loss = validate_fn(x_slice[None], y_slice[None])

            predicted.extend(y_pred)
            losses.append(loss)
            weights.append(y_pred.size)

        loss = np.average(losses, weights=weights)
        return np.stack(predicted, axis=-1), loss

    def predict(self, x, predict_fn):
        predicted = []
        for x_slice in iterate_slices(x, concatenate=0):
            y_pred = predict_fn(x_slice[None])
            predicted.extend(y_pred)

        return np.stack(predicted, axis=-1)
