import numpy as np

from .base import BatchPredict
from dpipe.config import register


class Simple(BatchPredict):
    def validate(self, x, y, *, validate_fn):
        prediction, loss = validate_fn(x[None], y[None])
        return prediction[0], loss

    def predict(self, x, *, predict_fn):
        return predict_fn(x[None])[0]


class Multiclass(BatchPredict):
    def validate(self, x, y, *, validate_fn):
        prediction, loss = validate_fn(x[None], y[None])
        return np.argmax(prediction), loss

    def predict(self, x, *, predict_fn):
        return np.argmax(predict_fn(x[None]))
