from .base import BatchPredict
from dpipe.config import register


@register()
class Simple(BatchPredict):
    def validate(self, x, y, *, validate_fn):
        return validate_fn(x[None], y[None])

    def predict(self, x, *, predict_fn):
        return predict_fn(x[None])
