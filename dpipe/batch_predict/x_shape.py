import numpy as np

from .base import DivideCombine, validate_parts, predict_parts


class DivideCombineXShape(DivideCombine):
    def validate(self, x, y, *, validate_fn):
        y_preds, loss = validate_parts(self.val_divide(x, y), validate_fn=validate_fn)
        return self.val_combine(y_preds, np.array(x.shape)), loss

    def predict(self, x, *, predict_fn):
        return self.test_combine(predict_parts(self.test_divide, predict_fn=predict_fn), np.array(x.shape))
