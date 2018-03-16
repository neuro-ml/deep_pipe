from itertools import starmap

from .base import BatchPredict


class DivideCombine(BatchPredict):
    def __init__(self, val_divide, val_combine, test_divide=None, test_combine=None):
        self.val_divide, self.test_divide = val_divide, test_divide or val_divide
        self.val_combine, self.test_combine = val_combine, test_combine or val_combine

    def validate(self, x, y, *, validate_fn):
        def separate_prediction():
            nonlocal size, mean_loss
            for prediction, loss in starmap(validate_fn, self.val_divide(x, y)):
                size += loss.size
                mean_loss += loss.sum()
                yield prediction

        mean_loss = size = 0
        return self.val_combine(separate_prediction()), mean_loss / size

    def predict(self, x, *, predict_fn):
        return self.test_combine(map(predict_fn, self.test_divide(x)))
