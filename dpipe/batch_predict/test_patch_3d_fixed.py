import unittest
from itertools import product

import numpy as np
from .patch_3d_fixed import Patch3DFixedPredictor, PatchDividable


class Model:
    def __init__(self, i, y_ndim):
        self.i = i
        self.y_ndim = y_ndim

    def validate(self, *inputs):
        return self.predict(*inputs), 1

    def predict(self, *inputs):
        if self.y_ndim == 4:
            if self.i == 0:
                return inputs[self.i][:, :, 1:-1, 1:-1, 1:-1]
            else:
                return inputs[self.i][:, :, 3:-3, 3:-3, 3:-3]
        else:
            if self.i == 0:
                return inputs[self.i][:, 0, 1:-1, 1:-1, 1:-1]
            else:
                return inputs[self.i][:, 0, 3:-3, 3:-3, 3:-3]


class TestPatch3DFixedPredictor(unittest.TestCase):
    def setUp(self):
        self.x_shape = [3, 20, 30, 40]
        self.x_patch_sizes = [[7, 7, 7], [11, 11, 11]]
        self.y_patch_size = [5, 5, 5]

    def test_patch_3d_predict_call(self):
        Patch3DFixedPredictor(self.x_patch_sizes, self.y_patch_size, padding_mode='min')

    def test_predictor(self):
        for i, y_ndim in product(range(2), (3, 4)):
            with self.subTest(f'{i}, {y_ndim}'):
                predictor = Patch3DFixedPredictor(self.x_patch_sizes, self.y_patch_size, padding_mode='min')
                model = Model(i, y_ndim)

                x = np.random.randn(*self.x_shape)
                y = x[0] if y_ndim == 3 else x
                y_pred, loss = predictor.validate(x, y, validate_fn=model.validate)
                np.testing.assert_equal(y_pred, y)
                self.assertEqual(1, loss)

                x = np.random.randn(*self.x_shape)
                y = x[0] if y_ndim == 3 else x
                y_pred = predictor.predict(x, predict_fn=model.predict)
                np.testing.assert_equal(y_pred, y)

    def test_predictor_big(self):
        for i, y_ndim in product(range(2), (3, 4)):
            self.x_patch_sizes = [[7, 17, 52], [11, 21, 56]]
            self.y_patch_size = [5, 15, 50]

            with self.subTest(f'{i}, {y_ndim}'):
                predictor = Patch3DFixedPredictor(self.x_patch_sizes, self.y_patch_size, padding_mode='min')
                model = Model(i, y_ndim)

                x = np.random.randn(*self.x_shape)
                y = x[0] if y_ndim == 3 else x
                y_pred, loss = predictor.validate(x, y, validate_fn=model.validate)
                np.testing.assert_equal(y_pred, y)
                self.assertEqual(1, loss)

                x = np.random.randn(*self.x_shape)
                y = x[0] if y_ndim == 3 else x
                y_pred = predictor.predict(x, predict_fn=model.predict)
                np.testing.assert_equal(y_pred, y)


class TestPatchDividable(unittest.TestCase):
    def setUp(self):
        self.shape = [3, 11, 11, 11]
        self.batch_predict = PatchDividable(3)

    def test_predict(self):
        x = np.random.rand(*self.shape)
        y = self.batch_predict.predict(x, predict_fn=lambda x: x)
        np.testing.assert_array_equal(x, y)
