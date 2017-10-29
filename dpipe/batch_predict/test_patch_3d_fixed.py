import unittest
from itertools import product

import numpy as np
from .patch_3d_fixed import Patch3DFixedPredictor, Patch3DFixedQuantilesPredictor


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


class TestPatch3DFixedQuantilesPredictor(unittest.TestCase):
    def setUp(self):
        self.x_shape = [3, 20, 30, 40]
        self.x_patch_sizes = [[7, 7, 7], [11, 11, 11]]
        self.y_patch_size = [5, 5, 5]

    def test_patch_3d_predict_call(self):
        Patch3DFixedQuantilesPredictor(self.x_patch_sizes, self.y_patch_size, n_quantiles=21, padding_mode='min')

    def test_divide_x(self):
        x = np.arange(300000).reshape((3, 10, 100, 100))
        batch_predict = Patch3DFixedQuantilesPredictor(self.x_patch_sizes, self.y_patch_size, n_quantiles=21,
                                                       padding_mode='min')
        xs_parts = batch_predict.divide_x(x)
        assert len(xs_parts[-1]) == len(xs_parts[0])
