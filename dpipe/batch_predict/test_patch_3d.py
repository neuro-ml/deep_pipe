import unittest

import numpy as np

from .patch_3d import Patch3DPredictor, Patch3DFixedPredictor


class Model:
    def __init__(self, y_ndim):
        self.y_ndim = y_ndim

    def validate(self, x, y):
        return self.predict(x), 1

    def predict(self, x):
        if self.y_ndim == 3:
            x = x[:, 0]
        return x[..., 3:-3, 3:-3, 3:-3]


class TestPatch3DPredictor(unittest.TestCase):
    def setUp(self):
        self.x_shape = [3, 20, 30, 40]
        self.x_patch_size = [11, 11, 11]
        self.y_patch_size = [5, 5, 5]

    def test_predictor(self):
        for y_ndim in (3, 4):
            with self.subTest(f'{y_ndim}'):
                predictor = Patch3DPredictor(self.x_patch_size, self.y_patch_size)
                model = Model(y_ndim)

                x = np.random.randn(*self.x_shape)
                y = x[0] if y_ndim == 3 else x
                y_pred, loss = predictor.validate(x, y, validate_fn=model.validate)
                np.testing.assert_equal(y_pred, y)
                self.assertEqual(1, loss)

                y_pred = predictor.predict(x, predict_fn=model.predict)
                np.testing.assert_equal(y_pred, y)


class TestPatch3DFixedPredictor(unittest.TestCase):
    def test_predictor(self):
        for y_ndim in (3, 4):
            with self.subTest(f'{y_ndim}'):
                predictor = Patch3DPredictor([11, 11, 11], [5, 5, 5])
                model = Model(y_ndim)

                x = np.random.randn(3, 20, 30, 40)
                y = x[0] if y_ndim == 3 else x
                y_pred, loss = predictor.validate(x, y, validate_fn=model.validate)
                np.testing.assert_equal(y_pred, y)
                self.assertEqual(1, loss)

                y_pred = predictor.predict(x, predict_fn=model.predict)
                np.testing.assert_equal(y_pred, y)

    def test_predictor_big(self):
        x_shape = [3, 20, 30, 40]
        x_patch_size = [11, 21, 56]

        for y_ndim in (3, 4):
            with self.subTest(f'{y_ndim}'):
                predictor = Patch3DFixedPredictor(x_patch_size, [5, 15, 50], padding_mode='min')
                model = Model(y_ndim)

                x = np.random.randn(*x_shape)
                y = x[0] if y_ndim == 3 else x
                y_pred, loss = predictor.validate(x, y, validate_fn=model.validate)
                np.testing.assert_equal(y_pred, y)
                self.assertEqual(1, loss)

                x = np.random.randn(*x_shape)
                y = x[0] if y_ndim == 3 else x
                y_pred = predictor.predict(x, predict_fn=model.predict)
                np.testing.assert_equal(y_pred, y)
