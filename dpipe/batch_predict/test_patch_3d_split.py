import unittest
from itertools import product

import numpy as np
from .patch_3d_split import make_patch_3d_predict


class Model:
    def __init__(self, i, y_ndim):
        self.i = i
        self.y_ndim = y_ndim

        self.x_shape = (3, 120, 120, 120)

    def predict(self, *x_batches):
        if self.y_ndim == 4:
            if self.i == 0:
                return x_batches[self.i][0][:, 1:-1, 1:-1, 1:-1]
            else:
                return x_batches[self.i][0][:, 3:-3, 3:-3, 3:-3]
        else:
            if self.i == 0:
                return x_batches[self.i][0][0, 1:-1, 1:-1, 1:-1]
            else:
                return x_batches[self.i][0][0, 3:-3, 3:-3, 3:-3]


class TestMake3DPredict(unittest.TestCase):
    def setUp(self):
        self.x_shape = [3, 20, 30, 40]
        self.x_patch_sizes = [[7, 7, 7], [11, 11, 11]]
        self.y_patch_size = [5, 5, 5]

    def test_make_patch_3d_predict_call(self):
        make_patch_3d_predict(None, self.x_patch_sizes, self.y_patch_size,
                              padding_mode='min')

    def test_divide_combine(self):
        models = [Model(i, y_ndim) for (i, y_ndim) in product(range(2), (3, 4))]
        for model in models:
            predict = make_patch_3d_predict(
                model, self.x_patch_sizes, self.y_patch_size, padding_mode='min'
            )

            a_1 = np.random.randn(*self.x_shape)
            a_2 = predict(a_1)
            np.testing.assert_equal(a_1, a_2)
