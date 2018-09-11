import unittest

import numpy as np

from dpipe.medim.utils import build_slices
from .predict import predict_dividable, predict_patches_dividable


class TestPredict(unittest.TestCase):
    def setUp(self):
        self.shape = (2, 30, 30, 30)
        self.x = np.random.randn(*self.shape)

    def test_predict_dividable(self):
        def predict(a):
            np.testing.assert_equal(a.shape, [1, 2, 32, 32, 32])
            return a

        np.testing.assert_equal(self.x, predict_dividable(self.x, predict=predict, divisor=4))

    def check_predict_patch_dividable(self, patch_size, stride):
        patch_size = np.array(patch_size)
        stride = np.array(stride)
        diff = (patch_size - stride) // 2
        divisor = 4
        ndim = 3

        slices = tuple([..., *build_slices(diff, -diff)])

        def predict(x_parts):
            results = []
            for patch in x_parts:
                np.testing.assert_equal(np.array(patch.shape[-ndim:]) % divisor, 0)
                np.testing.assert_array_less(np.array(patch.shape[-ndim:]), patch_size + 1)
                results.append(patch[slices])
            return results

        y_pred = predict_patches_dividable(self.x, predict, patch_size=patch_size, stride=stride, divisor=divisor)
        np.testing.assert_equal(self.x, y_pred)

    def test_divide_to_patches2predict2combine(self):
        for patch_size, stride in [([64] * 3, [32] * 3), ([20] * 3, [16] * 3)]:
            with self.subTest(f'patch size {patch_size} stride {stride}'):
                self.check_predict_patch_dividable(patch_size, stride)