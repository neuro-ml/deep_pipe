import unittest

import numpy as np

from dpipe.medim.utils import build_slices
from .pipe import add_remove_first_dims, pad_trim_last_dims_to_dividable, extract_dims, \
    divide_combine_patches


class TestPipe(unittest.TestCase):
    def setUp(self):
        self.shape = (2, 30, 30, 30)
        self.x = np.random.randn(*self.shape)

    def test_add_first_dims2predict2remove_first_dims(self):
        def predict(a):
            np.testing.assert_equal(extract_dims(a), self.x)
            return a

        np.testing.assert_equal(self.x, add_remove_first_dims(predict=predict)(self.x))

    def test_pad_to_dividable2predict2trim_spatial_size(self):
        def predict(a):
            np.testing.assert_equal(a.shape, [2, 32, 32, 32])
            return a

        np.testing.assert_equal(self.x, pad_trim_last_dims_to_dividable(predict=predict, divisor=4)(self.x))

    def check_divide_to_patches2predict2combine(self, patch_size, stride):
        patch_size = np.array(patch_size)
        stride = np.array(stride)
        diff = (patch_size - stride) // 2

        slices = tuple([..., *build_slices(diff, -diff)])

        def predict(x_parts):
            results = []
            for patch in x_parts:
                np.testing.assert_array_less(np.array(patch.shape[1:]), patch_size + 1)
                results.append(patch[slices])
            return results

        y_pred = divide_combine_patches(patch_size, stride, predict)(self.x)
        np.testing.assert_equal(self.x, y_pred)

    def test_divide_to_patches2predict2combine(self):
        for patch_size, stride in [([46] * 3, [30] * 3), ([20] * 3, [16] * 3)]:
            with self.subTest(f'patch size {patch_size} stride {stride}'):
                self.check_divide_to_patches2predict2combine(patch_size, stride)
