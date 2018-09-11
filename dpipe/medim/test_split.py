import unittest

import numpy as np
from .divide import divide, combine


class TestSplit(unittest.TestCase):
    def setUp(self):
        self.x_shape = np.array((3, 42, 58, 74))
        self.result_shape = np.array((3, 40, 56, 72))
        self.patch_size = np.array((3, 10, 10, 10))
        self.stride = np.array((3, 8, 8, 8))
        self.true_n_parts_per_axis = np.array((1, 5, 7, 9))
        self.slices = tuple([slice(None)] + [slice(1, -1)] * 3)

        self.spatial_patch_size = self.patch_size[1:]

    def test_divide__n_parts(self):
        x = np.zeros(self.x_shape)
        x_parts = divide(x, self.patch_size, self.stride)
        self.assertEqual(len(x_parts), np.prod(self.true_n_parts_per_axis))

    def test_divide_combine(self):
        a1 = np.random.randn(*self.x_shape)
        a_parts = divide(a1, self.patch_size, self.stride)
        a2 = combine([a[self.slices] for a in a_parts], self.result_shape)
        np.testing.assert_equal(a1[self.slices], a2)
