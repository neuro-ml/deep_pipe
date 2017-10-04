import unittest

import numpy as np
from .split import compute_n_parts_per_axis, divide_no_padding, divide, combine


class TestSplit(unittest.TestCase):
    def setUp(self):
        self.x_shape = np.array((3, 20, 30, 40))
        self.patch_size = np.array((3, 10, 10, 10))
        self.true_n_parts_per_axis = np.array((1, 2, 3, 4))
        self.intersection_size = np.array((0, 3, 3, 3))
        self.slices = [slice(None)] + [slice(3, -3)] * 3
        self.zero_intersection_size = np.array([0] * 4)

    def test_compute_n_parts_per_axis(self):
        np.testing.assert_equal(
            compute_n_parts_per_axis(self.x_shape, self.patch_size),
            self.true_n_parts_per_axis
        )

    def test_divide_no_padding_n_parts(self):
        x = np.zeros(self.x_shape)
        x_parts = divide_no_padding(x, self.patch_size, self.intersection_size)
        self.assertEqual(len(x_parts), np.prod(self.true_n_parts_per_axis))

    def test_divide_padded_combine(self):
        a_1 = np.random.randn(*self.x_shape)
        a_parts = divide_no_padding(a_1, self.patch_size,
                                    self.zero_intersection_size)
        a_2 = combine(a_parts, self.x_shape)
        np.testing.assert_equal(a_1, a_2)

    def test_divide_combine(self):
        a_1 = np.random.randn(*self.x_shape)
        a_parts = divide(a_1, self.patch_size, self.intersection_size,
                         padding_values=np.min(a_1, keepdims=True))
        a_parts = [a_part[self.slices] for a_part in a_parts]
        a_2 = combine(a_parts, self.x_shape)
        np.testing.assert_equal(a_1, a_2)
