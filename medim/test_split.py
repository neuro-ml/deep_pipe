import unittest

import numpy as np
from .utils import _combine_with_shape, _build_shape, combine, divide


class TestDivide(unittest.TestCase):

    def setUp(self):
        self.shape = (3, 20, 30, 40)
        self.n_parts_per_axis = [1, 2, 3, 4]
        self.padding = [0, 3, 3, 3]
        self.zero_padding = [0] * 4

    def test_divide_n_parts(self):
        x = np.zeros(self.shape)
        x_parts = divide(x, self.padding, self.n_parts_per_axis)
        self.assertEqual(len(x_parts), np.prod(self.n_parts_per_axis))

    def test_build_shape(self):
        x = np.zeros(self.shape)
        x_parts = divide(x, self.zero_padding, self.n_parts_per_axis)
        inferred_shape = _build_shape(x_parts, self.n_parts_per_axis)
        self.assertSequenceEqual(self.shape, inferred_shape)

    def test_combine_inner(self):
        a_1 = np.random.randn(*self.shape)
        a_parts = divide(a_1, self.zero_padding, self.n_parts_per_axis)
        a_2 = _combine_with_shape(a_parts, self.n_parts_per_axis,
                                  self.shape)
        self.assertSequenceEqual(a_1.shape, a_2.shape)
        self.assertTrue(np.all(a_1 == a_2))

    def test_divide_combine(self):
        a_1 = np.random.randn(*self.shape)
        a_parts = divide(a_1, self.zero_padding, self.n_parts_per_axis)
        a_2 = combine(a_parts, self.n_parts_per_axis)
        self.assertSequenceEqual(a_1.shape, a_2.shape)
        self.assertTrue(np.all(a_1 == a_2))