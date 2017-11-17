import unittest

import numpy as np
from .divide import compute_n_parts_per_axis, divide_no_padding, divide, \
    divide_spatial, combine


class TestSplit(unittest.TestCase):
    def setUp(self):
        self.x_shape = np.array((3, 40, 52, 72))
        self.patch_size = np.array((3, 8, 8, 8))
        self.true_n_parts_per_axis = np.array((1, 5, 7, 9))
        self.intersection_size = np.array((0, 1, 1, 1))
        self.slices = [slice(None)] + [slice(1, -1)] * 3
        self.zero_intersection_size = np.array([0] * 4)

        self.spatial_patch_size = self.patch_size[1:]
        self.spatial_intersection_size = self.intersection_size[1:]
        self.zero_spatial_intersection_size = self.zero_intersection_size[1:]
        self.spatial_dims = [-3, -2, -1]

    def test_compute_n_parts_per_axis(self):
        np.testing.assert_equal(
            compute_n_parts_per_axis(self.x_shape, self.patch_size),
            self.true_n_parts_per_axis
        )

    def test_divide_no_padding_n_parts(self):
        x = np.zeros(self.x_shape)
        x_parts = divide_no_padding(x, self.patch_size,
                                    self.zero_intersection_size)
        self.assertEqual(len(x_parts), np.prod(self.true_n_parts_per_axis))

    def test_divide_no_padding_combine(self):
        a1 = np.random.randn(*self.x_shape)
        a_parts = divide(a1, self.patch_size, self.zero_intersection_size)
        a2 = combine(a_parts, self.x_shape)
        np.testing.assert_equal(a1, a2)

    def test_divide(self):
        a1 = np.arange(np.prod(self.x_shape)).reshape(self.x_shape)
        a1_parts = divide(a1, patch_size=self.patch_size,
                          intersection_size=self.intersection_size,
                          padding_values=np.min(a1, keepdims=True))
        a1_parts = [a_part[self.slices] for a_part in a1_parts]

        a2_parts = divide(
            a1, patch_size=self.patch_size - 2 * self.intersection_size,
            intersection_size=self.zero_intersection_size,
            padding_values=np.min(a1, keepdims=True)
        )

        self.assertEqual(len(a1_parts), len(a2_parts))
        for x, y in zip(a1_parts, a2_parts):
            np.testing.assert_equal(x, y)

    def test_divide_combine(self):
        a_1 = np.random.randn(*self.x_shape)
        a_parts = divide(a_1, patch_size=self.patch_size,
                         intersection_size=self.intersection_size,
                         padding_values=np.min(a_1, keepdims=True))
        a_parts = [a_part[self.slices] for a_part in a_parts]
        a_2 = combine(a_parts, self.x_shape)
        np.testing.assert_equal(a_1, a_2)

    def test_divide_spatial_combine(self):
        a_1 = np.random.randn(*self.x_shape)
        a_parts = divide_spatial(
            a_1, spatial_patch_size=self.spatial_patch_size,
            spatial_intersection_size=self.spatial_intersection_size,
            padding_values=np.min(a_1, keepdims=True),
            spatial_dims=self.spatial_dims
        )
        a_parts = [a_part[self.slices] for a_part in a_parts]
        a_2 = combine(a_parts, self.x_shape)
        np.testing.assert_equal(a_1, a_2)
