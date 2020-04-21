import unittest

import numpy as np

from dpipe.im.grid import get_boxes, combine, divide


class TestGrid(unittest.TestCase):
    def setUp(self):
        self.x_shape = np.array((3, 42, 58, 74))
        self.patch_size = np.array((3, 10, 10, 10))
        self.stride = np.array((3, 8, 8, 8))
        self.spatial_patch_size = self.patch_size[1:]

    def test_divide_n_parts(self):
        x = np.zeros(self.x_shape)
        x_parts = list(divide(x, self.patch_size, self.stride))
        self.assertEqual(len(x_parts), np.prod((1, 5, 7, 9)))

    def test_divide_combine(self):
        result_shape = np.array((3, 40, 56, 72))
        slices = tuple([slice(None)] + [slice(1, -1)] * 3)
        a1 = np.random.randn(*self.x_shape)
        a_parts = divide(a1, self.patch_size, self.stride)
        a2 = combine([a[slices] for a in a_parts], result_shape, self.stride)
        np.testing.assert_equal(a1[slices], a2)

    def test_get_boxes_grid(self):
        shape = np.array((10, 15, 27, 3))
        box_size = (3, 3, 3, 3)
        grid = np.stack(list(get_boxes(shape, box_size, stride=1)))
        start, stop = grid[:, 0], grid[:, 1]

        self.assertEqual(np.prod(shape - box_size + 1), len(grid))
        self.assertTrue((start >= 0).all())
        self.assertTrue((stop <= shape).all())
        self.assertTrue((start + box_size == stop).all())

    def test_combine_int(self):
        patch_size = np.array([20] * 3, int)
        stride = patch_size // 2
        shape = [45, 43, 48]

        x = np.random.randint(0, 100, size=(1, *shape))
        np.testing.assert_array_almost_equal(x, combine(divide(x, patch_size, stride), shape, stride))

    def test_combine_grid_patches(self):
        stride = patch_size = [20] * 3
        for _ in range(20):
            shape = np.random.randint(40, 50, size=3)
            with self.subTest(shape=shape):
                x = np.random.randn(1, *shape)
                np.testing.assert_array_almost_equal(x, combine(divide(x, patch_size, stride), shape, stride))
