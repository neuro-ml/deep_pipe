import unittest

import numpy as np

from dpipe.medim.combine import combine_grid_patches
from dpipe.medim.divide import grid_patch


class TestCombine(unittest.TestCase):
    def test_combine_grid_patches(self):
        stride = patch_size = [20] * 3
        for _ in range(20):
            shape = np.random.randint(40, 50, size=3)
            with self.subTest(shape=shape):
                x = np.random.randn(1, *shape)
                np.testing.assert_array_almost_equal(
                    x, combine_grid_patches(list(grid_patch(x, patch_size, stride)), shape, stride))
