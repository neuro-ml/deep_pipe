import unittest

import numpy as np
from .features import get_coordinate_features


class TestCoordinateFeatures(unittest.TestCase):
    def setUp(self):
        self.x_shape = [1, 2, 3, 4]

    def test_small(self):
        shape = (3, 5)
        center_idx = (1, 2)
        patch_size = (1, 3)

        np.testing.assert_equal(get_coordinate_features(shape, center_idx, patch_size),
                                [[[0.5, 0.5, 0.5]],
                                 [[0.3, 0.5, 0.7]]])

    def test_big(self):
        shape = (3, 5)
        center_idx = (1, 2)
        patch_size = (3, 3)

        np.testing.assert_equal(get_coordinate_features(shape, center_idx, patch_size),
                                [
                                    [
                                        [1 / 6, 1 / 6, 1 / 6],
                                        [3 / 6, 3 / 6, 3 / 6],
                                        [5 / 6, 5 / 6, 5 / 6]],

                                    [
                                        [0.3, 0.5, 0.7],
                                        [0.3, 0.5, 0.7],
                                        [0.3, 0.5, 0.7]]
                                 ])
