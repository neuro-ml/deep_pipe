import unittest

import numpy as np
from .shape_utils import broadcast_shape_nd, compute_shape_from_spatial, \
    broadcast_shape


class TestBroadcastShapeND(unittest.TestCase):
    def setUp(self):
        self.x_shape = [1, 2, 3, 4]

    def test_broadcast_shape_nd(self):
        np.testing.assert_equal(broadcast_shape_nd(self.x_shape, 5),
                                [1] + self.x_shape)

    def test_broadcast_shape_nd_same(self):
        np.testing.assert_equal(broadcast_shape_nd([1] + self.x_shape, 5),
                                [1] + self.x_shape)

    def test_broadcast_shape_nd_value_error(self):
        with self.assertRaises(ValueError):
            broadcast_shape_nd([1] + self.x_shape, 4)


class TestBroadcastShape(unittest.TestCase):
    def test_broadcast_shape(self):
        np.testing.assert_equal(broadcast_shape([8, 1, 6, 1], [7, 1, 5]),
                                [8, 7, 6, 5])
        np.testing.assert_equal(broadcast_shape([8, 1, 6, 1], [1]),
                                [8, 1, 6, 1])

    def test_broadcast_shape_value_error(self):
        with self.assertRaises(ValueError):
            broadcast_shape([8, 1, 6, 1], [7, 2, 5])


class TestComputeShapeFromSpatial(unittest.TestCase):
    def test_compute_shape_from_spatial(self):
        complete_shape = compute_shape_from_spatial(
            [4, 240, 245, 255], [12, 12, 12], [-3, -2, -1]
        )
        np.testing.assert_equal(complete_shape, [4, 12, 12, 12])
