import unittest
from functools import partial

import numpy as np
from .test_utils import get_random_tuple
from .shape_utils import broadcast_shape_nd, compute_shape_from_spatial, broadcast_shape, shape_after_convolution


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


class TestShapeAfterConvolution(unittest.TestCase):
    def test_shape_after_convolution(self):
        import torch

        random_tuple = partial(get_random_tuple, size=2)

        for _ in range(100):
            params = {
                'kernel_size': random_tuple(1, 6),
                'padding': random_tuple(1, 4),
                'stride': random_tuple(1, 4),
                'dilation': random_tuple(1, 4)
            }
            shape = random_tuple(10, 45)
            tensor = torch.empty(1, 1, *shape)
            conv = torch.nn.Conv2d(1, 1, **params)

            with self.subTest(**params, shape=shape):
                try:
                    new_shape = shape_after_convolution(shape, **params)
                except ValueError:
                    # TODO: looks like pytorch doesn't fall if dilation > 1. remove this after the issue is resolved
                    if max(params['dilation']) == 1:
                        with self.assertRaises(RuntimeError):
                            conv(tensor)
                else:
                    self.assertTupleEqual(new_shape, tuple(conv(tensor).shape[2:]))
