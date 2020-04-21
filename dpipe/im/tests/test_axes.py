import unittest

from dpipe.im.axes import *


class TestExpandAxes(unittest.TestCase):
    def test_exceptions(self):
        with self.assertRaises(ValueError):
            expand_axes([1.2, 1, 2, 3], 1)

        with self.assertRaises(ValueError):
            expand_axes([1, 1, 2, 3], 1)

        with self.assertRaises(ValueError):
            expand_axes([[1], [2]], 1)

    def test_none(self):
        self.assertTupleEqual((-3, -2, -1), expand_axes(None, [1, 2, 3]))
        self.assertTupleEqual((-2, -1), expand_axes(None, [1, 1]))
        self.assertTupleEqual((-1,), expand_axes(None, [1]))
        self.assertTupleEqual((-1,), expand_axes(None, 1))

    def test_scalar(self):
        self.assertTupleEqual((1,), expand_axes(1, 1))
        self.assertTupleEqual((-1,), expand_axes(None, 1))


class TextBroadcastToAxes(unittest.TestCase):
    def test_exceptions(self):
        with self.assertRaises(ValueError):
            broadcast_to_axes(None, [1], [1, 2], [1, 2, 3])
        with self.assertRaises(ValueError):
            broadcast_to_axes([1, 2, 3], [1], [1, 2])
        with self.assertRaises(ValueError):
            broadcast_to_axes(None)

    def test_none(self):
        inputs = [
            [1],
            [1, 2, 3],
            [[1], [2], 3],
            [1, [1, 2], [3]]
        ]
        outputs = [
            [[1]],
            [[1], [2], [3]],
            [[1], [2], [3]],
            [[1, 1], [1, 2], [3, 3]]
        ]

        for i, o in zip(inputs, outputs):
            np.testing.assert_array_equal(o, broadcast_to_axes(None, *i)[1:])
