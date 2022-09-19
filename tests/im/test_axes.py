import unittest

import pytest

from dpipe.im.axes import *


class TextBroadcastToAxes(unittest.TestCase):
    def test_exceptions(self):
        with self.assertRaises(ValueError):
            broadcast_to_axis(None, [1], [1, 2], [1, 2, 3])
        with self.assertRaises(ValueError):
            broadcast_to_axis([1, 2, 3], [1], [1, 2])
        with self.assertRaises(ValueError):
            broadcast_to_axis(None)


def test_broadcast_none():
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
        with pytest.raises(ValueError):
            np.testing.assert_array_equal(o, broadcast_to_axis(None, *i)[1:])
