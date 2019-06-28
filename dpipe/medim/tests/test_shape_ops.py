import unittest
from functools import partial

import numpy as np
from dpipe.medim.shape_ops import *

assert_eq = np.testing.assert_array_equal


class TestPad(unittest.TestCase):
    def test_broadcasting(self):
        x = np.random.randint(0, 100, (3, 20, 23))
        main = pad(x, [[3, 3], [3, 3], [3, 3]])

        assert_eq(x, main[3:-3, 3:-3, 3:-3])
        assert_eq(main, pad(x, [3, 3, 3]))
        assert_eq(main, pad(x, 3, axes=[0, 1, 2]))
        assert_eq(main, pad(x, [3], axes=[0, 1, 2]))
        assert_eq(main, pad(x, [[3]], axes=[0, 1, 2]))
        assert_eq(main, pad(x, [[3, 3]], axes=[0, 1, 2]))
        assert_eq(main, pad(x, [[3], [3], [3]], axes=[0, 1, 2]))

        assert_eq(
            pad(x, 3, axes=[0, 1]),
            pad(x, [[3, 3], [3, 3], [0, 0]])
        )
        assert_eq(
            pad(x, [2, 4, 3]),
            pad(x, [[2, 2], [4, 4], [3, 3]])
        )
        p = pad(x, [[1, 2], [3, 4], [5, 6]])
        assert_eq(x, p[1:-2, 3:-4, 5:-6])

        p = pad(x, [[1, 2], [3, 4]], axes=[0, 2])
        assert_eq(x, p[1:-2, :, 3:-4])

        p = pad(x, [[1, 2], [3, 4]], axes=[2, 0])
        assert_eq(x, p[3:-4:, :, 1:-2])

        with self.assertRaises(ValueError):
            pad(x, [1, 2], axes=-1)

    def test_padding_values(self):
        x = np.array([
            [0, 0, 0, 0],
            [0, 0, 0, 0],
            [0, 0, 0, 0],
        ], dtype=int)

        p = pad(x, [1, 1], padding_values=1)
        assert_eq(p, [
            [1, 1, 1, 1, 1, 1],
            [1, 0, 0, 0, 0, 1],
            [1, 0, 0, 0, 0, 1],
            [1, 0, 0, 0, 0, 1],
            [1, 1, 1, 1, 1, 1],
        ])

        x = np.random.randint(0, 100, (3, 20, 23))
        assert_eq(
            pad(x, [1, 1], padding_values=x.min()),
            pad(x, [1, 1], padding_values=np.min),
        )
        assert_eq(
            pad(x, [1, 1], padding_values=x.min(axis=(1, 2), keepdims=True)),
            pad(x, [1, 1], padding_values=partial(np.min, axis=(1, 2), keepdims=True)),
        )
