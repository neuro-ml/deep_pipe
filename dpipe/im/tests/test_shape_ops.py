import unittest
from functools import partial

import pytest

import numpy as np
from dpipe.im.shape_ops import *

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

        with pytest.raises(ValueError):
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


class TestCropToBox(unittest.TestCase):
    def test_shape(self):
        for _ in range(100):
            shape = np.random.randint(10, 50, size=2)
            box_shape = np.random.randint(1, 50, size=2)
            box_center = [np.random.randint(s) for s in shape]
            start = box_center - box_shape // 2

            x = np.empty(shape)
            box = np.stack([start, start + box_shape])

            assert (crop_to_box(x, box, padding_values=0).shape == box_shape).all()

    def test_axes(self):
        x = np.random.randint(0, 100, (3, 20, 23))

        assert_eq(x[:, 1:15, 2:14], crop_to_box(x, np.array([[1, 2], [15, 14]])))
        assert_eq(x[:, 1:15, 2:14], crop_to_box(x, np.array([[1, 2], [15, 14]]), axes=[1, 2]))

        assert_eq(
            x[:, 1:, 2:],
            crop_to_box(x, np.array([[1, 2], [40, 33]]), padding_values=0)[:, :19, :21]
        )

        assert_eq(
            x[:, :15, :14],
            crop_to_box(x, np.array([[-10, -5], [15, 14]]), padding_values=0)[:, 10:, 5:]
        )

    def test_raises(self):
        x = np.empty((3, 20, 23))
        with pytest.raises(ValueError):
            crop_to_box(x, np.array([[1], [40]]))

        with pytest.raises(ValueError):
            crop_to_box(x, np.array([[-1], [1]]))
