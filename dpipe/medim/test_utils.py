import unittest
from functools import partial

import numpy as np

from dpipe.medim.preprocessing import normalize_multichannel_image, normalize_image
from .utils import pad, filter_mask, apply_along_axes, scale
from .itertools import zip_equal, flatten, extract, negate_indices


def get_random_tuple(low, high, size):
    return tuple(np.random.randint(low, high, size=size, dtype=int))


class TestPad(unittest.TestCase):
    def test_pad(self):
        x = np.arange(12).reshape((3, 2, 2))
        padding = np.array(((0, 0), (1, 2), (2, 1)))
        padding_values = np.min(x, axis=(1, 2), keepdims=True)

        y = pad(x, padding, padding_values)
        np.testing.assert_array_equal(y, np.array([
            [
                [0, 0, 0, 0, 0],
                [0, 0, 0, 1, 0],
                [0, 0, 2, 3, 0],
                [0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0],
            ],
            [
                [4, 4, 4, 4, 4],
                [4, 4, 4, 5, 4],
                [4, 4, 6, 7, 4],
                [4, 4, 4, 4, 4],
                [4, 4, 4, 4, 4],
            ],
            [
                [8, 8, 8, 8, 8],
                [8, 8, 8, 9, 8],
                [8, 8, 10, 11, 8],
                [8, 8, 8, 8, 8],
                [8, 8, 8, 8, 8],
            ],
        ]))


class TestUtils(unittest.TestCase):
    @staticmethod
    def get_map(size):
        return map(lambda x: x, range(size))

    def test_zip_equal_raises(self):
        for args in [[range(5), range(6)], [self.get_map(5), self.get_map(6)], [range(7), self.get_map(6)],
                     [self.get_map(6), range(5)], [self.get_map(6), range(5), range(7)]]:
            with self.subTest(args=args), self.assertRaises(ValueError):
                list(zip_equal(*args))

    def test_zip_equal(self):
        for args in [[range(5), range(5)], [self.get_map(5), self.get_map(5)],
                     [range(5), self.get_map(5)], [self.get_map(5), range(5)]]:
            with self.subTest(args=args):
                self.assertEqual(len(list(zip_equal(*args))), 5)

        for args in [[], [range(5)], [range(5), range(5)], [range(5), range(5), range(5)]]:
            with self.subTest(args=args):
                self.assertListEqual(list(zip_equal(*args)), list(zip(*args)))

    def test_flatten(self):
        self.assertListEqual(flatten([1, [2, 3], [[4]]]), [1, 2, 3, 4])
        self.assertListEqual(flatten([1, (2, 3), [[4]]]), [1, (2, 3), 4])
        self.assertListEqual(flatten([1, (2, 3), [[4]]], iterable_types=(list, tuple)), [1, 2, 3, 4])
        self.assertListEqual(flatten(1, iterable_types=list), [1])

    def test_extract(self):
        idx = [2, 5, 3, 9, 0]
        self.assertListEqual(extract(range(15), idx), idx)

    def test_filter_mask(self):
        # TODO: def randomized_test
        mask = np.random.randint(2, size=15, dtype=bool)
        values, = np.where(mask)
        np.testing.assert_array_equal(list(filter_mask(range(15), mask)), values)

    def test_negate_indices(self):
        idx = [2, 5, 3, 9, 0]
        other = [1, 4, 6, 7, 8, 10, 11, 12]
        np.testing.assert_array_equal(negate_indices(idx, 13), other)


class TestApplyAlongAxes(unittest.TestCase):
    def test_apply(self):
        x = np.random.rand(3, 10, 10) * 2 + 3
        np.testing.assert_array_almost_equal(
            apply_along_axes(partial(normalize_image, drop_percentile=20), x, (1, 2)),
            normalize_multichannel_image(x, drop_percentile=20)
        )

        axes = (0, 2)
        y = apply_along_axes(scale, x, axes)
        np.testing.assert_array_almost_equal(y.max(axes), 1)
        np.testing.assert_array_almost_equal(y.min(axes), 0)
