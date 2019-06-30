import unittest
from functools import partial

import numpy as np

from dpipe.layers import identity
from dpipe.medim.preprocessing import pad, min_max_scale, normalize
from dpipe.medim.utils import filter_mask, apply_along_axes
from dpipe.medim.itertools import zip_equal, flatten, extract, negate_indices, head_tail, peek


class TestPad(unittest.TestCase):
    def test_pad(self):
        x = np.arange(12).reshape((3, 2, 2))
        padding = np.array(((0, 0), (1, 2), (2, 1)))
        padding_values = np.min(x, axis=(1, 2), keepdims=True)

        y = pad(x, padding, padding_values=padding_values)
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


class TestItertools(unittest.TestCase):
    @staticmethod
    def get_map(size):
        return TestItertools.make_iterable(range(size))

    @staticmethod
    def make_iterable(it):
        return map(lambda x: x, it)

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

    def test_head_tail(self):
        for size in range(1, 20):
            it = np.random.randint(1000, size=size).tolist()
            head, tail = head_tail(self.make_iterable(it))
            self.assertEqual(head, it[0])
            self.assertListEqual(list(tail), it[1:])

    def test_peek(self):
        for size in range(1, 20):
            it = np.random.randint(1000, size=size).tolist()
            head, new_it = peek(self.make_iterable(it))
            self.assertEqual(head, it[0])
            self.assertListEqual(list(new_it), it)


class TestApplyAlongAxes(unittest.TestCase):
    def test_apply(self):
        x = np.random.rand(3, 10, 10) * 2 + 3
        np.testing.assert_array_almost_equal(
            apply_along_axes(normalize, x, axes=(1, 2), percentiles=20),
            normalize(x, percentiles=20, axes=0)
        )

        axes = (0, 2)
        y = apply_along_axes(min_max_scale, x, axes)
        np.testing.assert_array_almost_equal(y.max(axes), 1)
        np.testing.assert_array_almost_equal(y.min(axes), 0)

        np.testing.assert_array_almost_equal(apply_along_axes(identity, x, 1), x)
        np.testing.assert_array_almost_equal(apply_along_axes(identity, x, -1), x)
        np.testing.assert_array_almost_equal(apply_along_axes(identity, x, (0, 1)), x)
        np.testing.assert_array_almost_equal(apply_along_axes(identity, x, (0, 2)), x)
