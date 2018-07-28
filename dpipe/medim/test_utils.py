import unittest

import numpy as np
from .utils import pad, zip_equal


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

        self.assertEqual(len(list(zip_equal())), 0)
