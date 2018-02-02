import unittest

import numpy as np
from .patch import pad


class TestDivide(unittest.TestCase):
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
