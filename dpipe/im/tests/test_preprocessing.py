import unittest
import numpy as np

from dpipe.im.preprocessing import *


class TestPrep(unittest.TestCase):
    def setUp(self):
        self.x = np.random.rand(3, 10, 10) * 2 + 3

    def test_normalize_image(self):
        x = normalize(self.x)
        np.testing.assert_almost_equal(0, x.mean())
        np.testing.assert_almost_equal(1, x.std())

        x = normalize(self.x, mean=False)
        np.testing.assert_almost_equal(1, x.std())

        x = normalize(self.x, std=False)
        np.testing.assert_almost_equal(0, x.mean())

        y = np.array([-100, 1, 2, 1000])
        x = normalize(y, percentiles=25)
        np.testing.assert_equal(x, (y - 1.5) * 2)
        np.testing.assert_equal(
            normalize(y, percentiles=25),
            normalize(y, percentiles=[25, 75]),
        )

    def test_normalize_multichannel_image(self):
        x = normalize(self.x, axes=0)
        np.testing.assert_almost_equal(0, x.mean(axis=(1, 2)))
        np.testing.assert_almost_equal(1, x.std(axis=(1, 2)))
