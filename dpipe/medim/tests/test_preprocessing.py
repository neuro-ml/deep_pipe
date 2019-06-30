import unittest

import numpy as np
from dpipe.medim import prep
from dpipe.medim.preprocessing import *


class TestPrep(unittest.TestCase):
    def setUp(self):
        self.x = np.random.rand(3, 10, 10) * 2 + 3

    def _test_to_shape(self, func, shape, bad_shape):
        self.assertTupleEqual(func(self.x, shape).shape, shape)
        with self.assertRaises(ValueError):
            func(self.x, bad_shape)

    def test_scale_to_shape(self):
        shape = (3, 4, 15)
        self.assertTupleEqual(prep.zoom_to_shape(self.x, shape).shape, shape)
        self.assertTupleEqual(prep.zoom_to_shape(self.x, shape[::-1]).shape, shape[::-1])

    def test_pad_to_shape(self):
        self._test_to_shape(prep.pad_to_shape, (3, 15, 16), (3, 4, 10))

    def test_slice_to_shape(self):
        self._test_to_shape(prep.crop_to_shape, (3, 4, 8), (3, 15, 10))

    def test_scale(self):
        self.assertTupleEqual(prep.zoom(self.x, (3, 4, 15)).shape, (9, 40, 150))

        self.assertTupleEqual(prep.zoom(self.x, (4, 3)).shape, (3, 40, 30))

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
            normalize(y, percentiles=[25, 25]),
        )

    def test_normalize_multichannel_image(self):
        x = normalize(self.x, axes=0)
        np.testing.assert_almost_equal(0, x.mean(axis=(1, 2)))
        np.testing.assert_almost_equal(1, x.std(axis=(1, 2)))
