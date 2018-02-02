import functools
import unittest

import numpy as np

from dpipe.batch_iter.simple import simple
from dpipe.batch_iter.slices import slices


class TestSimple(unittest.TestCase):
    def test_simple(self):
        x, y = np.random.randn(2, 100)
        ids = list(range(len(x)))

        with simple(ids, x.__getitem__, y.__getitem__, 1) as pipeline:
            for i, (xs, ys) in enumerate(pipeline):
                self.assertEqual(xs, x[[i]])
                self.assertEqual(ys, y[[i]])


class TestSlices(unittest.TestCase):
    def setUp(self):
        self.x, self.y = np.random.randn(2, 1, 10, 10)
        self.ids = list(range(len(self.x)))
        self.iterator = functools.partial(slices, ids=self.ids, load_x=self.x.__getitem__,
                                          load_y=self.y.__getitem__, shuffle=False)

    def test_basic(self):
        with self.iterator(batch_size=1) as pipeline:
            for i, (xs, ys) in enumerate(pipeline):
                np.testing.assert_array_equal(xs, self.x[..., i])
                np.testing.assert_array_equal(ys, self.y[..., i])

    def test_batch(self):
        batch_size = 3

        with self.iterator(batch_size=batch_size) as pipeline:
            for i, (xs, ys) in enumerate(pipeline):
                i *= batch_size
                np.testing.assert_array_equal(xs, self.x[0, ..., i:i + batch_size].T)
                np.testing.assert_array_equal(ys, self.y[0, ..., i:i + batch_size].T)

    def test_slices(self):
        num_slices = 3

        with self.iterator(batch_size=1, slices=num_slices) as pipeline:
            for i, (xs, ys) in enumerate(pipeline):
                np.testing.assert_array_equal(xs, self.x[..., i:i + num_slices])
                np.testing.assert_array_equal(ys, self.y[..., i:i + num_slices])

    def test_slices_concat(self):
        num_slices = 3

        with self.iterator(batch_size=1, slices=num_slices, concatenate=0) as pipeline:
            for i, (xs, ys) in enumerate(pipeline):
                # basically this crazy reshape stuff does the same as concatenating along the first axis
                np.testing.assert_array_equal(xs, self.x[0, ..., i:i + num_slices].T.reshape(1, -1))
                np.testing.assert_array_equal(ys, self.y[0, ..., i:i + num_slices].T.reshape(1, -1))
