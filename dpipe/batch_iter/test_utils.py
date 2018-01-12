import unittest

import numpy as np

from dpipe.batch_iter.utils import make_batches


class TestMakeBatches(unittest.TestCase):
    def test_basic(self):
        x, y = np.random.rand(2, 20)
        batch_size = 3
        for i, (xs, ys) in enumerate(make_batches(zip(x, y), batch_size=batch_size)):
            i *= batch_size
            np.testing.assert_array_equal(xs, x[i:i + batch_size])
            np.testing.assert_array_equal(ys, y[i:i + batch_size])
