import unittest

import numpy as np

from dpipe.batch_iter.simple import load_combine


class TestSimple(unittest.TestCase):
    def test_simple(self):
        x, y = np.random.randn(2, 100)
        ids = list(range(len(x)))

        with load_combine(ids, x.__getitem__, y.__getitem__, 1) as pipeline:
            for i, (xs, ys) in enumerate(pipeline):
                self.assertEqual(xs, x[[i]])
                self.assertEqual(ys, y[[i]])
