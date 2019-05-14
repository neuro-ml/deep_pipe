import unittest
from itertools import repeat

from dpipe.batch_iter.pipeline import Infinite


class TestSimple(unittest.TestCase):
    @staticmethod
    def pipeline(source, iters=1, transformers=()):
        return Infinite(source, *transformers, batch_size=1, batches_per_epoch=iters, buffer_size=1)

    def test_elements(self):
        p = self.pipeline(repeat([1, 2]), 1)
        batch = list(p())
        self.assertTupleEqual(((1,), (2,)), batch[0])

    def test_iters(self):
        for i in [1, 5, 10, 100, 450]:
            p = self.pipeline(repeat([1]), i)
            self.assertEqual(len(list(p())), i)

    def test_enter(self):
        p = self.pipeline(repeat([1]), 1)
        with p:
            self.assertEqual(len(list(p())), 1)

    def test_multiple_iter(self):
        p = self.pipeline(repeat([1]), 1)

        self.assertEqual(len(list(p())), 1)
        self.assertEqual(len(list(p())), 1)
        self.assertEqual(len(list(p())), 1)

        p.close()

    def test_del(self):
        p = self.pipeline(repeat([1]), 1)
        self.assertEqual(len(list(p())), 1)
        del p
