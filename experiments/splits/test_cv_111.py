import unittest
import random

from .cv_111 import cv_111


class TestSplits(unittest.TestCase):
    def test_split(self):
        for i in range(10):
            total_size = random.randint(40, 500)
            val_size = random.randint(2, 40)
            n_splits = random.randint(2, 40)
            split = cv_111(total_size, val_size, n_splits)

            for j in range(13):
                sample = cv_111(total_size, val_size, n_splits)
                self.assertTrue(sample == split)
