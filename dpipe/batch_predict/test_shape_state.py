import unittest

import numpy as np

from .shape_state import PatchDividable


class TestPatchDividable(unittest.TestCase):
    def test_predict(self):
        x = np.random.rand(3, 11, 11, 11)
        y = PatchDividable(3).predict(x, predict_fn=lambda x: x)
        np.testing.assert_array_equal(x, y)

        y, loss = PatchDividable(3).validate(x, x, validate_fn=lambda x, y: (x, 1))
        np.testing.assert_array_equal(x, y)