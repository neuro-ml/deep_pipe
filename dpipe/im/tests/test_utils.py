import unittest

import numpy as np

from dpipe.layers import identity
from dpipe.im import min_max_scale, normalize
from dpipe.im.utils import apply_along_axes


class TestApplyAlongAxes(unittest.TestCase):
    def test_apply(self):
        x = np.random.rand(3, 10, 10) * 2 + 3
        np.testing.assert_array_almost_equal(
            apply_along_axes(normalize, x, axes=(1, 2), percentiles=20),
            normalize(x, percentiles=20, axes=0)
        )

        axes = (0, 2)
        y = apply_along_axes(min_max_scale, x, axes)
        np.testing.assert_array_almost_equal(y.max(axes), 1)
        np.testing.assert_array_almost_equal(y.min(axes), 0)

        np.testing.assert_array_almost_equal(apply_along_axes(identity, x, 1), x)
        np.testing.assert_array_almost_equal(apply_along_axes(identity, x, -1), x)
        np.testing.assert_array_almost_equal(apply_along_axes(identity, x, (0, 1)), x)
        np.testing.assert_array_almost_equal(apply_along_axes(identity, x, (0, 2)), x)
