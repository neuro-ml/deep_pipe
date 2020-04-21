import pytest
import numpy as np

from dpipe.im.utils import identity
from dpipe.predict.shape import *

assert_eq = np.testing.assert_array_almost_equal


def test_patches_grid():
    def check_equal(**kwargs):
        assert_eq(x, patches_grid(**kwargs)(identity)(x))

    x = np.random.randn(3, 23, 20, 27) * 10
    check_equal(patch_size=10, stride=1, padding_values=0)
    check_equal(patch_size=10, stride=1, padding_values=None)
    check_equal(patch_size=10, stride=10, padding_values=0)
    check_equal(patch_size=10, stride=10, padding_values=None)

    with pytest.raises(ValueError):
        check_equal(patch_size=30, stride=1, padding_values=None)


def test_divisible_patches():
    def check_equal(**kwargs):
        assert_eq(x, divisible_shape(divisible)(patches_grid(**kwargs)(identity))(x))

    size = [80] * 3
    stride = [20] * 3
    divisible = [8] * 3
    for shape in [(373, 302, 55), (330, 252, 67)]:
        x = np.random.randn(*shape)
        check_equal(patch_size=size, stride=stride)
