import sys

import pytest
import numpy as np

from dpipe.im.utils import identity
from dpipe.predict.shape import *
from dpipe.itertools import pmap

assert_eq = np.testing.assert_array_almost_equal


@pytest.fixture(params=[1, 2, 3, 4])
def batch_size(request):
    return request.param


@pytest.fixture(params=[False, True])
def use_torch(request):
    return request.param


@pytest.fixture(params=[False, True])
def async_predict(request):
    return request.param


def test_patches_grid(use_torch, async_predict, batch_size):
    def check_equal(**kwargs):
        predict = patches_grid(**kwargs, use_torch=use_torch, async_predict=async_predict, axis=-1, batch_size=batch_size)(lambda x: x + 1)
        predict = add_extract_dims(1)(predict)
        assert_eq(x + 1, predict(x))

    x = np.random.randn(3, 23, 20, 27) * 10
    check_equal(patch_size=10, stride=1, padding_values=0)
    check_equal(patch_size=10, stride=1, padding_values=None)
    check_equal(patch_size=10, stride=10, padding_values=0)

    with pytest.raises(ValueError):
        check_equal(patch_size=10, stride=10, padding_values=None)

    check_equal(patch_size=30, stride=1, padding_values=0)
    with pytest.raises(ValueError):
        check_equal(patch_size=30, stride=1, padding_values=None)

    check_equal(patch_size=9, stride=9, padding_values=None)
    check_equal(patch_size=15, stride=12, padding_values=None)


def test_divisible_patches():
    def check_equal(**kwargs):
        assert_eq(x, divisible_shape(divisible)(patches_grid(**kwargs)(identity))(x))

    size = [80] * 3
    stride = [20] * 3
    divisible = [8] * 3
    for shape in [(373, 302, 55), (330, 252, 67)]:
        x = np.random.randn(*shape)
        check_equal(patch_size=size, stride=stride)
