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
def stream(request):
    return request.param


def test_patches_grid(stream):
    def check_equal(**kwargs):
        assert_eq(x, patches_grid(**kwargs, stream=stream, axis=-1)(identity)(x))

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


def test_divisible_patches(stream):
    def check_equal(**kwargs):
        assert_eq(x, divisible_shape(divisible)(patches_grid(**kwargs, stream=stream)(identity))(x))

    size = [80] * 3
    stride = [20] * 3
    divisible = [8] * 3
    for shape in [(373, 302, 55), (330, 252, 67)]:
        x = np.random.randn(*shape)
        check_equal(patch_size=size, stride=stride)


@pytest.mark.skipif(sys.version_info < (3, 7), reason='Requires python3.7 or higher.')
def test_batched_patches_grid(batch_size):
    from more_itertools import batched
    from itertools import chain

    def patch_predict(patch):
        return patch + 1

    def stream_predict(patches_generator):
        return chain.from_iterable(pmap(patch_predict, map(np.array, batched(patches_generator, batch_size))))

    x = np.random.randn(3, 23, 20, 27) * 10

    assert_eq(x + 1, patches_grid(patch_size=(6, 8, 9), stride=(4, 3, 2), stream=True, axis=(-1, -2, -3))(stream_predict)(x))
