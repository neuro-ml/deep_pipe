import numpy as np

from dpipe.im.dist import *

eq = np.testing.assert_array_almost_equal


def test_uniform():
    dist = np.ones(10)
    dist /= dist.sum(0, keepdims=True)
    eq(expectation(dist, 0, order=0), 1)
    eq(expectation(dist, 0, order=1), 10 ** 2 / 2 / 10)


def test_constant():
    dist = np.random.uniform(0, 1, size=10)
    dist /= dist.sum(0, keepdims=True)
    eq(expectation(dist, 0, np.ones), 0)
    eq(expectation(dist, 0, np.arange), 1)


def test_conditional():
    dist = np.ones((30, 30, 40))
    dist /= dist.sum((0, 1), keepdims=True)
    centers = np.full((2, 40), 15)

    eq(np.stack(marginal_expectation(dist, (0, 1), polynomial)), centers)
