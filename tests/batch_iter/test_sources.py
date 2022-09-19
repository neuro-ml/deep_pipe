from itertools import islice

import numpy as np

from dpipe.batch_iter import sample
import time

almost_eq = np.testing.assert_almost_equal


def test_sample():
    # multidimensional arrays support
    sequence = [[0], [1]]
    n = 100_000

    def get_sum(weights=None):
        return sum(x[0] for x in islice(sample(sequence, weights), n)) / n

    assert get_sum([0, 1]) == 1
    assert get_sum([1, 0]) == 0

    almost_eq(get_sum(), 0.5, decimal=2)
    almost_eq(get_sum([1, 4]), 0.8, decimal=2)


def test_sample_n_numpy():
    now = lambda: time.time()

    ids = ['0', '1', '2', '3']
    prob = [0.15, 0.35, 0.15, 0.35]
    N = 100_000

    def sample_bynumpy(sequence, weights):
        while True:
            yield sequence[np.random.choice(len(sequence), p=weights)]

    def get_dist(sequence, weights, sample_function):
        dist = {ex: 0. for ex in sequence}
        for example in islice(sample_function(sequence, weights), N):
            dist[example] += 1 / N
        return dist

    t1, sample_dist, t2 = now(), get_dist(ids, prob, sample), now()
    t3, numpy_dist, t4 = now(), get_dist(ids, prob, sample_bynumpy), now()

    for ex, p in zip(ids, prob):
        # sample gives the required probability distribution
        almost_eq(sample_dist[ex], p, decimal=2)
        almost_eq(numpy_dist[ex], p, decimal=2)
    # sample works faster than np.random.choice with the weights parameter
    np.testing.assert_array_less(t2 - t1, t4 - t3)

    t1, sample_dist, t2 = now(), get_dist(ids, None, sample), now()
    t3, numpy_dist, t4 = now(), get_dist(ids, None, sample_bynumpy), now()
    for ex in ids:
        # sample without the weights parameter gives an uniform distribution
        almost_eq(sample_dist[ex], numpy_dist[ex], decimal=2)
    # sample works faster than np.random.choice without the weights parameter
    np.testing.assert_array_less(t2 - t1, t4 - t3)
