from itertools import islice

import numpy as np

from dpipe.batch_iter import sample

almost_eq = np.testing.assert_almost_equal


def test_sample():
    # multidimensional arrays support
    sequence = [[0], [1]]
    n = 10000

    def get_sum(weights=None):
        return sum(x[0] for x in islice(sample(sequence, weights), n)) / n

    assert get_sum([0, 1]) == 1
    assert get_sum([1, 0]) == 0

    almost_eq(get_sum(), 0.5, decimal=2)
    almost_eq(get_sum([1, 4]), 0.8, decimal=2)
