import unittest
from unittest.mock import Mock
import random
import string
import numpy as np
from .monte_carlo import monte_carlo


def random_string(string_len):
    return ''.join(random.choices(string.ascii_uppercase + string.digits, k=string_len))


class TestMonteCarloSplitter(unittest.TestCase):
    def setUp(self):
        self.ids_num = 100
        ids = np.array([random_string(10) for _ in range(self.ids_num)])
        self.dataset = Mock()
        self.dataset.patient_ids = ids

    def test_successful_split(self):
        n_splits = 5

        train_fraction = 0.8
        val_fraction = 0.1
        splits = monte_carlo(
            self.dataset, train_fraction=train_fraction, val_fraction=val_fraction, n_splits=n_splits
        )

        assert(len(splits) == n_splits)

        for (train, val, test) in splits:
            assert(len(train) == int(round(self.ids_num * train_fraction)))
            assert(len(val) == int(round(self.ids_num * val_fraction)))
            assert(len(test) == int(round(self.ids_num * (1 - val_fraction - train_fraction))))

    def test_bad_params_value_error(self):
        with self.assertRaises(ValueError):
            monte_carlo(self.dataset, train_fraction=0.9, val_fraction=0.5, n_splits=5)
