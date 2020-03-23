import itertools
import shutil
import unittest
from pathlib import Path

import pytest
from resource_manager import read_config

from dpipe.io import load


@pytest.mark.integration
class TestMNIST(unittest.TestCase):
    # TODO: use a temp dir
    base_path = Path('~/tests/MNIST').expanduser()
    experiment_path = base_path / 'exp'
    config_path = 'dpipe/tests/mnist/setup.config'
    config = read_config(config_path)

    @classmethod
    def tearDownClass(cls):
        if cls.experiment_path.exists():
            shutil.rmtree(cls.experiment_path)

    def test_pipeline(self):
        # build
        self.config.layout.build(self.config_path, self.experiment_path)
        names = {p.name for p in self.experiment_path.iterdir()}
        assert {'resources.config'} | {f'experiment_{i}' for i in range(len(names) - 1)} == names

        # split
        ids = set(self.config.ids)
        test_ids = []

        for exp in self.experiment_path.iterdir():
            if exp.is_dir():
                train, val, test = [set(load(exp / f'{name}_ids.json')) for name in ['train', 'val', 'test']]
                test_ids.append(test)

                assert not train & val
                assert not train & test
                assert not val & test
                assert ids == train | val | test

        for first, second in itertools.permutations(test_ids, 2):
            assert not first & second

        # training
        self.config.layout.run(self.experiment_path / 'resources.config', folds=[0])
        assert load(self.experiment_path / 'experiment_0/test_metrics/accuracy.json') >= .95
