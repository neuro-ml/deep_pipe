import os
import itertools
import shutil
import unittest
from pathlib import Path
from urllib.request import urlretrieve

from dpipe.config import get_resource_manager
from dpipe.medim.io import load_json


class TestMNIST(unittest.TestCase):
    sortTestMethodsUsing = None

    base_path = Path('~/tests/MNIST').expanduser()
    base_rm = get_resource_manager('dpipe/tests/mnist/setup.config')
    experiment_path = base_rm.experiment_path

    def setUp(self):
        def download(filename):
            path = self.base_path / filename
            if not path.exists():
                os.makedirs(self.base_path, exist_ok=True)
                urlretrieve('http://yann.lecun.com/exdb/mnist/' + filename, path)

        download('train-images-idx3-ubyte.gz')
        download('train-labels-idx1-ubyte.gz')

    def test_build_experiment(self):
        self.base_rm.build_experiment

        names = {p.name for p in self.experiment_path.iterdir()}
        self.assertSetEqual({'resources.config'} | {f'experiment_{i}' for i in range(len(names) - 1)}, names)

    def test_ids(self):
        ids = set(self.base_rm.dataset.ids)
        test_ids = []

        for exp in self.experiment_path.iterdir():
            if exp.is_dir():
                train, val, test = [set(load_json(exp / f'{name}_ids.json')) for name in ['train', 'val', 'test']]
                test_ids.append(test)

                self.assertFalse(train & val)
                self.assertFalse(train & test)
                self.assertFalse(val & test)
                self.assertSetEqual(ids, train | val | test)

        for first, second in itertools.permutations(test_ids, 2):
            self.assertFalse(first & second)

    def tearDown(self):
        if self.experiment_path.exists():
            shutil.rmtree(self.experiment_path)
