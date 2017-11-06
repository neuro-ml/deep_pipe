import gzip
import os
from typing import Sequence
from urllib.request import urlretrieve

import numpy as np

from dpipe.config import register
from dpipe.dataset import DataSet


@register()
class MNIST(DataSet):
    def load_x(self, identifier: str):
        return self.xs[identifier]

    def load_y(self, identifier: str):
        return self.ys[identifier]

    @property
    def ids(self) -> Sequence[str]:
        return list(range(len(self.xs)))

    def __init__(self, data_path):
        self.data_path = data_path
        self.xs = self._get_or_load('train-images-idx3-ubyte.gz', 16).reshape(-1, 1, 28, 28).astype('float32')
        self.ys = self._get_or_load('train-labels-idx1-ubyte.gz', 8).astype('int')

    def _get_or_load(self, file, offset):
        path = os.path.join(self.data_path, file)

        if not os.path.exists(path):
            urlretrieve(f'http://yann.lecun.com/exdb/mnist/{file}', path)
        with gzip.open(path, 'rb') as f:
            data = np.frombuffer(f.read(), np.uint8, offset=offset)
        return data
