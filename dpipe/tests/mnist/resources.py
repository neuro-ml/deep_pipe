from os.path import join as jp, expanduser
import gzip

import numpy as np

from dpipe.dataset import Dataset


def load_mnist(folder, filename, offset):
    with gzip.open(expanduser(jp(folder, filename)), 'rb') as f:
        return np.frombuffer(f.read(), np.uint8, offset=offset)


class MNIST(Dataset):
    def __init__(self, folder):
        self.xs = load_mnist(folder, 'train-images-idx3-ubyte.gz', 16).reshape(-1, 1, 28, 28).astype('float32')
        self.ys = load_mnist(folder, 'train-labels-idx1-ubyte.gz', 8).astype('long')
        self.ids = tuple(range(len(self.xs)))
        self.n_chans_image = 1

    def load_image(self, identifier: str):
        return self.xs[int(identifier)]

    def load_label(self, identifier: str):
        return self.ys[int(identifier)]
