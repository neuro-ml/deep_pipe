import os

import numpy as np


def read_lines(path):
    with open(path, 'r') as file:
        return [l for l in map(lambda x: x.strip(), file) if l != '']


def load_by_id(path, id):
    return np.load(os.path.join(path, f'{id}.npy'))
