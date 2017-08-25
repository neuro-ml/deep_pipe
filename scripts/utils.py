import os

import numpy as np


def load_by_id(path, identifier):
    return np.load(os.path.join(path, f'{identifier}.npy'))
