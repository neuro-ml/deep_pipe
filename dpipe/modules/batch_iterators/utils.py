import numpy as np


def combine_batch(inputs):
    return [np.array(o) for o in zip(*inputs)]