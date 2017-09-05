import numpy as np


def segm_prob2msegm(x, dataset):
    return dataset.segm2msegm(np.argmax(x, axis=0))
