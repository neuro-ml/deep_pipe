from typing import Sequence

import numpy as np


def segm_prob2msegm(x, segm2msegm):
    return segm2msegm(np.argmax(x, axis=0))


def binarize(x, thresholds):
    assert len(x) == len(thresholds)
    thresholds = np.asarray(thresholds)
    return x > thresholds[:, None, None, None]


def chain(functions: Sequence):
    def wrapped(*args, **kwargs):
        x = functions[-1](*args, **kwargs)
        for func in reversed(functions[:-1]):
            x = func(x)
        return x

    return wrapped
