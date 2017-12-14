import numpy as np


def segm_prob2msegm(x, dataset):
    return dataset.segm2msegm(np.argmax(x, axis=0))


def binarize(x, thresholds):
    assert len(x) == len(thresholds)
    thresholds = np.asarray(thresholds)
    return x > thresholds[:, None, None, None]
