import numpy as np

from dpipe.config import register


@register(module_type='transform')
def segm_prob2msegm(x, dataset):
    return dataset.segm2msegm(np.argmax(x, axis=0))
