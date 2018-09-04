import numpy as np

from dpipe.medim.itertools import lmap
from dpipe.medim.preprocessing import pad_to_shape


def pad_batch_equal(batch):
    max_shapes = np.max(lmap(np.shape, batch), axis=0)
    return np.array([pad_to_shape(x, max_shapes, padding_values=np.min(x)) for x in batch])


def pad_batches_even(batches):
    return tuple(map(pad_batch_equal, batches))
