from typing import Iterable

import numpy as np
from dpipe.medim.preprocessing import pad_to_shape


def make_batches(iterable: Iterable, batch_size: int):
    assert batch_size > 0
    buffers = None
    for i in iterable:
        if buffers is None:
            buffers = [[] for _ in range(len(i))]
        assert len(buffers) == len(i)
        for buffer, val in zip(buffers, i):
            buffer.append(val)

        if len(buffer) == batch_size:
            yield [np.asarray(buffer) for buffer in buffers]
            buffers = None


def combine_batches_even(inputs):
    result = []
    for o in zip(*inputs):
        shapes = np.array([x.shape for x in o])
        padded = [pad_to_shape(x, shapes.max(axis=0)) for x in o]
        result.append(np.array(padded))
    return result
