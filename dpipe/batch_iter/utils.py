from typing import Iterable

import numpy as np


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
