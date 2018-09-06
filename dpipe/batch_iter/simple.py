from typing import Sequence, Callable

import pdp

from dpipe.medim.utils import load_by_ids
from dpipe.batch_iter import make_batch_iter_from_finite
from .blocks import make_batch_blocks


def load_combine(ids: Sequence, load_x: Callable, load_y: Callable, batch_size: int, *, shuffle: bool = False):
    """
    A simple batch iterator that loads the data and packs it into batches of size ``batch_size``.

    Parameters
    ----------
    ids
    load_x: Callable(id)
    load_y: Callable(id)
    batch_size
    shuffle
        whether to shuffle the ids before yielding batches.
    """

    def pipeline():
        return pdp.Pipeline(
            pdp.Source(load_by_ids(load_x, load_y, ids=ids, shuffle=shuffle), buffer_size=30),
            *make_batch_blocks(batch_size=batch_size, buffer_size=3)
        )

    return make_batch_iter_from_finite(pipeline)
