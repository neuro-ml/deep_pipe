from typing import Sequence

import pdp

from dpipe.medim.utils import load_by_ids
from dpipe.train.batch_iter import make_batch_iter_from_finite
from .blocks import make_batch_blocks


def load_combine(ids: Sequence, load_x: callable, load_y: callable, batch_size: int, *, shuffle: bool = False):
    """
    A simple batch iterator that loads the data and packs it into batches.

    Parameters
    ----------
    ids: Sequence
    load_x: callable(id)
    load_y: callable(id)
    batch_size: int
    shuffle: bool, optional
        whether to shuffle the ids before yielding batches.

    Yields
    ------
    batches of size `batch_size`
    """

    def pipeline():
        return pdp.Pipeline(
            pdp.Source(load_by_ids(load_x, load_y, ids=ids, shuffle=shuffle), buffer_size=30),
            *make_batch_blocks(batch_size=batch_size, buffer_size=3)
        )

    return make_batch_iter_from_finite(pipeline)
