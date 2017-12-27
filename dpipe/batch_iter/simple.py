from typing import Sequence

from dpipe.batch_iter.slices import combine_batches_even
from dpipe.medim.utils import load_by_ids

import pdp
from .pipeline import pipeline


def simple(ids: Sequence, load_x: callable, load_y: callable, batch_size: int, *, shuffle: bool = False):
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
    return pipeline([
        pdp.Source(load_by_ids(load_x, load_y, ids, shuffle), buffer_size=30),
        pdp.Many2One(chunk_size=batch_size, buffer_size=2),
        pdp.One2One(combine_batches_even, buffer_size=3)
    ])
