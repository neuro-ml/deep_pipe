from typing import Sequence

import pdp

from dpipe.medim.utils import load_by_ids


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
    return pdp.Pipeline(
        pdp.Source(load_by_ids(load_x, load_y, ids=ids, shuffle=shuffle), buffer_size=30),
        pdp.Many2One(chunk_size=batch_size, buffer_size=2),
        pdp.One2One(pdp.combine_batches, buffer_size=3)
    )
