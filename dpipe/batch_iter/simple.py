from typing import Sequence

import pdp
from pdp import product_generator

from dpipe.medim.utils import load_by_ids


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
    return product_generator(
        pdp.Source(load_by_ids(load_x, load_y, ids, shuffle), buffer_size=30),
        pdp.Many2One(chunk_size=batch_size, buffer_size=2),
        pdp.One2One(pdp.combine_batches, buffer_size=3)
    )
