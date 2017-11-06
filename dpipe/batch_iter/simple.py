from dpipe.batch_iter.slices import combine_batches_even
from dpipe.config import register
from dpipe.medim.utils import load_by_ids

import pdp


@register()
def simple(ids, load_x, load_y, batch_size, *, shuffle=False):
    return pdp.Pipeline(
        pdp.Source(load_by_ids(load_x, load_y, ids, shuffle), buffer_size=30),
        pdp.Many2One(chunk_size=batch_size, buffer_size=2),
        pdp.One2One(combine_batches_even, buffer_size=3)
    )
