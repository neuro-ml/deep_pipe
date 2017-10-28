from dpipe.config import register
from dpipe.medim.utils import load_by_ids

import pdp


@register()
def simple(ids, load_x, load_y, batch_size, *, shuffle=False):
    return pdp.Pipeline(
        pdp.Source(load_by_ids(load_x, load_y, ids, shuffle), buffer_size=30),
        pdp.Many2One(chunk_size=batch_size, buffer_size=2),
        pdp.One2One(pdp.combine_batches, buffer_size=3)
    )
