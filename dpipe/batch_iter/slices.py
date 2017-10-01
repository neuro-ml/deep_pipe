import dpipe.externals.pdp.pdp as pdp
from dpipe.config import register
from dpipe.medim.slices import iterate_slices
from dpipe.medim.utils import load_by_ids


@register()
def slices(ids, load_x, load_y, batch_size, *, shuffle, axis=-1, slices=1, pad=0,
           concatenate=None):
    def slicer():
        for x, y in load_by_ids(load_x, load_y, ids, shuffle):
            yield from iterate_slices(x, y, axis=axis, slices=slices, pad=pad,
                                      concatenate=concatenate)

    return pdp.Pipeline(
        pdp.Source(slicer(), buffer_size=30),
        pdp.Chunker(chunk_size=batch_size, buffer_size=2),
        pdp.LambdaTransformer(pdp.combine_batches, buffer_size=3)
    )
