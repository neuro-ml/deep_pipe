import functools

import pdp

from dpipe.config import register
from dpipe.medim.slices import iterate_slices
from dpipe.medim.utils import load_by_ids
from dpipe.medim.augmentation import spacial_augmentation


@register()
def slices(ids, load_x, load_y, batch_size, *, shuffle, axis=-1, slices=1,
           pad=0, concatenate=None):
    def slicer():
        for x, y in load_by_ids(load_x, load_y, ids, shuffle):
            for x_slice, y_slice in iterate_slices(
                    x, y, axis=axis, slices=slices, pad=pad,
                    concatenate=concatenate):
                if y_slice.any():
                    yield x_slice, y_slice

    return pdp.Pipeline(
        pdp.Source(slicer(), buffer_size=5),
        pdp.Many2One(chunk_size=batch_size, buffer_size=2),
        pdp.One2One(pdp.combine_batches, buffer_size=3)
    )


@register()
def slices_augmented(ids, load_x, load_y, batch_size, *, shuffle, axis=-1,
                     slices=1, pad=0, concatenate=None):
    def slicer():
        for x, y in load_by_ids(load_x, load_y, ids, shuffle):
            for x_slice, y_slice in iterate_slices(
                    x, y, axis=axis, slices=slices, pad=pad,
                    concatenate=concatenate):
                if y_slice.any():
                    yield x_slice, y_slice

    augment = pdp.pack_args(functools.partial(spacial_augmentation, axes=[-1, -2]))

    return pdp.Pipeline(
        pdp.Source(slicer(), buffer_size=5),
        pdp.One2One(augment, buffer_size=20),
        pdp.Many2One(chunk_size=batch_size, buffer_size=2),
        pdp.One2One(pdp.combine_batches, buffer_size=3),
    )
