import pdp
import numpy as np

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

    @pdp.pack_args
    def augment(x, y):
        convert = y.ndim == 2
        if convert:
            unique = np.unique(y)
            y = np.asarray([y == i for i in unique])

        x, y = spacial_augmentation(x, y, axes=[-1, -2])

        if convert:
            y = np.argmax(y, axis=0)
            #     restoring old int tensor
            if set(unique) - set(range(len(unique))):
                for i, val in enumerate(unique):
                    y[y == i] = val

        return x, y

    return pdp.Pipeline(
        pdp.Source(slicer(), buffer_size=5),
        pdp.One2One(augment, buffer_size=20),
        pdp.Many2One(chunk_size=batch_size, buffer_size=2),
        pdp.One2One(pdp.combine_batches, buffer_size=3),
    )
