"""Contains several wrappers around the `pdp` library"""

from typing import Sequence, Iterable

from pdp import Pipeline, One2One, Many2One, Source, pack_args, One2Many
import pdp

import numpy as np


def unravel_transformers(sequence):
    result = []
    for value in sequence:
        if isinstance(value, pdp.interface.TransformerDescription) or isinstance(value, Source):
            result.append(value)
        else:
            result.extend(unravel_transformers(value))
    return result


def combine_batches_(inputs):
    return [np.asarray(o) for o in zip(*inputs)]


def enter_and_yield(context_manager):
    with context_manager:
        yield from context_manager


def make_finite(iterable: Iterable, iterations: int):
    iterable = iter(iterable)
    
    for _ in range(iterations):
        yield next(iterable)

            
def pipeline(transformers: Sequence, batch_size: int = None):
    assert len(transformers) > 0

    transformers = unravel_transformers(transformers)
    if batch_size is not None:
        transformers.extend([
            Many2One(chunk_size=batch_size, buffer_size=2),
            One2One(combine_batches_, buffer_size=2),
        ])
    yield from enter_and_yield(Pipeline(*transformers))


def one2one(f, pack=False, n_workers=1, buffer_size=1):
    if pack:
        f = pack_args(f)
    return One2One(f, n_workers=n_workers, buffer_size=buffer_size)


def one2many(f, pack=False, n_workers=1, buffer_size=1):
    if pack:
        f = pack_args(f)
    return One2Many(f, n_workers=n_workers, buffer_size=buffer_size)
