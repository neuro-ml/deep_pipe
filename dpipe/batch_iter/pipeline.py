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


def make_finite(iterable: Iterable, n_iterations: int):
    """Yields n_iterations from an iterable"""
    iterable = iter(iterable)
    for _ in range(n_iterations):
        yield next(iterable)


def pipeline(transformers: Sequence, batch_size: int = None):
    """
    Wrapper around the `pdp` library which returns a regular generator
    from transformers.

    Parameters
    ----------
    transformers: Sequence
        `pdp` transformers to create a generator from
    batch_size: int, optional
        if provided `pdp.Many2One` and `combine_batches` are added to transformers
    """
    assert len(transformers) > 0

    transformers = unravel_transformers(transformers)
    if batch_size is not None:
        transformers.extend([
            Many2One(chunk_size=batch_size, buffer_size=2),
            One2One(combine_batches_, buffer_size=2),
        ])
    with Pipeline(*transformers) as p:
        yield from p


def one2one(f, pack=False, n_workers=1, buffer_size=1):
    if pack:
        f = pack_args(f)
    return One2One(f, n_workers=n_workers, buffer_size=buffer_size)


def one2many(f, pack=False, n_workers=1, buffer_size=1):
    if pack:
        f = pack_args(f)
    return One2Many(f, n_workers=n_workers, buffer_size=buffer_size)
