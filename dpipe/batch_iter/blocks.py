import random
import functools

import pdp


def make_source_sequence(ids):
    return pdp.Source([{'id': i} for i in ids], buffer_size=3)


def make_source_random(ids):
    return pdp.Source(iter(lambda: {'id': random.choice(ids)}, None), buffer_size=3)


def cache_block_function(func):
    cache = {}

    @functools.wraps(func)
    def cached_function(o):
        i = o['id']
        if i in cache:
            y = cache[i].copy()
        else:
            y = func(o.copy())
            cache[i] = y

        return y

    return cached_function


def make_block_load_x_y(load_x, load_y, *, buffer_size):
    @cache_block_function
    def add_x_y(o):
        return {**o, 'x': load_x(o['id']), 'y': load_y(o['id'])}

    return pdp.One2One(add_x_y, buffer_size=buffer_size)


def make_batch_blocks(batch_size, buffer_size):
    return (pdp.Many2One(chunk_size=batch_size, buffer_size=3),
            pdp.One2One(pdp.combine_batches, buffer_size=buffer_size))
