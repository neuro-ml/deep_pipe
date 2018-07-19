import random

import pdp


def make_source_sequence(ids):
    return pdp.Source([{'id': i} for i in ids], buffer_size=3)


def make_source_random(ids):
    return pdp.Source(iter(lambda: {'id': random.choice(ids)}, None), buffer_size=3)


def cache_function(func):
    cache = {}

    def cached_function(x):
        if x['id'] in cache:
            y = cache[x['id']].copy()
        else:
            y = func(x.copy())
            cache[x['id']] = y

        return y

    return cached_function


def make_block_load_x_y(load_x, load_y, *, buffer_size):
    @cache_function
    def add_x_y(o):
        o['x'] = load_x(o['id'])
        o['y'] = load_y(o['id'])
        return o

    return pdp.One2One(add_x_y, buffer_size=buffer_size)


def make_batch_blocks(batch_size, buffer_size):
    return (pdp.Many2One(chunk_size=batch_size, buffer_size=3),
            pdp.One2One(pdp.combine_batches, buffer_size=buffer_size))
