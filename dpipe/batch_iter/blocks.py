import random
import functools

import pdp


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


class ExpirationPool:
    def __init__(self, expiration_time, pool_size):
        self.pool_size = pool_size
        self.expiration_time = expiration_time

        self.data = []
        self.expiration_timer = []

    def is_full(self):
        return len(self.data) == self.pool_size

    def put(self, value):
        assert not self.is_full()
        self.data.append(value)
        self.expiration_timer.append(self.expiration_time)

    def draw(self):
        assert self.is_full()
        i = random.randint(0, self.pool_size - 1)
        value = self.data[i]
        self.expiration_timer[i] -= 1

        assert self.expiration_timer[i] >= 0
        if self.expiration_timer[i] == 0:
            del self.data[i]
            del self.expiration_timer[i]

        return value
