import numpy as np

from collections import Counter

from dpipe.batch_iter import expiration_pool

def test_expiration_pool_repetitions():
    iterable = list(range(10))
    pool_size = 2
    repetitions = 4
    
    sampled = []
    
    for x in expiration_pool(iterable, pool_size, repetitions):
        sampled.append(x)
        
    for value in Counter(sampled).values():
        assert value == repetitions
    

def test_expiration_pool_size():
    iterable = list(range(100))
    pool_size = 10
    repetitions = 2
    
    for _ in range(10):
        sampled = []
        exp_pool = expiration_pool(iterable, pool_size, repetitions)
        for i in range(10):
            sampled.append(next(exp_pool))
            
        assert max(sampled) < 10