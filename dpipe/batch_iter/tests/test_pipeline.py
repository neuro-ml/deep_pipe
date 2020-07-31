import time
from itertools import repeat

import pytest
from pdp import StopEvent

from dpipe.batch_iter import Infinite, combine_to_arrays, Threads, Loky
from dpipe.batch_iter.pipeline import wrap_pipeline


def test_wrapper():
    size = 100
    for i, item in enumerate(wrap_pipeline(range(size), lambda x: x ** 2)):
        assert item == i ** 2

    assert i == size - 1


def test_parallel():
    size = 10
    sleep = 1

    start = time.time()
    for item in wrap_pipeline(range(size), lambda x: time.sleep(sleep)):
        pass
    delta = time.time() - start

    start = time.time()
    for item in wrap_pipeline(range(size), Threads(lambda x: time.sleep(sleep), n_workers=2)):
        pass
    faster = time.time() - start
    assert abs(faster - delta / 2) < sleep

    start = time.time()
    for item in wrap_pipeline(range(size), Loky(lambda x: time.sleep(sleep), n_workers=2)):
        pass
    faster = time.time() - start
    assert abs(faster - delta / 2) < sleep


def test_loky():
    size = 100
    for i, item in enumerate(wrap_pipeline(range(size), Loky(lambda x: x ** 2, n_workers=2))):
        assert item == i ** 2
    assert i == size - 1
    # at this point the first worker is killed
    # start a new one
    for i, item in enumerate(wrap_pipeline(range(size), Loky(lambda x: x ** 2, n_workers=2))):
        assert item == i ** 2
    assert i == size - 1

    # several workers
    for i, item in enumerate(wrap_pipeline(
            range(size),
            Loky(lambda x: x ** 2, n_workers=2),
            Loky(lambda x: x ** 2, n_workers=2))):
        assert item == i ** 4
    assert i == size - 1


def test_premature_stop():
    with wrap_pipeline(range(10), Threads(lambda x: x ** 2, n_workers=2)) as p:
        for item in p:
            break

    with wrap_pipeline(range(10), Loky(lambda x: x ** 2, n_workers=2)) as p:
        for item in p:
            break


def test_loky_exceptions():
    def raiser(x):
        raise ZeroDivisionError

    xs = range(10)
    # source
    with pytest.raises(StopEvent):
        list(wrap_pipeline(map(raiser, xs)))

    for mapper in [Threads, Loky]:
        with pytest.raises(StopEvent):
            list(wrap_pipeline(xs, mapper(raiser, n_workers=2)))


def pipeline(source, iters=1, transformers=(), batch_size=1, combiner=combine_to_arrays):
    return Infinite(source, *transformers, batch_size=batch_size, batches_per_epoch=iters, buffer_size=1,
                    combiner=combiner)


def test_elements():
    p = pipeline(repeat([1, 2]), 1)
    batch = list(p())
    assert ((1,), (2,)) == batch[0]


def test_batch_size():
    for bs in [1, 5, 10]:
        p = pipeline(repeat([1]), batch_size=bs)
        assert len(list(p())[0][0]) == bs

    with pytest.raises(TypeError):
        pipeline([1], batch_size=0.5)

    with pytest.raises(TypeError):
        pipeline([1], batch_size=[1])

    def max_three(chunk, item):
        return sum(map(len, chunk + [item])) <= 3

    with pipeline([[1, 2], [1], [1], [1], [1], [1, 2, 3], [1, 2], [1]], 4, batch_size=max_three, combiner=list) as p:
        for b in p():
            assert sum(map(len, b)) == 3


def test_iters():
    for i in [1, 5, 10, 100, 450]:
        p = pipeline(repeat([1]), i)
        assert len(list(p())) == i


def test_enter():
    p = pipeline(repeat([1]), 1)
    with p as pp:
        assert len(list(pp())) == 1


def test_multiple_iter():
    p = pipeline(repeat([1]), 1)

    assert len(list(p())) == 1
    assert len(list(p())) == 1
    assert len(list(p())) == 1

    p.close()


def test_del():
    p = pipeline(repeat([1]), 1)
    assert len(list(p())) == 1
    del p
