from itertools import repeat

import pytest

from dpipe.batch_iter import Infinite, combine_to_arrays


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
