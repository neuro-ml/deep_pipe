from abc import abstractmethod, ABC
from itertools import islice
from contextlib import contextmanager

import numpy as np

__all__ = ['BatchIter', 'make_batch_iter_from_finite', 'make_batch_iter_from_infinite']


@contextmanager
def build_contextmanager(o):
    yield o


def maybe_build_contextmanager(o):
    """If input has no context manager on it's own, turn it into object with empty context manager."""
    if hasattr(o, '__enter__'):
        return o
    else:
        return build_contextmanager(o)


class BatchIter(ABC):
    """Interface for training functions, that unifies interface for infinite and finite batch generators.

    Examples
    --------
    >>> # BatchIter should be created from one of the implementations
    >>> batch_iter : BatchIter = None
    >>> with batch_iter:
    >>>     for epoch in range(10):
    >>>         for x, y in batch_iter():
    >>>             pass

    """

    @abstractmethod
    def __call__(self):
        pass

    def __enter__(self):
        return self

    @abstractmethod
    def __exit__(self, exc_type, exc_val, exc_tb):
        pass


class BatchIterRepeater(BatchIter):
    def __init__(self, get_batch_iter):
        self.get_batch_iter = get_batch_iter
        self.batch_iter = None

    def __call__(self):
        assert self.batch_iter is None, 'Iterator has already been open'
        self.batch_iter = maybe_build_contextmanager(self.get_batch_iter())
        with self.batch_iter:
            yield from self.batch_iter
        self.batch_iter = None

    def __exit__(self, exc_type, exc_val, exc_tb):
        # We need this part in case during batch iteration there was an error in the main thread
        # This part could be dangerous. If there was an error in the main thread, self.batch_iter.__exit__ will
        # be called twice. pdp backend just ignores second exit, so, no problem here.
        if self.batch_iter is not None:
            result = self.batch_iter.__exit__(exc_type, exc_val, exc_tb)
            self.batch_iter = None
            return result
        else:
            return False


class BatchIterSlicer(BatchIter):
    def __init__(self, get_batch_iter, n_iters_per_epoch):
        self.infinite_batch_iter = maybe_build_contextmanager(get_batch_iter())
        self.n_iters_per_epoch = n_iters_per_epoch

    def __call__(self):
        yield from islice(self.infinite_batch_iter, self.n_iters_per_epoch)

    def __enter__(self):
        self.infinite_batch_iter.__enter__()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        return self.infinite_batch_iter.__exit__(exc_type, exc_val, exc_tb)


@np.deprecate(message='Use `dpipe.batch_iter.base.BatchIterRepeater` instead.')  # 04.06.19
def make_batch_iter_from_finite(get_batch_iter):
    return BatchIterRepeater(get_batch_iter)


@np.deprecate(message='Use `dpipe.batch_iter.base.BatchIterSlicer` instead.')  # 04.06.19
def make_batch_iter_from_infinite(get_batch_iter, n_iters_per_epoch):
    return BatchIterSlicer(get_batch_iter, n_iters_per_epoch)
